//! wgpu-based GPU backend implementation.

use super::{GpuBackend, GpuMemoryInfo};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU backend using wgpu for cross-platform compute (Vulkan/Metal/DX12).
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    matmul_pipeline: wgpu::ComputePipeline,
    distance_pipeline: wgpu::ComputePipeline,
    relu_pipeline: wgpu::ComputePipeline,
    sigmoid_pipeline: wgpu::ComputePipeline,
    softmax_pipeline: wgpu::ComputePipeline,
    row_reduce_pipeline: wgpu::ComputePipeline,
    bias_add_pipeline: wgpu::ComputePipeline,
    relu_grad_pipeline: wgpu::ComputePipeline,
    sigmoid_grad_pipeline: wgpu::ComputePipeline,
    elementwise_mul_pipeline: wgpu::ComputePipeline,
    memory_info: GpuMemoryInfo,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend").finish()
    }
}

impl WgpuBackend {
    /// Try to create a new GPU backend. Returns None if no GPU adapter is available.
    pub fn try_new() -> Option<Self> {
        pollster::block_on(Self::try_new_async())
    }

    /// Create a new GPU backend, returning error if no GPU available.
    pub fn new() -> Result<Self> {
        Self::try_new().ok_or_else(|| FerroError::invalid_input("No GPU adapter available"))
    }

    async fn try_new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ferroml-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let limits = device.limits();
        let memory_info = GpuMemoryInfo {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        };

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(super::kernels::MATMUL_SHADER.into()),
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: None,
            module: &matmul_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let distance_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distance_shader"),
            source: wgpu::ShaderSource::Wgsl(super::kernels::DISTANCE_SHADER.into()),
        });

        let distance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("distance_pipeline"),
            layout: None,
            module: &distance_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Helper to create a pipeline from shader source
        let make_pipeline =
            |source: &str, shader_label: &str, pipeline_label: &str| -> wgpu::ComputePipeline {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(shader_label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(pipeline_label),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

        let relu_pipeline =
            make_pipeline(super::kernels::RELU_SHADER, "relu_shader", "relu_pipeline");
        let sigmoid_pipeline = make_pipeline(
            super::kernels::SIGMOID_SHADER,
            "sigmoid_shader",
            "sigmoid_pipeline",
        );
        let softmax_pipeline = make_pipeline(
            super::kernels::SOFTMAX_SHADER,
            "softmax_shader",
            "softmax_pipeline",
        );
        let row_reduce_pipeline = make_pipeline(
            super::kernels::ROW_REDUCE_SHADER,
            "row_reduce_shader",
            "row_reduce_pipeline",
        );
        let bias_add_pipeline = make_pipeline(
            super::kernels::BIAS_ADD_SHADER,
            "bias_add_shader",
            "bias_add_pipeline",
        );
        let relu_grad_pipeline = make_pipeline(
            super::kernels::RELU_GRAD_SHADER,
            "relu_grad_shader",
            "relu_grad_pipeline",
        );
        let sigmoid_grad_pipeline = make_pipeline(
            super::kernels::SIGMOID_GRAD_SHADER,
            "sigmoid_grad_shader",
            "sigmoid_grad_pipeline",
        );
        let elementwise_mul_pipeline = make_pipeline(
            super::kernels::ELEMENTWISE_MUL_SHADER,
            "elementwise_mul_shader",
            "elementwise_mul_pipeline",
        );

        Some(Self {
            device,
            queue,
            matmul_pipeline,
            distance_pipeline,
            relu_pipeline,
            sigmoid_pipeline,
            softmax_pipeline,
            row_reduce_pipeline,
            bias_add_pipeline,
            relu_grad_pipeline,
            sigmoid_grad_pipeline,
            elementwise_mul_pipeline,
            memory_info,
        })
    }

    /// Upload f64 data to GPU as f32 buffer.
    fn create_storage_buffer(&self, data: &[f64], label: &str) -> wgpu::Buffer {
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let bytes = bytemuck::cast_slice(&f32_data);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create an output buffer for GPU results.
    fn create_output_buffer(&self, size: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for reading results back to CPU.
    fn create_staging_buffer(&self, size: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer with dimension data.
    fn create_uniform_buffer(&self, dims: &[u32; 4], label: &str) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice(dims);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Returns GPU memory information.
    pub fn memory_info(&self) -> &GpuMemoryInfo {
        &self.memory_info
    }

    /// Pre-flight check: verify buffer size does not exceed GPU storage buffer binding limit.
    fn check_buffer_size(&self, n_elements: usize, label: &str) -> Result<()> {
        let byte_size = (n_elements * std::mem::size_of::<f32>()) as u64;
        let limit = self.memory_info.max_storage_buffer_binding_size as u64;
        if byte_size > limit {
            return Err(FerroError::invalid_input(format!(
                "GPU {}: buffer size {} bytes ({} f32 elements) exceeds \
                 max_storage_buffer_binding_size {} bytes",
                label, byte_size, n_elements, limit
            )));
        }
        Ok(())
    }

    /// Read f32 results from GPU buffer and convert to f64 Vec.
    fn read_buffer(&self, staging: &wgpu::Buffer, size: usize) -> Vec<f64> {
        let buffer_slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let f32_data: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = f32_data[..size].iter().map(|&v| v as f64).collect();
        drop(data);
        staging.unmap();
        result
    }
}

impl WgpuBackend {
    /// Run a unary element-wise shader (3 bindings: uniform, input, output).
    fn run_elementwise_unary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        x: &Array2<f64>,
        label: &str,
    ) -> Result<Array2<f64>> {
        let (rows, cols) = x.dim();
        let total = rows * cols;
        let x_std = x.as_standard_layout();
        let x_slice = x_std.as_slice().unwrap();

        let dims = [total as u32, 0u32, 0u32, 0u32];
        let uniform_buf = self.create_uniform_buffer(&dims, &format!("{}_dims", label));
        let input_buf = self.create_storage_buffer(x_slice, &format!("{}_input", label));
        let out_buf = self.create_output_buffer(total, &format!("{}_out", label));
        let staging = self.create_staging_buffer(total, &format!("{}_staging", label));

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}_bind_group", label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}_encoder", label)),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}_pass", label)),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg = (total as u32 + 255) / 256;
            pass.dispatch_workgroups(wg, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            (total * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let result_data = self.read_buffer(&staging, total);
        Array2::from_shape_vec((rows, cols), result_data)
            .map_err(|e| FerroError::numerical(format!("GPU {} reshape: {}", label, e)))
    }

    /// Run a row-wise shader with rows/cols uniform (3 bindings: uniform, input, output).
    fn run_rowwise(
        &self,
        pipeline: &wgpu::ComputePipeline,
        x: &Array2<f64>,
        out_size: usize,
        dims: [u32; 4],
        label: &str,
    ) -> Result<Vec<f64>> {
        let x_std = x.as_standard_layout();
        let x_slice = x_std.as_slice().unwrap();
        let (rows, _) = x.dim();

        let uniform_buf = self.create_uniform_buffer(&dims, &format!("{}_dims", label));
        let input_buf = self.create_storage_buffer(x_slice, &format!("{}_input", label));
        let out_buf = self.create_output_buffer(out_size, &format!("{}_out", label));
        let staging = self.create_staging_buffer(out_size, &format!("{}_staging", label));

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}_bind_group", label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}_encoder", label)),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}_pass", label)),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg = (rows as u32 + 255) / 256;
            pass.dispatch_workgroups(wg, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            (out_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(self.read_buffer(&staging, out_size))
    }
}

impl GpuBackend for WgpuBackend {
    fn relu(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.run_elementwise_unary(&self.relu_pipeline, x, "relu")
    }

    fn sigmoid(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.run_elementwise_unary(&self.sigmoid_pipeline, x, "sigmoid")
    }

    fn softmax(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (rows, cols) = x.dim();
        let dims = [rows as u32, cols as u32, 0u32, 0u32];
        let result_data =
            self.run_rowwise(&self.softmax_pipeline, x, rows * cols, dims, "softmax")?;
        Array2::from_shape_vec((rows, cols), result_data)
            .map_err(|e| FerroError::numerical(format!("GPU softmax reshape: {}", e)))
    }

    fn row_sum(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let (rows, cols) = x.dim();
        let dims = [rows as u32, cols as u32, 0u32, 0u32]; // mode=0 sum
        let result_data = self.run_rowwise(&self.row_reduce_pipeline, x, rows, dims, "row_sum")?;
        Ok(Array1::from_vec(result_data))
    }

    fn row_max(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let (rows, cols) = x.dim();
        let dims = [rows as u32, cols as u32, 1u32, 0u32]; // mode=1 max
        let result_data = self.run_rowwise(&self.row_reduce_pipeline, x, rows, dims, "row_max")?;
        Ok(Array1::from_vec(result_data))
    }

    fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> Result<Array2<f64>> {
        let (rows, cols) = x.dim();
        if bias.len() != cols {
            return Err(FerroError::shape_mismatch(
                format!("matrix cols = {}", cols),
                format!("bias len = {}", bias.len()),
            ));
        }

        let x_std = x.as_standard_layout();
        let x_slice = x_std.as_slice().unwrap();
        let bias_slice = bias.as_slice().unwrap();
        let total = rows * cols;

        let dims = [rows as u32, cols as u32, 0u32, 0u32];
        let uniform_buf = self.create_uniform_buffer(&dims, "bias_add_dims");
        let input_buf = self.create_storage_buffer(x_slice, "bias_add_input");
        let bias_buf = self.create_storage_buffer(bias_slice, "bias_add_bias");
        let out_buf = self.create_output_buffer(total, "bias_add_out");
        let staging = self.create_staging_buffer(total, "bias_add_staging");

        let bind_group_layout = self.bias_add_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bias_add_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bias_add_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bias_add_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bias_add_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg = (total as u32 + 255) / 256;
            pass.dispatch_workgroups(wg, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            (total * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let result_data = self.read_buffer(&staging, total);
        Array2::from_shape_vec((rows, cols), result_data)
            .map_err(|e| FerroError::numerical(format!("GPU bias_add reshape: {}", e)))
    }

    fn relu_grad(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        self.run_elementwise_unary(&self.relu_grad_pipeline, z, "relu_grad")
    }

    fn sigmoid_grad(&self, output: &Array2<f64>) -> Result<Array2<f64>> {
        self.run_elementwise_unary(&self.sigmoid_grad_pipeline, output, "sigmoid_grad")
    }

    fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();
        if rows_a != rows_b || cols_a != cols_b {
            return Err(FerroError::shape_mismatch(
                format!("a shape = ({}, {})", rows_a, cols_a),
                format!("b shape = ({}, {})", rows_b, cols_b),
            ));
        }

        let total = rows_a * cols_a;
        let a_std = a.as_standard_layout();
        let b_std = b.as_standard_layout();
        let a_slice = a_std.as_slice().unwrap();
        let b_slice = b_std.as_slice().unwrap();

        let dims = [total as u32, 0u32, 0u32, 0u32];
        let uniform_buf = self.create_uniform_buffer(&dims, "emul_dims");
        let a_buf = self.create_storage_buffer(a_slice, "emul_a");
        let b_buf = self.create_storage_buffer(b_slice, "emul_b");
        let out_buf = self.create_output_buffer(total, "emul_out");
        let staging = self.create_staging_buffer(total, "emul_staging");

        let bind_group_layout = self.elementwise_mul_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emul_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("emul_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.elementwise_mul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg = (total as u32 + 255) / 256;
            pass.dispatch_workgroups(wg, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            (total * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let result_data = self.read_buffer(&staging, total);
        Array2::from_shape_vec((rows_a, cols_a), result_data)
            .map_err(|e| FerroError::numerical(format!("GPU elementwise_mul reshape: {}", e)))
    }

    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(FerroError::shape_mismatch(
                format!("A cols = {}", k),
                format!("B rows = {}", k2),
            ));
        }

        // Pre-flight memory checks
        self.check_buffer_size(m * k, "matmul input A")?;
        self.check_buffer_size(k * n, "matmul input B")?;
        self.check_buffer_size(m * n, "matmul output C")?;

        // Ensure contiguous layout
        let a_std = a.as_standard_layout();
        let b_std = b.as_standard_layout();
        let a_slice = a_std.as_slice().unwrap();
        let b_slice = b_std.as_slice().unwrap();

        let dims = [m as u32, k as u32, n as u32, 0u32];
        let uniform_buf = self.create_uniform_buffer(&dims, "matmul_dims");
        let a_buf = self.create_storage_buffer(a_slice, "matmul_a");
        let b_buf = self.create_storage_buffer(b_slice, "matmul_b");
        let c_buf = self.create_output_buffer(m * n, "matmul_c");
        let staging = self.create_staging_buffer(m * n, "matmul_staging");

        let bind_group_layout = self.matmul_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (m as u32 + 15) / 16;
            let wg_y = (n as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &c_buf,
            0,
            &staging,
            0,
            (m * n * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let result_data = self.read_buffer(&staging, m * n);
        Array2::from_shape_vec((m, n), result_data)
            .map_err(|e| FerroError::numerical(format!("GPU matmul reshape: {}", e)))
    }

    fn pairwise_distances(&self, x: &Array2<f64>, centers: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, d) = x.dim();
        let (n_centers, d2) = centers.dim();
        if d != d2 {
            return Err(FerroError::shape_mismatch(
                format!("X features = {}", d),
                format!("centers features = {}", d2),
            ));
        }

        // Pre-flight memory checks
        self.check_buffer_size(n_samples * d, "pairwise_distances input X")?;
        self.check_buffer_size(n_centers * d, "pairwise_distances input centers")?;
        self.check_buffer_size(n_samples * n_centers, "pairwise_distances output")?;

        let x_std = x.as_standard_layout();
        let c_std = centers.as_standard_layout();
        let x_slice = x_std.as_slice().unwrap();
        let c_slice = c_std.as_slice().unwrap();

        let dims = [n_samples as u32, n_centers as u32, d as u32, 0u32];
        let uniform_buf = self.create_uniform_buffer(&dims, "dist_dims");
        let x_buf = self.create_storage_buffer(x_slice, "dist_x");
        let c_buf = self.create_storage_buffer(c_slice, "dist_centers");
        let out_buf = self.create_output_buffer(n_samples * n_centers, "dist_out");
        let staging = self.create_staging_buffer(n_samples * n_centers, "dist_staging");

        let bind_group_layout = self.distance_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dist_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dist_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.distance_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (n_samples * n_centers) as u32;
            let wg = (total + 255) / 256;
            pass.dispatch_workgroups(wg, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            (n_samples * n_centers * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let result_data = self.read_buffer(&staging, n_samples * n_centers);
        Array2::from_shape_vec((n_samples, n_centers), result_data)
            .map_err(|e| FerroError::numerical(format!("GPU distance reshape: {}", e)))
    }

    fn is_available(&self) -> bool {
        true
    }
}
