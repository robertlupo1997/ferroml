//! wgpu-based GPU backend implementation.

use super::GpuBackend;
use crate::{FerroError, Result};
use ndarray::Array2;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU backend using wgpu for cross-platform compute (Vulkan/Metal/DX12).
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    matmul_pipeline: wgpu::ComputePipeline,
    distance_pipeline: wgpu::ComputePipeline,
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

        Some(Self {
            device,
            queue,
            matmul_pipeline,
            distance_pipeline,
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

impl GpuBackend for WgpuBackend {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        if k != k2 {
            return Err(FerroError::shape_mismatch(
                format!("A cols = {}", k),
                format!("B rows = {}", k2),
            ));
        }

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
