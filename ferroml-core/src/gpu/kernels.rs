//! WGSL compute shader source code for GPU operations.

/// Tiled matrix multiplication shader (f32).
///
/// Computes C = A @ B using 16x16 tiles for shared memory optimization.
/// Uniforms: M, K, N dimensions.
/// Bindings: A (M*K), B (K*N), C (M*N) as storage buffers.
pub const MATMUL_SHADER: &str = r"
struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>; // 16*16
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;

    var acc: f32 = 0.0;
    let n_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE + lc;
        if (row < dims.M && a_col < dims.K) {
            tile_a[lr * TILE + lc] = a[row * dims.K + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        // Load tile from B
        let b_row = t * TILE + lr;
        if (b_row < dims.K && col < dims.N) {
            tile_b[lr * TILE + lc] = b[b_row * dims.N + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate dot product for this tile
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tile_a[lr * TILE + k] * tile_b[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = acc;
    }
}
";

/// Element-wise ReLU shader (f32).
///
/// Computes output[i] = max(0, input[i]).
/// Uniforms: len (number of elements).
/// Bindings: input, output as storage buffers.
pub const RELU_SHADER: &str = r"
struct Dims {
    len: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) {
        return;
    }
    output[idx] = max(0.0, input[idx]);
}
";

/// Element-wise sigmoid shader (f32).
///
/// Computes output[i] = 1 / (1 + exp(-input[i])).
/// Uniforms: len (number of elements).
/// Bindings: input, output as storage buffers.
pub const SIGMOID_SHADER: &str = r"
struct Dims {
    len: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) {
        return;
    }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}
";

/// Row-wise softmax shader (f32).
///
/// Two-pass: first find row max, then compute exp(x-max)/sum(exp(x-max)).
/// Each thread handles one row.
/// Uniforms: rows, cols.
/// Bindings: input (rows*cols), output (rows*cols).
pub const SOFTMAX_SHADER: &str = r"
struct Dims {
    rows: u32,
    cols: u32,
    _pad: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }

    let offset = row * dims.cols;

    // Pass 1: find row max
    var row_max: f32 = input[offset];
    for (var j: u32 = 1u; j < dims.cols; j = j + 1u) {
        row_max = max(row_max, input[offset + j]);
    }

    // Pass 2: compute exp(x - max) and sum
    var sum_exp: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let e = exp(input[offset + j] - row_max);
        output[offset + j] = e;
        sum_exp = sum_exp + e;
    }

    // Pass 3: normalize
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        output[offset + j] = output[offset + j] / sum_exp;
    }
}
";

/// Row-wise reduction shader (f32).
///
/// Computes row-wise sum (mode=0) or max (mode=1).
/// Each thread handles one row.
/// Uniforms: rows, cols, mode.
/// Bindings: input (rows*cols), output (rows).
pub const ROW_REDUCE_SHADER: &str = r"
struct Dims {
    rows: u32,
    cols: u32,
    mode: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.rows) {
        return;
    }

    let offset = row * dims.cols;

    if (dims.mode == 0u) {
        // Sum reduction
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
            acc = acc + input[offset + j];
        }
        output[row] = acc;
    } else {
        // Max reduction
        var m: f32 = input[offset];
        for (var j: u32 = 1u; j < dims.cols; j = j + 1u) {
            m = max(m, input[offset + j]);
        }
        output[row] = m;
    }
}
";

/// Broadcast bias-add shader (f32).
///
/// Computes output[i,j] = input[i,j] + bias[j].
/// Uniforms: rows, cols.
/// Bindings: input (rows*cols), bias (cols), output (rows*cols).
pub const BIAS_ADD_SHADER: &str = r"
struct Dims {
    rows: u32,
    cols: u32,
    _pad: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.rows * dims.cols;
    if (idx >= total) {
        return;
    }
    let col = idx % dims.cols;
    output[idx] = input[idx] + bias[col];
}
";

/// ReLU gradient shader (f32).
///
/// Computes output[i] = (input[i] > 0) ? 1.0 : 0.0.
/// Uniforms: len.
/// Bindings: input, output as storage buffers.
pub const RELU_GRAD_SHADER: &str = r"
struct Dims {
    len: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) {
        return;
    }
    if (input[idx] > 0.0) {
        output[idx] = 1.0;
    } else {
        output[idx] = 0.0;
    }
}
";

/// Sigmoid gradient shader (f32).
///
/// Computes output[i] = input[i] * (1.0 - input[i]) where input is the sigmoid output.
/// Uniforms: len.
/// Bindings: input, output as storage buffers.
pub const SIGMOID_GRAD_SHADER: &str = r"
struct Dims {
    len: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) {
        return;
    }
    output[idx] = input[idx] * (1.0 - input[idx]);
}
";

/// Element-wise multiplication shader (f32).
///
/// Computes output[i] = a[i] * b[i].
/// Uniforms: len.
/// Bindings: a, b, output as storage buffers.
pub const ELEMENTWISE_MUL_SHADER: &str = r"
struct Dims {
    len: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) {
        return;
    }
    output[idx] = a[idx] * b[idx];
}
";

/// Pairwise squared Euclidean distance shader (f32).
///
/// Computes dist[i,j] = sum_k (x[i,k] - centers[j,k])^2
/// Uniforms: N (samples), C (centers), D (features).
/// Bindings: X (N*D), centers (C*D), distances (N*C).
pub const DISTANCE_SHADER: &str = r"
struct Dims {
    N: u32,
    C: u32,
    D: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> centers: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let i = idx / dims.C;
    let j = idx % dims.C;

    if (i >= dims.N || j >= dims.C) {
        return;
    }

    var dist: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.D; k = k + 1u) {
        let diff = x[i * dims.D + k] - centers[j * dims.D + k];
        dist = dist + diff * diff;
    }

    distances[i * dims.C + j] = dist;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Shader source non-empty checks
    // ========================================================================

    #[test]
    fn test_matmul_shader_non_empty() {
        assert!(!MATMUL_SHADER.is_empty());
        assert!(MATMUL_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_distance_shader_non_empty() {
        assert!(!DISTANCE_SHADER.is_empty());
        assert!(DISTANCE_SHADER.len() > 100, "Shader suspiciously short");
    }

    // ========================================================================
    // Matmul shader structure checks
    // ========================================================================

    #[test]
    fn test_matmul_shader_has_workgroup_size() {
        assert!(
            MATMUL_SHADER.contains("@workgroup_size"),
            "Matmul shader missing @workgroup_size annotation"
        );
    }

    #[test]
    fn test_matmul_shader_workgroup_16x16() {
        assert!(
            MATMUL_SHADER.contains("@workgroup_size(16, 16)"),
            "Matmul shader should use 16x16 tile workgroup size"
        );
    }

    #[test]
    fn test_matmul_shader_has_compute() {
        assert!(
            MATMUL_SHADER.contains("@compute"),
            "Matmul shader missing @compute annotation"
        );
    }

    #[test]
    fn test_matmul_shader_has_main_entry_point() {
        assert!(
            MATMUL_SHADER.contains("fn main"),
            "Matmul shader missing main entry point"
        );
    }

    #[test]
    fn test_matmul_shader_buffer_bindings() {
        // Should have 4 bindings: uniform dims, A, B, C
        assert!(MATMUL_SHADER.contains("@binding(0)"));
        assert!(MATMUL_SHADER.contains("@binding(1)"));
        assert!(MATMUL_SHADER.contains("@binding(2)"));
        assert!(MATMUL_SHADER.contains("@binding(3)"));
    }

    #[test]
    fn test_matmul_shader_has_uniform_dims() {
        assert!(
            MATMUL_SHADER.contains("var<uniform>"),
            "Matmul shader should have uniform buffer for dimensions"
        );
    }

    #[test]
    fn test_matmul_shader_has_storage_buffers() {
        assert!(
            MATMUL_SHADER.contains("var<storage, read>"),
            "Matmul shader should have read storage buffers"
        );
        assert!(
            MATMUL_SHADER.contains("var<storage, read_write>"),
            "Matmul shader should have read_write output buffer"
        );
    }

    #[test]
    fn test_matmul_shader_has_workgroup_barrier() {
        assert!(
            MATMUL_SHADER.contains("workgroupBarrier()"),
            "Tiled matmul shader requires workgroup barriers for synchronization"
        );
    }

    #[test]
    fn test_matmul_shader_has_tile_shared_memory() {
        assert!(
            MATMUL_SHADER.contains("var<workgroup>"),
            "Tiled matmul should use workgroup shared memory"
        );
    }

    #[test]
    fn test_matmul_shader_dims_struct() {
        assert!(
            MATMUL_SHADER.contains("struct Dims"),
            "Matmul shader should define Dims struct"
        );
        assert!(MATMUL_SHADER.contains("M: u32"));
        assert!(MATMUL_SHADER.contains("K: u32"));
        assert!(MATMUL_SHADER.contains("N: u32"));
    }

    // ========================================================================
    // Distance shader structure checks
    // ========================================================================

    #[test]
    fn test_distance_shader_has_workgroup_size() {
        assert!(
            DISTANCE_SHADER.contains("@workgroup_size"),
            "Distance shader missing @workgroup_size annotation"
        );
    }

    #[test]
    fn test_distance_shader_workgroup_256() {
        assert!(
            DISTANCE_SHADER.contains("@workgroup_size(256)"),
            "Distance shader should use workgroup size 256"
        );
    }

    #[test]
    fn test_distance_shader_has_compute() {
        assert!(
            DISTANCE_SHADER.contains("@compute"),
            "Distance shader missing @compute annotation"
        );
    }

    #[test]
    fn test_distance_shader_has_main_entry_point() {
        assert!(
            DISTANCE_SHADER.contains("fn main"),
            "Distance shader missing main entry point"
        );
    }

    #[test]
    fn test_distance_shader_buffer_bindings() {
        assert!(DISTANCE_SHADER.contains("@binding(0)"));
        assert!(DISTANCE_SHADER.contains("@binding(1)"));
        assert!(DISTANCE_SHADER.contains("@binding(2)"));
        assert!(DISTANCE_SHADER.contains("@binding(3)"));
    }

    #[test]
    fn test_distance_shader_has_uniform_dims() {
        assert!(
            DISTANCE_SHADER.contains("var<uniform>"),
            "Distance shader should have uniform buffer for dimensions"
        );
    }

    #[test]
    fn test_distance_shader_has_storage_buffers() {
        assert!(
            DISTANCE_SHADER.contains("var<storage, read>"),
            "Distance shader should have read storage buffers"
        );
        assert!(
            DISTANCE_SHADER.contains("var<storage, read_write>"),
            "Distance shader should have read_write output buffer"
        );
    }

    #[test]
    fn test_distance_shader_dims_struct() {
        assert!(
            DISTANCE_SHADER.contains("struct Dims"),
            "Distance shader should define Dims struct"
        );
        assert!(DISTANCE_SHADER.contains("N: u32"));
        assert!(DISTANCE_SHADER.contains("C: u32"));
        assert!(DISTANCE_SHADER.contains("D: u32"));
    }

    #[test]
    fn test_distance_shader_uses_f32() {
        // Both shaders should use f32 (GPU precision)
        assert!(DISTANCE_SHADER.contains("f32"));
        assert!(MATMUL_SHADER.contains("f32"));
    }

    #[test]
    fn test_distance_shader_group_annotation() {
        // All bindings should be in group 0
        assert!(DISTANCE_SHADER.contains("@group(0)"));
        assert!(MATMUL_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Cross-shader consistency
    // ========================================================================

    #[test]
    fn test_both_shaders_have_global_invocation_id() {
        assert!(MATMUL_SHADER.contains("global_invocation_id"));
        assert!(DISTANCE_SHADER.contains("global_invocation_id"));
    }

    // ========================================================================
    // ReLU shader structure checks
    // ========================================================================

    #[test]
    fn test_relu_shader_non_empty() {
        assert!(!RELU_SHADER.is_empty());
        assert!(RELU_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_relu_shader_has_workgroup_size() {
        assert!(RELU_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_relu_shader_workgroup_256() {
        assert!(RELU_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_relu_shader_has_compute() {
        assert!(RELU_SHADER.contains("@compute"));
    }

    #[test]
    fn test_relu_shader_has_main_entry_point() {
        assert!(RELU_SHADER.contains("fn main"));
    }

    #[test]
    fn test_relu_shader_buffer_bindings() {
        assert!(RELU_SHADER.contains("@binding(0)"));
        assert!(RELU_SHADER.contains("@binding(1)"));
        assert!(RELU_SHADER.contains("@binding(2)"));
    }

    #[test]
    fn test_relu_shader_has_uniform_dims() {
        assert!(RELU_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_relu_shader_has_storage_buffers() {
        assert!(RELU_SHADER.contains("var<storage, read>"));
        assert!(RELU_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_relu_shader_dims_struct() {
        assert!(RELU_SHADER.contains("struct Dims"));
        assert!(RELU_SHADER.contains("len: u32"));
    }

    #[test]
    fn test_relu_shader_uses_f32() {
        assert!(RELU_SHADER.contains("f32"));
    }

    #[test]
    fn test_relu_shader_group_annotation() {
        assert!(RELU_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Sigmoid shader structure checks
    // ========================================================================

    #[test]
    fn test_sigmoid_shader_non_empty() {
        assert!(!SIGMOID_SHADER.is_empty());
        assert!(SIGMOID_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_sigmoid_shader_has_workgroup_size() {
        assert!(SIGMOID_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_sigmoid_shader_workgroup_256() {
        assert!(SIGMOID_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_sigmoid_shader_has_compute() {
        assert!(SIGMOID_SHADER.contains("@compute"));
    }

    #[test]
    fn test_sigmoid_shader_has_main_entry_point() {
        assert!(SIGMOID_SHADER.contains("fn main"));
    }

    #[test]
    fn test_sigmoid_shader_buffer_bindings() {
        assert!(SIGMOID_SHADER.contains("@binding(0)"));
        assert!(SIGMOID_SHADER.contains("@binding(1)"));
        assert!(SIGMOID_SHADER.contains("@binding(2)"));
    }

    #[test]
    fn test_sigmoid_shader_has_uniform_dims() {
        assert!(SIGMOID_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_sigmoid_shader_has_storage_buffers() {
        assert!(SIGMOID_SHADER.contains("var<storage, read>"));
        assert!(SIGMOID_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_sigmoid_shader_dims_struct() {
        assert!(SIGMOID_SHADER.contains("struct Dims"));
        assert!(SIGMOID_SHADER.contains("len: u32"));
    }

    #[test]
    fn test_sigmoid_shader_uses_f32() {
        assert!(SIGMOID_SHADER.contains("f32"));
    }

    #[test]
    fn test_sigmoid_shader_group_annotation() {
        assert!(SIGMOID_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Softmax shader structure checks
    // ========================================================================

    #[test]
    fn test_softmax_shader_non_empty() {
        assert!(!SOFTMAX_SHADER.is_empty());
        assert!(SOFTMAX_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_softmax_shader_has_workgroup_size() {
        assert!(SOFTMAX_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_softmax_shader_workgroup_256() {
        assert!(SOFTMAX_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_softmax_shader_has_compute() {
        assert!(SOFTMAX_SHADER.contains("@compute"));
    }

    #[test]
    fn test_softmax_shader_has_main_entry_point() {
        assert!(SOFTMAX_SHADER.contains("fn main"));
    }

    #[test]
    fn test_softmax_shader_buffer_bindings() {
        assert!(SOFTMAX_SHADER.contains("@binding(0)"));
        assert!(SOFTMAX_SHADER.contains("@binding(1)"));
        assert!(SOFTMAX_SHADER.contains("@binding(2)"));
    }

    #[test]
    fn test_softmax_shader_has_uniform_dims() {
        assert!(SOFTMAX_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_softmax_shader_has_storage_buffers() {
        assert!(SOFTMAX_SHADER.contains("var<storage, read>"));
        assert!(SOFTMAX_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_softmax_shader_dims_struct() {
        assert!(SOFTMAX_SHADER.contains("struct Dims"));
        assert!(SOFTMAX_SHADER.contains("rows: u32"));
        assert!(SOFTMAX_SHADER.contains("cols: u32"));
    }

    #[test]
    fn test_softmax_shader_uses_f32() {
        assert!(SOFTMAX_SHADER.contains("f32"));
    }

    #[test]
    fn test_softmax_shader_group_annotation() {
        assert!(SOFTMAX_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Row reduce shader structure checks
    // ========================================================================

    #[test]
    fn test_row_reduce_shader_non_empty() {
        assert!(!ROW_REDUCE_SHADER.is_empty());
        assert!(ROW_REDUCE_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_row_reduce_shader_has_workgroup_size() {
        assert!(ROW_REDUCE_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_row_reduce_shader_workgroup_256() {
        assert!(ROW_REDUCE_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_row_reduce_shader_has_compute() {
        assert!(ROW_REDUCE_SHADER.contains("@compute"));
    }

    #[test]
    fn test_row_reduce_shader_has_main_entry_point() {
        assert!(ROW_REDUCE_SHADER.contains("fn main"));
    }

    #[test]
    fn test_row_reduce_shader_buffer_bindings() {
        assert!(ROW_REDUCE_SHADER.contains("@binding(0)"));
        assert!(ROW_REDUCE_SHADER.contains("@binding(1)"));
        assert!(ROW_REDUCE_SHADER.contains("@binding(2)"));
    }

    #[test]
    fn test_row_reduce_shader_has_uniform_dims() {
        assert!(ROW_REDUCE_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_row_reduce_shader_has_storage_buffers() {
        assert!(ROW_REDUCE_SHADER.contains("var<storage, read>"));
        assert!(ROW_REDUCE_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_row_reduce_shader_dims_struct() {
        assert!(ROW_REDUCE_SHADER.contains("struct Dims"));
        assert!(ROW_REDUCE_SHADER.contains("rows: u32"));
        assert!(ROW_REDUCE_SHADER.contains("cols: u32"));
        assert!(ROW_REDUCE_SHADER.contains("mode: u32"));
    }

    #[test]
    fn test_row_reduce_shader_uses_f32() {
        assert!(ROW_REDUCE_SHADER.contains("f32"));
    }

    #[test]
    fn test_row_reduce_shader_group_annotation() {
        assert!(ROW_REDUCE_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Bias add shader structure checks
    // ========================================================================

    #[test]
    fn test_bias_add_shader_non_empty() {
        assert!(!BIAS_ADD_SHADER.is_empty());
        assert!(BIAS_ADD_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_bias_add_shader_has_workgroup_size() {
        assert!(BIAS_ADD_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_bias_add_shader_workgroup_256() {
        assert!(BIAS_ADD_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_bias_add_shader_has_compute() {
        assert!(BIAS_ADD_SHADER.contains("@compute"));
    }

    #[test]
    fn test_bias_add_shader_has_main_entry_point() {
        assert!(BIAS_ADD_SHADER.contains("fn main"));
    }

    #[test]
    fn test_bias_add_shader_buffer_bindings() {
        assert!(BIAS_ADD_SHADER.contains("@binding(0)"));
        assert!(BIAS_ADD_SHADER.contains("@binding(1)"));
        assert!(BIAS_ADD_SHADER.contains("@binding(2)"));
        assert!(BIAS_ADD_SHADER.contains("@binding(3)"));
    }

    #[test]
    fn test_bias_add_shader_has_uniform_dims() {
        assert!(BIAS_ADD_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_bias_add_shader_has_storage_buffers() {
        assert!(BIAS_ADD_SHADER.contains("var<storage, read>"));
        assert!(BIAS_ADD_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_bias_add_shader_dims_struct() {
        assert!(BIAS_ADD_SHADER.contains("struct Dims"));
        assert!(BIAS_ADD_SHADER.contains("rows: u32"));
        assert!(BIAS_ADD_SHADER.contains("cols: u32"));
    }

    #[test]
    fn test_bias_add_shader_uses_f32() {
        assert!(BIAS_ADD_SHADER.contains("f32"));
    }

    #[test]
    fn test_bias_add_shader_group_annotation() {
        assert!(BIAS_ADD_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // ReLU grad shader structure checks
    // ========================================================================

    #[test]
    fn test_relu_grad_shader_non_empty() {
        assert!(!RELU_GRAD_SHADER.is_empty());
        assert!(RELU_GRAD_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_relu_grad_shader_has_workgroup_size() {
        assert!(RELU_GRAD_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_relu_grad_shader_workgroup_256() {
        assert!(RELU_GRAD_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_relu_grad_shader_has_compute() {
        assert!(RELU_GRAD_SHADER.contains("@compute"));
    }

    #[test]
    fn test_relu_grad_shader_has_main_entry_point() {
        assert!(RELU_GRAD_SHADER.contains("fn main"));
    }

    #[test]
    fn test_relu_grad_shader_buffer_bindings() {
        assert!(RELU_GRAD_SHADER.contains("@binding(0)"));
        assert!(RELU_GRAD_SHADER.contains("@binding(1)"));
        assert!(RELU_GRAD_SHADER.contains("@binding(2)"));
    }

    #[test]
    fn test_relu_grad_shader_has_uniform_dims() {
        assert!(RELU_GRAD_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_relu_grad_shader_has_storage_buffers() {
        assert!(RELU_GRAD_SHADER.contains("var<storage, read>"));
        assert!(RELU_GRAD_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_relu_grad_shader_dims_struct() {
        assert!(RELU_GRAD_SHADER.contains("struct Dims"));
        assert!(RELU_GRAD_SHADER.contains("len: u32"));
    }

    #[test]
    fn test_relu_grad_shader_uses_f32() {
        assert!(RELU_GRAD_SHADER.contains("f32"));
    }

    #[test]
    fn test_relu_grad_shader_group_annotation() {
        assert!(RELU_GRAD_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Sigmoid grad shader structure checks
    // ========================================================================

    #[test]
    fn test_sigmoid_grad_shader_non_empty() {
        assert!(!SIGMOID_GRAD_SHADER.is_empty());
        assert!(SIGMOID_GRAD_SHADER.len() > 100, "Shader suspiciously short");
    }

    #[test]
    fn test_sigmoid_grad_shader_has_workgroup_size() {
        assert!(SIGMOID_GRAD_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_sigmoid_grad_shader_workgroup_256() {
        assert!(SIGMOID_GRAD_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_sigmoid_grad_shader_has_compute() {
        assert!(SIGMOID_GRAD_SHADER.contains("@compute"));
    }

    #[test]
    fn test_sigmoid_grad_shader_has_main_entry_point() {
        assert!(SIGMOID_GRAD_SHADER.contains("fn main"));
    }

    #[test]
    fn test_sigmoid_grad_shader_buffer_bindings() {
        assert!(SIGMOID_GRAD_SHADER.contains("@binding(0)"));
        assert!(SIGMOID_GRAD_SHADER.contains("@binding(1)"));
        assert!(SIGMOID_GRAD_SHADER.contains("@binding(2)"));
    }

    #[test]
    fn test_sigmoid_grad_shader_has_uniform_dims() {
        assert!(SIGMOID_GRAD_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_sigmoid_grad_shader_has_storage_buffers() {
        assert!(SIGMOID_GRAD_SHADER.contains("var<storage, read>"));
        assert!(SIGMOID_GRAD_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_sigmoid_grad_shader_dims_struct() {
        assert!(SIGMOID_GRAD_SHADER.contains("struct Dims"));
        assert!(SIGMOID_GRAD_SHADER.contains("len: u32"));
    }

    #[test]
    fn test_sigmoid_grad_shader_uses_f32() {
        assert!(SIGMOID_GRAD_SHADER.contains("f32"));
    }

    #[test]
    fn test_sigmoid_grad_shader_group_annotation() {
        assert!(SIGMOID_GRAD_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Elementwise mul shader structure checks
    // ========================================================================

    #[test]
    fn test_elementwise_mul_shader_non_empty() {
        assert!(!ELEMENTWISE_MUL_SHADER.is_empty());
        assert!(
            ELEMENTWISE_MUL_SHADER.len() > 100,
            "Shader suspiciously short"
        );
    }

    #[test]
    fn test_elementwise_mul_shader_has_workgroup_size() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_elementwise_mul_shader_workgroup_256() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_elementwise_mul_shader_has_compute() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("@compute"));
    }

    #[test]
    fn test_elementwise_mul_shader_has_main_entry_point() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("fn main"));
    }

    #[test]
    fn test_elementwise_mul_shader_buffer_bindings() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("@binding(0)"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("@binding(1)"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("@binding(2)"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("@binding(3)"));
    }

    #[test]
    fn test_elementwise_mul_shader_has_uniform_dims() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("var<uniform>"));
    }

    #[test]
    fn test_elementwise_mul_shader_has_storage_buffers() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("var<storage, read>"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("var<storage, read_write>"));
    }

    #[test]
    fn test_elementwise_mul_shader_dims_struct() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("struct Dims"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("len: u32"));
    }

    #[test]
    fn test_elementwise_mul_shader_uses_f32() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("f32"));
    }

    #[test]
    fn test_elementwise_mul_shader_group_annotation() {
        assert!(ELEMENTWISE_MUL_SHADER.contains("@group(0)"));
    }

    // ========================================================================
    // Cross-shader consistency (expanded)
    // ========================================================================

    #[test]
    fn test_all_new_shaders_have_global_invocation_id() {
        assert!(RELU_SHADER.contains("global_invocation_id"));
        assert!(SIGMOID_SHADER.contains("global_invocation_id"));
        assert!(SOFTMAX_SHADER.contains("global_invocation_id"));
        assert!(ROW_REDUCE_SHADER.contains("global_invocation_id"));
        assert!(BIAS_ADD_SHADER.contains("global_invocation_id"));
        assert!(RELU_GRAD_SHADER.contains("global_invocation_id"));
        assert!(SIGMOID_GRAD_SHADER.contains("global_invocation_id"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("global_invocation_id"));
    }

    #[test]
    fn test_all_new_shaders_group0() {
        assert!(RELU_SHADER.contains("@group(0)"));
        assert!(SIGMOID_SHADER.contains("@group(0)"));
        assert!(SOFTMAX_SHADER.contains("@group(0)"));
        assert!(ROW_REDUCE_SHADER.contains("@group(0)"));
        assert!(BIAS_ADD_SHADER.contains("@group(0)"));
        assert!(RELU_GRAD_SHADER.contains("@group(0)"));
        assert!(SIGMOID_GRAD_SHADER.contains("@group(0)"));
        assert!(ELEMENTWISE_MUL_SHADER.contains("@group(0)"));
    }

    #[test]
    fn test_shaders_are_valid_wgsl_structure() {
        // Basic structural check: braces are balanced
        fn check_balanced(src: &str) -> bool {
            let mut depth = 0i32;
            for c in src.chars() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                if depth < 0 {
                    return false;
                }
            }
            depth == 0
        }
        assert!(
            check_balanced(MATMUL_SHADER),
            "Matmul shader has unbalanced braces"
        );
        assert!(
            check_balanced(DISTANCE_SHADER),
            "Distance shader has unbalanced braces"
        );
        assert!(
            check_balanced(RELU_SHADER),
            "ReLU shader has unbalanced braces"
        );
        assert!(
            check_balanced(SIGMOID_SHADER),
            "Sigmoid shader has unbalanced braces"
        );
        assert!(
            check_balanced(SOFTMAX_SHADER),
            "Softmax shader has unbalanced braces"
        );
        assert!(
            check_balanced(ROW_REDUCE_SHADER),
            "Row reduce shader has unbalanced braces"
        );
        assert!(
            check_balanced(BIAS_ADD_SHADER),
            "Bias add shader has unbalanced braces"
        );
        assert!(
            check_balanced(RELU_GRAD_SHADER),
            "ReLU grad shader has unbalanced braces"
        );
        assert!(
            check_balanced(SIGMOID_GRAD_SHADER),
            "Sigmoid grad shader has unbalanced braces"
        );
        assert!(
            check_balanced(ELEMENTWISE_MUL_SHADER),
            "Elementwise mul shader has unbalanced braces"
        );
    }
}
