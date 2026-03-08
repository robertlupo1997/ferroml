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
    }
}
