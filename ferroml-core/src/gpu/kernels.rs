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
