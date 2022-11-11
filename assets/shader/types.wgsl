#define_import_path sdf::types

struct MeshHeader {
    mesh_count: u32,
    tri_count: u32,
    // index, offset, count, unused
    index_offset_count: array<vec4<u32>>, // len = mesh_count
}

struct Transforms {
    t: array<mat4x4<f32>>, // len = mesh_count
}

struct CascadeInfo {
    origin: vec4<i32>,
    redraw: vec4<i32>,
    tile_size: f32,
    index: u32,
}

struct CascadeInfos {
    count: u32,
    cascades: array<CascadeInfo>,
}

struct ProcessedTri {
    vertex_positions: array<vec4<f32>, 3>,
}
