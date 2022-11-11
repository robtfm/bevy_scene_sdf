#define_import_path sdf::bind

#import sdf::consts as consts
#import sdf::types as types

@group(0) @binding(0)
var<storage> mesh_headers: types::MeshHeader;

@group(0) @binding(1)
var<storage> transforms: types::Transforms;

@group(0) @binding(2)
var<storage> vertex_positions: array<vec4<f32>>;

@group(0) @binding(3)
var<uniform> cascade_info: types::CascadeInfo;

@group(0) @binding(4)
var<storage, read_write> coarse_processed_tris: array<types::ProcessedTri>;

// x = tile id, y = tri count top, per virtual tile
@group(0) @binding(5)
var<storage, read_write> coarse_tile_counts: array<vec2<u32>, consts::MAX_TILES>;

@group(0) @binding(6)
var<storage, read_write> coarse_ids: array<u32, consts::MAX_ID_COUNT>;

// subvoxel bits
@group(0) @binding(7)
var seed_jfa: texture_storage_3d<rg32uint, read_write>;

// xyz = seed voxel offset
// w = subvoxel index + 1 (or 0 for no seed found)
@group(0) @binding(8)
var nearest_jfa: texture_storage_3d<rgba8sint, read_write>;

@group(0) @binding(9)
var<storage, read_write> fine_output: array<u32, consts::FINE_OUTPUT_SIZE>;

