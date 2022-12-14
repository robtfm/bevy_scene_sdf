#define_import_path sdf::consts

// this file will be autogenerated from the constants in lib.rs

let TILE_DIM_COUNT: u32 = number of tiles in each dimension;
let VOXELS_PER_TILE_DIM: u32 = number of voxels in a tile in each dimension;
let VOXELS_PER_DIM: u32 = tiles per dim * voxels per tile dim
let SUBVOXELS_PER_DIM: u32 = number of subvoxels on each dim axis

let TILE_COUNT: u32 = total number of tiles in the 3d grid
let VOXELS_PER_TILE: u32 = total number of voxels in each tile

let MAX_ID_COUNT: u32 = total number of ids we can allocate to tiles across all tiles
let MAX_IDS_PER_TILE: u32 = max ids each tile can hold

let FINE_OUTPUT_SIZE: u32 = 2 u32s per voxel * max tiles