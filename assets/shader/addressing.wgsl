#define_import_path sdf::addressing

#import sdf::consts as consts
#import sdf::bind as bind

// local coords are relative to the sdf space
// i.e. local tile (0,0,0) is at the cascade top left

// local positions are f32s relative to cascade origin world position

// grid coords are 
// - the global invocation
// - the texture address key, used for textureLoad/Store in the jfa_seed and jfa_nearest textures

// % that mods negative numbers to [0..e2)
fn mod3i(e1: vec3<i32>, e2: i32) -> vec3<i32> {
    return e1 - vec3<i32>(floor(vec3<f32>(e1) / f32(e2))) * e2;
    // return e1 % e2;
}

// convert u32 index to local xyz
fn tile_index_to_local(index: u32) -> vec3<u32> {
    return vec3<u32>(
        index % consts::TILE_DIM_COUNT, 
        (index / consts::TILE_DIM_COUNT) % consts::TILE_DIM_COUNT, 
        index / (consts::TILE_DIM_COUNT * consts::TILE_DIM_COUNT), 
    );
}

// convert subvoxel u32 index to xyz
fn subvoxel_index_to_subvoxel(index: u32) -> vec3<u32> {
    return vec3<u32>(
        index % consts::SUBVOXELS_PER_VOXEL_DIM, 
        (index / consts::SUBVOXELS_PER_VOXEL_DIM) % consts::SUBVOXELS_PER_VOXEL_DIM, 
        index / (consts::SUBVOXELS_PER_VOXEL_DIM * consts::SUBVOXELS_PER_VOXEL_DIM), 
    );
}

fn subvoxel_local_position(voxel: vec3<i32>, subvoxel: vec3<u32>) -> vec3<f32> {
    return (vec3<f32>(voxel * i32(consts::SUBVOXELS_PER_VOXEL_DIM) + vec3<i32>(subvoxel)) + 0.5) * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM * consts::SUBVOXELS_PER_VOXEL_DIM);
}

// get the jfa texture coordinates for a given voxel, using 'toroidal addressing', i.e. modulo'd
fn voxel_local_to_grid(local_voxel: vec3<u32>) -> vec3<i32> {
    let local_tile = local_voxel / consts::VOXELS_PER_TILE_DIM;
    let tile_voxel = local_voxel % consts::VOXELS_PER_TILE_DIM;
    let world_tile = bind::cascade_info.origin.xyz + vec3<i32>(local_tile);
    let coords = mod3i(world_tile, i32(consts::TILE_DIM_COUNT)) * i32(consts::VOXELS_PER_TILE_DIM) + vec3<i32>(tile_voxel);
    return coords + vec3<i32>(vec3<u32>(consts::VOXELS_PER_DIM * bind::cascade_info.index, 0u, 0u));
}

// get the jfa texture coordinates for a given voxel, using 'toroidal addressing', i.e. modulo'd
// returns -1 if invalid
fn voxel_local_to_maybe_grid(local_voxel: vec3<i32>) -> vec3<i32> {
    if all(clamp(local_voxel, vec3<i32>(0), vec3<i32>(i32(consts::VOXELS_PER_DIM - 1u))) == local_voxel) {
        return voxel_local_to_grid(vec3<u32>(local_voxel));
    } else {
        return vec3<i32>(-1);
    }
}

// get the local voxel corresponding to the given grid/texture point
fn voxel_grid_to_local(grid_voxel: vec3<u32>) -> vec3<u32> {
    let tile_offset = vec3<u32>(i32(consts::TILE_DIM_COUNT) - bind::cascade_info.origin.xyz) % consts::TILE_DIM_COUNT;
    return (grid_voxel + vec3<u32>(consts::VOXELS_PER_TILE_DIM * tile_offset)) % consts::VOXELS_PER_DIM;
}

fn voxel_local_to_local_position(local_voxel: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(local_voxel) + 0.5) * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM);
}

// square dist from x to y
fn distance_squared(x: vec3<f32>, y: vec3<f32>) -> f32 {
    let v = y - x;
    return dot(v, v);
}

struct CheckResult {
    best_dist_sq: f32,
    write_value: vec4<i32>,
}

// return dist to source and offset
fn check_source(target_point: vec3<f32>, local_voxel: vec3<i32>, local_jump_source: vec3<i32>, max_dist_sq: f32) -> CheckResult {
    var res: CheckResult;
    res.best_dist_sq = max_dist_sq;

    let voxel_half_vec = vec3<f32>(0.5 * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM));

    let jump_source_coords = voxel_local_to_maybe_grid(local_jump_source);
    if jump_source_coords.x > -1 { // source is within our live grid
        let source_data = textureLoad(bind::nearest_jfa, jump_source_coords);

        if source_data.a != 0 { // source has a seed 
            let local_seed_voxel = vec3<i32>(local_jump_source + source_data.xyz);

            if all(source_data.xyz == vec3<i32>(0)) { 
                // jump source points to itself, need to check every subvoxel
                let seed_coords = voxel_local_to_maybe_grid(local_seed_voxel);

                if seed_coords.x > -1 { // seed is within our live grid
                    let seed_value = textureLoad(bind::seed_jfa, seed_coords).rg;

                    // check if it can be better (not sure this really helps)
                    let seed_position = voxel_local_to_local_position(local_seed_voxel);
                    let best_possible_dist_sq = distance_squared(target_point, clamp(target_point, seed_position - voxel_half_vec, seed_position + voxel_half_vec));
                    if best_possible_dist_sq < res.best_dist_sq {
                        for (var sx=0u; sx<consts::SUBVOXELS_PER_VOXEL_DIM; sx++) {
                            for (var sy=0u; sy<consts::SUBVOXELS_PER_VOXEL_DIM; sy++) {
                                for (var sz=0u; sz<consts::SUBVOXELS_PER_VOXEL_DIM; sz++) {
                                    var seed_subvoxel: bool;
                                    let shift = (((sz * consts::SUBVOXELS_PER_VOXEL_DIM) + sy) * consts::SUBVOXELS_PER_VOXEL_DIM) + sx;
                                    if shift < 32u {
                                        seed_subvoxel = ((seed_value.r >> shift) & 1u) == 1u;
                                    } else {
                                        seed_subvoxel = ((seed_value.g >> (shift - 32u)) & 1u) == 1u;
                                    }

                                    if seed_subvoxel {
                                        let local_seed_position = subvoxel_local_position(local_seed_voxel, vec3<u32>(sx,sy,sz));
                                        let dist_sq = distance_squared(local_seed_position, target_point);

                                        if dist_sq < res.best_dist_sq {
                                            res.best_dist_sq = dist_sq;
                                            res.write_value = vec4<i32>(local_seed_voxel - local_voxel, i32(shift + 1u));
                                        }
                                    }
                                }
                            }                                    
                        }
                    }
                }
            } else {
                // jump source points to another seed cell, we can assume their subvoxel is the one we want
                let subvoxel = subvoxel_index_to_subvoxel(u32(source_data.a - 1));
                let local_seed_position = subvoxel_local_position(local_seed_voxel, subvoxel);
                let dist_sq = distance_squared(local_seed_position, target_point);
                let offset = local_seed_voxel - local_voxel;

                if dist_sq < res.best_dist_sq && all(abs(offset) < vec3(128)) {
                    res.best_dist_sq = dist_sq;
                    res.write_value = vec4<i32>(offset, source_data.a);
                }
            }
        }
    }

    return res;
}