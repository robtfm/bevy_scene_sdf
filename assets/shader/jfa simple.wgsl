#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::addressing as addr

// jump flood algorithm

// todo
// - storage barrier for non-apple, avoid multiple draws?
// * only jump within update segment bounds (reduces 64,32,16,8 to 1 axis only). maybe specialised shader for those?
// * use simple distance

// todo: push constant?
struct JfaParams {
    jump_size: i32,
}

@group(1) @binding(0)
var<uniform> jfa_params: JfaParams;

@compute @workgroup_size(8,8,8)
fn jfa(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let update_min = (vec3<i32>(i32(consts::VOXELS_PER_DIM)) - max(bind::cascade_info.redraw.xyz, vec3<i32>(0)) * i32(consts::VOXELS_PER_TILE_DIM)) % i32(consts::VOXELS_PER_DIM);
    let update_max = ((vec3<i32>(i32(consts::VOXELS_PER_DIM)) - min(bind::cascade_info.redraw.xyz, vec3<i32>(0)) * i32(consts::VOXELS_PER_TILE_DIM) - 1) % i32(consts::VOXELS_PER_DIM)) + 1;
    let adjusted_grid_id = update_min + vec3<i32>(g_id);

    let local_voxel = adjusted_grid_id;
    let target_point = addr::voxel_local_to_local_position(local_voxel);
    let half_vec = vec3<f32>(0.5 * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM * consts::SUBVOXELS_PER_VOXEL_DIM));
    let worst_dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT - 1u) * 9999.0;

    var best_dist_sq = worst_dist * worst_dist;
    var best_write = vec4<i32>(0);
    var local_jump_source: vec3<i32>;

    for (var x=-1; x<=1; x++) {
        local_jump_source.x = local_voxel.x + x * jfa_params.jump_size;
        if local_jump_source.x < update_min.x || local_jump_source.x >= update_max.x {
            continue;
        }
        for (var y=-1; y<=1; y++) {
            local_jump_source.y = local_voxel.y + y * jfa_params.jump_size;
            if local_jump_source.y < update_min.y || local_jump_source.y >= update_max.y {
                continue;
            }
            for (var z=-1; z<=1; z++) {
                local_jump_source.z = local_voxel.z + z * jfa_params.jump_size;
                if local_jump_source.z < update_min.z || local_jump_source.z >= update_max.z {
                    continue;
                }

                let jump_source_coords = addr::voxel_local_to_maybe_grid(local_jump_source);
                if jump_source_coords.x > -1 { // source is within our live grid
                    let source_data = textureLoad(bind::nearest_jfa, jump_source_coords);

                    if source_data.a != 0 { // source has a seed 
                        let local_seed_voxel = vec3<i32>(local_jump_source + source_data.xyz);
                        let seed_coords = addr::voxel_local_to_maybe_grid(local_seed_voxel);

                        if seed_coords.x > -1 { // seed is within our live grid
                            let seed_value = textureLoad(bind::seed_jfa, seed_coords).rg;

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
                                            let local_seed_position = addr::subvoxel_local_position(local_seed_voxel, vec3<u32>(sx,sy,sz));
                                            let dist_sq = addr::distance_squared(local_seed_position, target_point);

                                            if dist_sq < best_dist_sq {
                                                best_dist_sq = dist_sq;
                                                best_write = vec4<i32>(local_seed_voxel - local_voxel, i32(shift + 1u));
                                            }
                                        }
                                    }
                                }                                    
                            }
                        }
                    }
                }
            }
        }
    }

    textureStore(bind::nearest_jfa, addr::voxel_local_to_maybe_grid(local_voxel), best_write);
}