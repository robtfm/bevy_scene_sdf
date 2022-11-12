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

    let update_min = vec3<i32>(0); //(vec3<i32>(i32(consts::VOXELS_PER_DIM)) - max(bind::cascade_info.redraw.xyz, vec3<i32>(0)) * i32(consts::VOXELS_PER_TILE_DIM)) % i32(consts::VOXELS_PER_DIM);
    let update_max = vec3<i32>(i32(consts::VOXELS_PER_DIM + 1u)); //((vec3<i32>(i32(consts::VOXELS_PER_DIM)) - min(bind::cascade_info.redraw.xyz, vec3<i32>(0)) * i32(consts::VOXELS_PER_TILE_DIM) - 1) % i32(consts::VOXELS_PER_DIM)) + 1;

    let local_voxel = adjusted_grid_id;
    let target_point = addr::voxel_local_to_local_position(local_voxel);
    let voxel_half_vec = vec3<f32>(0.5 * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM));
    let worst_dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT - 1u) * 9999.0;
    let write_address = addr::voxel_local_to_maybe_grid(local_voxel);
    var best_dist_sq = worst_dist * worst_dist;

    var best_write = textureLoad(bind::nearest_jfa, write_address);
    if best_write.a != 0 {
        if all(best_write.xyz == vec3(0)) && best_write.a != i32(consts::SUBVOXELS_PER_VOXEL_DIM * consts::SUBVOXELS_PER_VOXEL_DIM * consts::SUBVOXELS_PER_VOXEL_DIM + 1u) {
            // we're a seed cell, and already calculated
            return;
        }

        let current = addr::check_source(target_point, local_voxel, local_voxel, best_dist_sq);
        best_dist_sq = current.best_dist_sq;
        best_write = current.write_value;
    }

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

                if x == 0 && y == 0 && z == 0 {
                    continue;
                }

                let res = addr::check_source(target_point, local_voxel, local_jump_source, best_dist_sq);
                if res.best_dist_sq < best_dist_sq {
                    best_dist_sq = res.best_dist_sq;
                    best_write = res.write_value;
                }
            }
        }
    }

    textureStore(bind::nearest_jfa, write_address, best_write);
}