#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::addressing as addr

// join the update region to the original region
// https://www.docdroid.net/YNntL0e/godot-sdfgi-pdf#page=30 claim that you only need to test the nearest
// i find that quite flaky, it's significantly better checking others as well
// this still isn't perfect but errors seem to be smaller.
// note we add a correction in the output stage anyway since it is still not perfect, but without additional
// samples the correction needs to be sqrt(2)/2 = 0.707 which is very big.

@compute @workgroup_size(8,8,8)
fn stitch(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let local_voxel = vec3<i32>(addr::voxel_grid_to_local(g_id));

    let stitch_axis = bind::cascade_info.redraw.xyz != vec3<i32>(0);
    let stitch_dir = vec3<i32>(stitch_axis);
    let stitch_mask = vec3<i32>(1) - stitch_dir;

    let stitch_right = (i32(consts::VOXELS_PER_DIM) - bind::cascade_info.redraw.xyz * i32(consts::VOXELS_PER_TILE_DIM)) % i32(consts::VOXELS_PER_DIM);
    let stitch_left = stitch_right - stitch_dir;

    let target_point = addr::voxel_local_to_local_position(local_voxel);
    let worst_dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT - 1u) * 9999.0;
    let write_address = addr::voxel_local_to_maybe_grid(local_voxel);
    var best_dist_sq = worst_dist * worst_dist;

    // init to current
    var best_write = textureLoad(bind::nearest_jfa, write_address);
    if best_write.a != 0 {
        if all(best_write.xyz == vec3(0)) {
            // we're a seed cell (and must already be calculated)
            return;
        }

        let current = addr::check_source(target_point, local_voxel, local_voxel, best_dist_sq);
        best_dist_sq = current.best_dist_sq;
        best_write = current.write_value;
    }

    var stitch_source_voxel = vec3<i32>(0);
    if all(local_voxel >= stitch_right) {
        stitch_source_voxel = local_voxel * stitch_mask + stitch_left;
    } else {
        stitch_source_voxel = local_voxel * stitch_mask + stitch_right;
    }

    let stitch_offset = local_voxel - stitch_source_voxel;
    let jump_size = abs(stitch_offset.x + stitch_offset.y + stitch_offset.z);

    let iter_min = -abs(stitch_mask);
    let iter_max = abs(stitch_mask);
    var local_jump_source = vec3<i32>(0);

    if true {
        // this really should be better than checking diagonals around the stitch point but it seems to be the same
        // i think i did something wrong here...

        // check up to 2x away for stitch cell
        let res = addr::check_source(target_point, local_voxel, stitch_source_voxel, best_dist_sq * 2.0);
        if res.best_dist_sq < best_dist_sq {
            best_dist_sq = res.best_dist_sq;
            best_write = res.write_value;
        }

        if res.best_dist_sq < best_dist_sq * 2.0 {
            // check points in a circle around the stitch point at distance from stitch point equal to distance from stitch point to stitch seed
            let stitch_distance = sqrt(addr::distance_squared(addr::voxel_local_to_local_position(stitch_source_voxel), addr::voxel_local_to_local_position(local_voxel + res.write_value.xyz)));
            let stitch_voxels =  stitch_distance * f32(consts::VOXELS_PER_TILE_DIM) / bind::cascade_info.tile_size;

            var jumps = array(
                vec2(-0.707, -0.707),
                vec2(0.0, -1.0),
                vec2(0.707, -0.707),
                vec2(-1.0, 0.0),
                vec2(1.0, 0.0),
                vec2(-0.707, 0.707),
                vec2(0.0, 1.0),
                vec2(0.707, 0.707),
            );

            for (var i=0u; i<8u; i++) {
                let jump = jumps[i];
                local_jump_source = stitch_source_voxel;
                if stitch_axis.x {
                    local_jump_source += vec3<i32>(vec3<f32>(0.0, jump.x, jump.y) * stitch_distance);
                } else if stitch_axis.y {
                    local_jump_source += vec3<i32>(vec3<f32>(jump.x, 0.0, jump.y) * stitch_distance);
                } else if stitch_axis.z {
                    local_jump_source += vec3<i32>(vec3<f32>(0.0, jump.x, jump.y) * stitch_distance);
                }

                if all(local_jump_source > vec3(0) && local_jump_source < vec3(i32(consts::VOXELS_PER_DIM))) {
                    let res = addr::check_source(target_point, local_voxel, stitch_source_voxel, best_dist_sq);
                    if res.best_dist_sq < best_dist_sq {
                        best_dist_sq = res.best_dist_sq;
                        best_write = res.write_value;
                    }
                }
            }
        }
    } else {
        // check grid based on dist to stitch
        for (var x=iter_min.x; x<=iter_max.x; x++) {
            local_jump_source.x = stitch_source_voxel.x + x * jump_size;
            if local_jump_source.x < 0 || local_jump_source.x >= i32(consts::VOXELS_PER_DIM) {
                continue;
            }
            for (var y=iter_min.y; y<=iter_max.y; y++) {
                local_jump_source.y = stitch_source_voxel.y + y * jump_size;
                if local_jump_source.y < 0 || local_jump_source.y >= i32(consts::VOXELS_PER_DIM) {
                    continue;
                }
                for (var z=iter_min.z; z<=iter_max.z; z++) {
                    local_jump_source.z = stitch_source_voxel.z + z * jump_size;
                    if local_jump_source.z < 0 || local_jump_source.z >= i32(consts::VOXELS_PER_DIM) {
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
    }

    textureStore(bind::nearest_jfa, write_address, best_write);
}