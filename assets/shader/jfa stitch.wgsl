#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::addressing as addr

// join the update region to the original region
// https://www.docdroid.net/YNntL0e/godot-sdfgi-pdf#page=30 claim that you only need to test the nearest
// i find that quite flaky, it's significantly better checking diagonals as well (9 samples)
// this still isn't perfect but errors seem to be pretty small

@compute @workgroup_size(8,8,8)
fn stitch(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let local_voxel = vec3<i32>(addr::voxel_grid_to_local(g_id));

    let stitch_dir = vec3<i32>(bind::cascade_info.redraw.xyz != vec3<i32>(0));
    let stitch_mask = vec3<i32>(1) - stitch_dir;

    let stitch_right = (i32(consts::VOXELS_PER_DIM) - bind::cascade_info.redraw.xyz * i32(consts::VOXELS_PER_TILE_DIM)) % i32(consts::VOXELS_PER_DIM);
    let stitch_left = stitch_right - stitch_dir;

    let target_point = addr::voxel_local_to_local_position(local_voxel);
    let worst_dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT - 1u) * 9999.0;
    var best_dist_sq = worst_dist * worst_dist;
    var best_write = vec4<i32>(0,0,0,0);

    let jump_source_coords = addr::voxel_local_to_maybe_grid(local_voxel);
    let source_data = textureLoad(bind::nearest_jfa, jump_source_coords);
    if source_data.a > 0 {
        let local_seed_voxel = vec3<i32>(local_voxel + source_data.xyz);
        best_write = vec4<i32>(local_seed_voxel - local_voxel, source_data.a);
        let subvoxel = addr::subvoxel_index_to_subvoxel(u32(source_data.a - 1));
        let local_seed_position = addr::subvoxel_local_position(local_seed_voxel, subvoxel);
        let dist_sq = addr::distance_squared(local_seed_position, target_point);
        best_dist_sq = dist_sq;
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

                let jump_source_coords = addr::voxel_local_to_maybe_grid(local_jump_source);
                let source_data = textureLoad(bind::nearest_jfa, jump_source_coords);
                if source_data.a > 0 {
                    let stitch_seed_voxel = local_jump_source + source_data.xyz;
                    let stitch_seed_subvoxel = addr::subvoxel_index_to_subvoxel(u32(source_data.a - 1));
                    let stitch_seed_position = addr::subvoxel_local_position(stitch_seed_voxel, stitch_seed_subvoxel);
                    let dist_sq = addr::distance_squared(stitch_seed_position, target_point);
                    let offset = stitch_seed_voxel - local_voxel;
                    if dist_sq < best_dist_sq && all(abs(offset) < vec3<i32>(128)) { 
                        best_dist_sq = dist_sq;
                        best_write = vec4<i32>(offset, source_data.a);
                    }
                }
            }
        }
    }

    textureStore(bind::nearest_jfa, addr::voxel_local_to_maybe_grid(local_voxel), best_write);
}