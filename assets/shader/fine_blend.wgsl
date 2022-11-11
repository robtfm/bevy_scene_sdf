// merge fine output into a single final seed texture

// todo
// - parallel over workgroups with i division, take up to end of your working tile when you reach upper limit

#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::addressing as addr


@compute @workgroup_size(8,8,8)
//@compute @workgroup_size(4,4,4)
fn fine_blend(@builtin(local_invocation_id) invocation_id: vec3<u32>, @builtin(local_invocation_index) thread_id: u32) {
    var current_tile = 0u;
    var data = vec2<u32>(0u);
    let redraw = bind::cascade_info.redraw.xyz;

    for (var i=0u; i<consts::MAX_TILES; i++) {
        let next_tile = bind::coarse_tile_counts[i].x;

        for (; current_tile < min(next_tile, consts::TILE_COUNT); current_tile++) {
            let local_tile = addr::tile_index_to_local(current_tile);

            let valid = 
                redraw == vec3<i32>(0) 
                || 
                (redraw < vec3<i32>(0) && vec3<i32>(local_tile) < -redraw)
                ||
                (redraw > vec3<i32>(0) && local_tile >= consts::TILE_DIM_COUNT - vec3<u32>(redraw))
            ;

            if !all(valid) {
                data = vec2<u32>(0u, 0u);
                continue;
            }

            let write_index = addr::voxel_local_to_grid(local_tile * consts::VOXELS_PER_TILE_DIM + invocation_id);

            textureStore(
                bind::seed_jfa,
                write_index, 
                vec4<u32>(data, 0u, 1u)
            );
            var write_seed = 1;
            if all(data == vec2<u32>(0u)) {
                write_seed = 0;
            }
            // xyz = 0 -> seed is self
            textureStore(
                bind::nearest_jfa,
                write_index, 
                vec4<i32>(vec3<i32>(0), write_seed)
            );

            data = vec2<u32>(0u, 0u);
        }

        if next_tile == 0xFFFFFFFFu {
            i = consts::MAX_TILES;
            continue;
        }

        let sdf_tile_offset = i * consts::VOXELS_PER_TILE;
        let index = sdf_tile_offset + thread_id;

        data.x |= bind::fine_output[index*2u];
        data.y |= bind::fine_output[index*2u+1u];
    }
}