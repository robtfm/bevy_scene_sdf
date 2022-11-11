#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::dist as dist
#import sdf::intersect as intersect

var<workgroup> voxels: array<atomic<u32>, consts::VOXELS_PER_TILE>;

fn tile_index(xyz: vec3<u32>) -> u32 {
    return (((xyz.z * consts::TILE_DIM_COUNT) + xyz.y) * consts::TILE_DIM_COUNT) + xyz.x;
}

fn tile_coords(workgroup_id: u32) -> vec3<u32> {
    var ix = workgroup_id;
    let x_slices = u32(abs(bind::cascade_info.redraw.x));
    let x_slice_size = consts::TILE_DIM_COUNT * consts::TILE_DIM_COUNT;
    if x_slices * x_slice_size > ix {
        let slice = ix / x_slice_size;
        ix = ix - slice * x_slice_size;
        if bind::cascade_info.redraw.x > 0 { // draw right side of cube
            return vec3<u32>(consts::TILE_DIM_COUNT - slice - 1u, ix / consts::TILE_DIM_COUNT, ix % consts::TILE_DIM_COUNT);
        } else { // draw left side of cube
            return vec3<u32>(consts::TILE_DIM_COUNT, ix / consts::TILE_DIM_COUNT, ix % consts::TILE_DIM_COUNT);
        }
    }

    ix -= x_slice_size * x_slices;

    let y_slices = u32(abs(bind::cascade_info.redraw.y));
    let y_slice_size = (consts::TILE_DIM_COUNT - x_slices) * consts::TILE_DIM_COUNT;
    if y_slices * y_slice_size > ix {
        let slice = ix / y_slice_size;
        ix = ix - slice * y_slice_size;

        if bind::cascade_info.redraw.y > 0 { // draw top of cube
            return vec3<u32>((ix / consts::TILE_DIM_COUNT) + u32(max(bind::cascade_info.redraw.x, 0)), consts::TILE_DIM_COUNT - slice - 1u, ix % consts::TILE_DIM_COUNT);
        } else {
            return vec3<u32>((ix / consts::TILE_DIM_COUNT) + u32(max(bind::cascade_info.redraw.x, 0)), slice, ix % consts::TILE_DIM_COUNT);
        }
    }

    ix -= y_slice_size * y_slices;

    let z_slices = u32(abs(bind::cascade_info.redraw.z));
    let z_slice_size = (consts::TILE_DIM_COUNT - x_slices) * (consts::TILE_DIM_COUNT - y_slices);
    if z_slices * z_slice_size > ix {
        let slice = ix / z_slice_size;
        ix = ix - slice * z_slice_size;

        if bind::cascade_info.redraw.z > 0 { // draw back of cube
            return vec3<u32>(ix / (consts::TILE_DIM_COUNT - y_slices) + u32(max(bind::cascade_info.redraw.x, 0)), ix % (consts::TILE_DIM_COUNT - y_slices) + u32(max(bind::cascade_info.redraw.y, 0)), consts::TILE_DIM_COUNT - slice - 1u);
        } else { // draw front of cube
            return vec3<u32>(ix / (consts::TILE_DIM_COUNT - y_slices) + u32(max(bind::cascade_info.redraw.x, 0)), ix % (consts::TILE_DIM_COUNT - y_slices) + u32(max(bind::cascade_info.redraw.y, 0)), slice);
        }
    }

    //panic
    return vec3<u32>(999u,999u,999u);
}

@compute @workgroup_size(32,1,1)
fn fine_raster(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_id: u32) {
    let tile_xyz = tile_coords(workgroup_id.x);
    let tile_index = tile_index(tile_xyz);

    if thread_id == 0u {
        // init our local texture
        for (var i = 0u; i < consts::VOXELS_PER_TILE; i++) {
            atomicStore(&voxels[i], 0xFFFFFFFFu); 
        }
    }

    workgroupBarrier();

    var tri_start_index = 0u;
    if tile_index > 0u {
        tri_start_index = bind::coarse_counts[tile_index - 1u];
    }
    let tri_end_index = bind::coarse_counts[tile_index];

    let tile_min = vec3<f32>(tile_xyz) * bind::cascade_info.tile_size;
    let t = bind::cascade_info.tile_size;
    let v = t / f32(consts::VOXELS_PER_TILE_DIM);

    // convert dist_sq to u16 range (25% buffer on max theoretical distance)
    let max_dist = bind::cascade_info.tile_size * 1.5;
    let max_dist_sq = max_dist * max_dist;
    let dist_ratio = f32(0xFFFF) / max_dist;
    let dist_sq_ratio = f32(0xFFFF) / max_dist_sq;

    for (var tri_id_pos = tri_start_index + thread_id; tri_id_pos < tri_end_index; tri_id_pos += 32u) {
        let tri_id = bind::coarse_ids[tri_id_pos];
        var tri: dist::TriAccel = dist::build_tri(tri_id);
        for (var x=0u; x < consts::VOXELS_PER_TILE_DIM; x++) {
            for (var y=0u; y < consts::VOXELS_PER_TILE_DIM; y++) {
                for (var z=0u; z < consts::VOXELS_PER_TILE_DIM; z++) {
                    let target_point = tile_min + (vec3<f32>(vec3<u32>(x, y, z)) + 0.5) * v;
                    if !intersect::voxel_test(target_point, &tri) {
                        continue;
                    }

                    let best_dist = dist::tri_distance(&tri, target_point, max_dist_sq);
                    let fwd = 0.0; //dist::tri_infront(tri, target_point);

                    // 0 - 65535
                    let ranged_dist = u32((sqrt(best_dist) - fwd) * dist_ratio);
                    // [ 31  ..  16  |  15    ..    0 ]
                    // [  distance   |  local tri_ix  ]
                    let write_value = (ranged_dist << 16u) + (tri_id_pos - tri_start_index);
                    atomicMin(&voxels[(((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x], write_value);
                }
            }                    
        }
    }

    workgroupBarrier();

    // write to sdf output
    if thread_id == 0u {
        let sdf_tile_offset = tile_xyz * consts::VOXELS_PER_TILE_DIM;
        for (var x=0u; x < consts::VOXELS_PER_TILE_DIM; x++) {
            for (var y=0u; y < consts::VOXELS_PER_TILE_DIM; y++) {
                for (var z=0u; z < consts::VOXELS_PER_TILE_DIM; z++) {
                    // let distance = (atomicLoad(&voxels[(((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x]) & 0xFFFF0000u) >> 16u;
                    let local_tri_pos = atomicLoad(&voxels[(((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x]) & 0xFFFFu;

                    var tri_id = 0u;
                    if local_tri_pos != 0xFFFFu {
                        tri_id = bind::coarse_ids[tri_start_index + local_tri_pos] + 1u;
                    }
                    
                    textureStore(
                        bind::primary_jfa, 
                        vec3<i32>(sdf_tile_offset + vec3<u32>(x, y, z)), 
                        // vec4<u32>((((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x, 0u, 0u, 1u)
                        // vec4<u32>(distance, 0u, 0u, 1u)
                        vec4<u32>(tri_id, 0u, 0u, 1u)
                    );
                }
            }
        }
    }
}
