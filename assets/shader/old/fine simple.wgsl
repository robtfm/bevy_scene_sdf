// todo 
// * run bounds by bb, 
// - remove bb test in intersect (currently bounds are per voxel, need to move to subvoxel)
// - precompute axis test per axis pair
// - pick output format for subvoxel count

#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::dist as dist
#import sdf::intersect as intersect

var<workgroup> voxels_r: array<atomic<u32>, consts::VOXELS_PER_TILE>;
var<workgroup> voxels_g: array<atomic<u32>, consts::VOXELS_PER_TILE>;

fn tile_index(xyz: vec3<u32>) -> u32 {
    return (((xyz.z * consts::TILE_DIM_COUNT) + xyz.y) * consts::TILE_DIM_COUNT) + xyz.x;
}

fn tile_index_to_xyz(index: u32) -> vec3<u32> {
    return vec3<u32>(
        index % consts::TILE_DIM_COUNT, 
        (index / consts::TILE_DIM_COUNT) % consts::TILE_DIM_COUNT, 
        index / (consts::TILE_DIM_COUNT * consts::TILE_DIM_COUNT), 
    );
}

@compute @workgroup_size(128,1,1)
fn fine_raster(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_id: u32) {
    let FINE_THREADS=128u;

    let params = bind::coarse_tile_counts[workgroup_id.x];

    let tile_index = params.x;
    let tile_xyz = tile_index_to_xyz(params.x);

    if thread_id == 0u {
        // init our local texture
        for (var i = 0u; i < consts::VOXELS_PER_TILE; i++) {
            atomicStore(&voxels_r[i], 0u);
            atomicStore(&voxels_g[i], 0u);
        }
    }

    workgroupBarrier();

    var tri_start_index = 0u;
    if workgroup_id.x > 0u {
        tri_start_index = bind::coarse_tile_counts[workgroup_id.x - 1u].y;
    }
    let tri_end_index = params.y;

    let tile_min = vec3<f32>(tile_xyz) * bind::cascade_info.tile_size;
    let v = bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM);
    let subv = v / f32(consts::SUBVOXELS_PER_VOXEL_DIM);
    let epsilon = 1e-5 * bind::cascade_info.tile_size;

    for (var tri_id_pos = tri_start_index + thread_id; tri_id_pos < tri_end_index; tri_id_pos += FINE_THREADS) {
        let tri_id = bind::coarse_ids[tri_id_pos];
        var tri: intersect::TriAccel = intersect::build_tri(tri_id);

        let bb_min = vec3<u32>(max(vec3<i32>(0), vec3<i32>(floor((tri.min - tile_min) / v))));
        let bb_max = vec3<u32>(min(vec3<i32>(i32(consts::VOXELS_PER_TILE_DIM)), vec3<i32>(ceil((tri.max + epsilon - tile_min) / v))));

        for (var x=bb_min.x; x < bb_max.x; x++) {
            for (var y=bb_min.y; y < bb_max.y; y++) {
                for (var z=bb_min.z; z < bb_max.z; z++) {
                    let index = (((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x;

                    let current_r = atomicLoad(&voxels_r[index]);
                    let current_g = atomicLoad(&voxels_g[index]);

                    var write_value_r: u32 = 0u;
                    var write_value_g: u32 = 0u;

                    for (var sx=0u; sx < consts::SUBVOXELS_PER_VOXEL_DIM; sx++) {
                        for (var sy=0u; sy < consts::SUBVOXELS_PER_VOXEL_DIM; sy++) {
                            for (var sz=0u; sz < consts::SUBVOXELS_PER_VOXEL_DIM; sz++) {
                                let target_point = tile_min + (vec3<f32>(vec3<u32>(x, y, z) * consts::SUBVOXELS_PER_VOXEL_DIM + vec3<u32>(sx, sy, sz)) + vec3<f32>(0.5)) * subv;
                                if intersect::voxel_test(target_point, &tri) {
                                    let shift = (((sz * consts::SUBVOXELS_PER_VOXEL_DIM) + sy) * consts::SUBVOXELS_PER_VOXEL_DIM) + sx;
                                    if shift < 32u {
                                        write_value_r |= 1u << shift;
                                    } else {
                                        write_value_g |= 1u << (shift - 32u);
                                    }
                                }
                            }
                        }
                    }

                    atomicOr(&voxels_r[(((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x], write_value_r);
                    atomicOr(&voxels_g[(((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x], write_value_g);
                }
            }                    
        }
    }

    workgroupBarrier();

    // write to sdf output
    if thread_id == 0u {
        let sdf_tile_offset = workgroup_id.x * consts::VOXELS_PER_TILE;
        for (var x=0u; x < consts::VOXELS_PER_TILE_DIM; x++) {
            for (var y=0u; y < consts::VOXELS_PER_TILE_DIM; y++) {
                for (var z=0u; z < consts::VOXELS_PER_TILE_DIM; z++) {
                    let index = (((z * consts::VOXELS_PER_TILE_DIM) + y) * consts::VOXELS_PER_TILE_DIM) + x;

                    let r = atomicLoad(&voxels_r[index]);
                    let g = atomicLoad(&voxels_g[index]);

                    bind::fine_output[(index + sdf_tile_offset) * 2u] = r;
                    bind::fine_output[(index + sdf_tile_offset) * 2u + 1u] = g;
                }
            }
        }
    }
}
