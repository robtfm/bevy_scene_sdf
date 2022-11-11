// fill subvoxels within the tile

// todo 
// * run bounds by bb, 
// - remove bb test in intersect (currently bounds are per voxel, need to move to subvoxel)
// - precompute axis test per axis pair
// - pick output format for subvoxel count

#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::intersect as intersect
#import sdf::addressing as addr

var<workgroup> voxels_r: array<atomic<u32>, consts::VOXELS_PER_TILE>;
var<workgroup> voxels_g: array<atomic<u32>, consts::VOXELS_PER_TILE>;

@compute @workgroup_size(128,1,1)
fn fine_raster(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) thread_id: u32) {
    let FINE_THREADS=128u;

    let params = bind::coarse_tile_counts[workgroup_id.x];

    let tile_index = params.x;
    let tile_xyz = addr::tile_index_to_local(params.x);

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

        let bb_min = vec3<u32>(max(vec3<i32>(0), vec3<i32>(floor((tri.min - tile_min) / subv))));
        let bb_max = vec3<u32>(min(vec3<i32>(i32(consts::VOXELS_PER_TILE_DIM * consts::SUBVOXELS_PER_VOXEL_DIM)), vec3<i32>(ceil((tri.max - tile_min + epsilon) / subv))));

        for (var x=bb_min.x; x < bb_max.x; x++) {
            for (var y=bb_min.y; y < bb_max.y; y++) {
                let z_range = intersect::subvoxel_z_range(tile_min, x, y, subv, &tri);
                for (var z=max(bb_min.z, z_range.x); z < min(bb_max.z, z_range.y); z++) {
                // for (var z=bb_min.z; z < bb_max.z; z++) {
                    let voxel = vec3<u32>(x, y, z) / consts::SUBVOXELS_PER_VOXEL_DIM;
                    let subvoxel = vec3<u32>(x, y, z) % consts::SUBVOXELS_PER_VOXEL_DIM;

                    let index = (((voxel.z * consts::VOXELS_PER_TILE_DIM) + voxel.y) * consts::VOXELS_PER_TILE_DIM) + voxel.x;
                    let target_point = tile_min + (vec3<f32>(voxel * consts::SUBVOXELS_PER_VOXEL_DIM + subvoxel) + vec3<f32>(0.5)) * subv;

                    if intersect::voxel_test(target_point, &tri) {
                        let shift = (((subvoxel.z * consts::SUBVOXELS_PER_VOXEL_DIM) + subvoxel.y) * consts::SUBVOXELS_PER_VOXEL_DIM) + subvoxel.x;
                        if shift < 32u {
                            atomicOr(&voxels_r[index], 1u << shift);
                        } else {
                            atomicOr(&voxels_g[index], 1u << (shift - 32u));
                        }
                    }
                }
            }                    
        }
    }

    workgroupBarrier();

    // write to seed output
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
