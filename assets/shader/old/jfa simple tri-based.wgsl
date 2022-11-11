#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::dist as dist

struct JfaParams {
    jump_size: i32,
    from_primary: u32,
}

@group(1) @binding(0)
var<uniform> jfa_params: JfaParams;

var<private> ids: array<u32, 27>;

@compute @workgroup_size(8,8,8)
fn jfa(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let id = vec3<i32>(g_id);
    let target_point = (vec3<f32>(id) + 0.5) * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM);
    var best_dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT - 1u);
    best_dist = best_dist * best_dist;
    var best_id = 0u;
    var count = 0u;

    // bias +/- 1/20th of a voxel to prefer triangles that show target point is outside 
    let epsilon = bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM) * 0.05;

    var source: vec3<i32>;
    for (var x=-1; x<=1; x++) {
        source.x = id.x + x * jfa_params.jump_size;
        if source.x >= 0 && source.x < i32(consts::VOXELS_PER_DIM) {
            for (var y=-1; y<=1; y++) {
                source.y = id.y + y * jfa_params.jump_size;
                if source.y >= 0 && source.y < i32(consts::VOXELS_PER_DIM) {
                    for (var z=-1; z<=1; z++) {
                        source.z = id.z + z * jfa_params.jump_size;
                        if source.z >= 0 && source.z < i32(consts::VOXELS_PER_DIM) {
                            count++;
                            var source_id: u32;
                            if jfa_params.from_primary == 1u {
                                source_id = textureLoad(bind::primary_jfa, source).r;
                            } else {
                                source_id = textureLoad(bind::secondary_jfa, source).r;
                            }

                            if source_id != 0u {
                                var tri: dist::TriAccel = dist::build_tri(source_id - 1u);
                                let dist = dist::tri_distance(&tri, target_point, best_dist) - dist::tri_infront(tri, target_point) * epsilon;
                                if dist + 1.0 * epsilon < best_dist {
                                    best_dist = dist;
                                    best_id = source_id;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let ratio = f32(0xFFFFFFFFu) / (bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT) * 2.0);
    if jfa_params.from_primary == 1u {
        textureStore(bind::secondary_jfa, id, vec4<u32>(best_id, u32(sqrt(best_dist) * ratio), 0u, 1u));
    } else {
        textureStore(bind::primary_jfa, id, vec4<u32>(best_id, u32(sqrt(best_dist) * ratio), 0u, 1u));
    }
}