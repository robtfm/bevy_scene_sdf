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
    var count = 0u;
    var i = 0u;

    // gather ids
    let target_point = (vec3<f32>(id) + 0.5) * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM);
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
                                var found = false;
                                for (i=0u; i<count; i++) {
                                    if ids[i] == source_id {
                                        found = true;
                                        i = count;
                                    }
                                }

                                if !found {
                                    ids[count] = source_id;
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // bias +/- 1/20th of a voxel to prefer triangles that show target point is outside 
    let epsilon = bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM) * 0.05;
    var best_dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT) * 1.75;
    var best_id = 0u;
    for (i=0u; i<count; i++) {
        let source_id = ids[i];
        var tri = dist::build_tri(source_id - 1u);
        let dist = dist::tri_distance(&tri, target_point, best_dist) - epsilon * dist::tri_infront(tri, target_point);
        if dist + 2.0 * epsilon < best_dist {
            best_dist = dist + 0.0001;
            best_id = source_id;
        }
    }

    let ratio = f32(0xFFFFFFFFu) / (bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT) * 1.75);
    if jfa_params.from_primary == 1u {
        textureStore(bind::secondary_jfa, id, vec4<u32>(best_id, u32(sqrt(best_dist) * ratio), 0u, 1u));
    } else {
        textureStore(bind::primary_jfa, id, vec4<u32>(best_id, u32(sqrt(best_dist) * ratio), 0u, 1u));
    }
}