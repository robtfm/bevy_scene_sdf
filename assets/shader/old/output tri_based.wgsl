#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::dist as dist

@group(1) @binding(0)
var output_texture: texture_storage_3d<r32float, read_write>;

@compute @workgroup_size(8,8,8)
fn output(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let id = vec3<i32>(g_id);
    let target_point = (vec3<f32>(id) + 0.5) * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM);
    let source_id = textureLoad(bind::secondary_jfa, id).r;
    var dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT) * 1.75;
    var in_front: f32 = 1.0;
    if source_id != 0u {
        var tri = dist::build_tri(source_id - 1u);
        dist = dist::tri_distance(&tri, target_point, dist*dist);
        in_front = dist::tri_infront(tri, target_point);
    }

    let cascade_offset = consts::VOXELS_PER_DIM * bind::cascade_info.index;
    let write_index = id + vec3<i32>(i32(cascade_offset), 0, 0);

    let write_val = sqrt(dist) * in_front;

    textureStore(output_texture, write_index, vec4<f32>(write_val, 0.0, 0.0, 1.0));
}