#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::addressing as addr

// make an sdf !

@group(1) @binding(0)
var output_texture: texture_storage_3d<r32float, read_write>;

@compute @workgroup_size(8,8,8)
fn output(@builtin(global_invocation_id) local_voxel: vec3<u32>) {

    let target_point = addr::voxel_local_to_local_position(vec3<i32>(local_voxel));
    let voxel_coords = addr::voxel_local_to_grid(local_voxel);
    let source_data = textureLoad(bind::nearest_jfa, voxel_coords);

    var dist = bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT - 1u);
    if source_data.a != 0 {
        let local_seed_voxel = vec3<i32>(local_voxel) + source_data.xyz;
        let subvoxel = addr::subvoxel_index_to_subvoxel(u32(source_data.a - 1));
        let local_seed_position = addr::subvoxel_local_position(local_seed_voxel, subvoxel);
        let dist_sq = addr::distance_squared(local_seed_position, target_point);
        dist = sqrt(dist_sq);
    }

    let cascade_offset = consts::VOXELS_PER_DIM * bind::cascade_info.index;
    let write_index = vec3<i32>(local_voxel + vec3<u32>(cascade_offset, 0u, 0u));

    // textureStore(output_texture, write_index, vec4<f32>(dist - 0.5 * bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT) / f32(consts::VOXELS_PER_DIM), 0.0, 0.0, 1.0));
    // sqrt(2) / 2 == 0.707 is the worst case overestimate of distance from the jump stitch
    textureStore(output_texture, write_index, vec4<f32>(dist * 0.70711 - 0.5 * bind::cascade_info.tile_size * f32(consts::TILE_DIM_COUNT) / f32(consts::VOXELS_PER_DIM), 0.0, 0.0, 1.0));
}