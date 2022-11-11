#import bevy_core_pipeline::fullscreen_vertex_shader as fs
#import bevy_pbr::mesh_view_types

#import sdf::consts as consts
#import sdf::types

@group(0) @binding(0)
var<uniform> view: bevy_pbr::mesh_view_types::View;

@group(0) @binding(1)
var<storage> cascades_info: sdf::types::CascadeInfos;

@group(0) @binding(2)
var sdf_texture: texture_3d<f32>;

@group(0) @binding(3)
var sdf_sampler: sampler;

struct SampleResult {
    outside_cascade: bool,
    distance: f32,
    
    debug_querypoint: vec3<f32>,
}

fn sample_distance(pos: vec3<f32>, cascade: u32) -> SampleResult {
    var res: SampleResult;

    let info = cascades_info.cascades[cascade];

    let cascade_size = f32(consts::TILE_DIM_COUNT) * info.tile_size;
    let border = 0.5 / f32(consts::VOXELS_PER_DIM);
    let cascade_min = vec3<f32>(info.origin.xyz) * info.tile_size;
    // let cascade_max = cascade_min + cascade_size;

    // let nearest_point = clamp(pos, cascade_min, cascade_max);
    // within cascade -> 0-1 
    var texture_point = (pos - cascade_min) / cascade_size;

    // [half a voxel, 1-half a voxel]
    // let clamped_texture_point = clamp(texture_point, vec3<f32>(0.0), vec3<f32>(1.0));
    let clamped_texture_point = clamp(texture_point, vec3<f32>(border), vec3<f32>(1.0 - border));
    res.outside_cascade = any(clamped_texture_point != texture_point);

    let cascade_texture_point = vec3<f32>((
        clamped_texture_point.x + f32(cascade)) / f32(cascades_info.count),
        clamped_texture_point.yz
    );
    res.distance = textureSample(sdf_texture, sdf_sampler, cascade_texture_point).r;

    res.debug_querypoint = cascade_texture_point;
    return res;
}

@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let origin = view.inverse_view_proj * vec4<f32>((in.uv - 0.5) * vec2<f32>(2.0, -2.0), 1.0, 1.0);
    let origin = origin.xyz / origin.w;
    let offset = view.inverse_view_proj * vec4<f32>((in.uv - 0.5) * vec2<f32>(2.0, -2.0), 0.5, 1.0);
    let offset = offset.xyz / offset.w;
    let ray = normalize(offset - origin);

    let max_steps = 50;

    var min_dist = 1e20;
    var distance_sq = 0.0;
    var cascade = 0u;
    let base_hit_threshold = 2.0 * 0.5 * cascades_info.cascades[cascade].tile_size / f32(consts::VOXELS_PER_TILE_DIM * consts::SUBVOXELS_PER_VOXEL_DIM);
    var hit_threshold = base_hit_threshold;
    var min_step_size = base_hit_threshold;
    var exited_cascade = false;
    var steps: i32;

    var dist = hit_threshold * 2.0;
    var total_dist = dist;

    var debug: vec3<f32>;
    var pos = origin + dist * ray;

    for (steps = 0; cascade < cascades_info.count && dist > hit_threshold && steps < max_steps; steps++) {
        let res = sample_distance(pos, cascade);

        exited_cascade = res.outside_cascade;
        if exited_cascade {
            cascade++;
            steps -= 2;
            if cascade < cascades_info.count {
                total_dist -= max(dist, min_step_size);
                pos = origin + total_dist * ray;
                min_step_size = 2.0 * 0.5 * cascades_info.cascades[cascade].tile_size / f32(consts::VOXELS_PER_TILE_DIM * consts::SUBVOXELS_PER_VOXEL_DIM);
                hit_threshold = min_step_size;
            }
        } else {
            dist = res.distance;
            total_dist += max(dist, min_step_size);
            pos = origin + total_dist * ray;
            min_dist = min(min_dist, dist);
        }

        debug = res.debug_querypoint;
    }

    var step_color = f32(steps) / f32(max_steps);
    var outside_color = f32(cascade) / f32(cascades_info.count);
    // if cascade == cascades_info.count {
    //     outside_color = 1.0;
    //     dist = max_distance;
    // }
    var hit_color = 0.0;
    if dist <= hit_threshold {
        hit_color = 0.2;
    }
    // hit_color = clamp((min_dist - hit_threshold) / (hit_threshold * 0.25), 0.0, 1.0);
    let dist_color = sqrt(dot(pos - origin,pos - origin)) / (cascades_info.cascades[min(cascade, cascades_info.count - 1u)].tile_size * 8.0 / 1.75);

    // return vec4<f32>(hit_color * 0.1, step_color, hit_color * 0.2, 1.0);
    return vec4<f32>(outside_color, step_color * dist_color, 0.2, 1.0);
    // return vec4<f32>(1.0 - distance_color, step_color, hit_color, 1.0);
    // return vec4<f32>(debug, 1.0);
}