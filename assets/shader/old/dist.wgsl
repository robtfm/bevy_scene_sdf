#define_import_path sdf::dist

#import sdf::bind as bind

struct TriAccel {
    v: array<vec3<f32>, 3>,
    e: array<vec3<f32>, 3>,
    inv_area: f32,
    n_d: vec4<f32>,
}

fn build_tri(id: u32) -> TriAccel {
    let tri = bind::coarse_processed_tris[id];

    var accel: TriAccel;

    // edges / normal / plane intercept / area
    accel.v[0] = tri.vertex_positions[0].xyz;
    accel.v[1] = tri.vertex_positions[1].xyz;
    accel.v[2] = tri.vertex_positions[2].xyz;
    accel.e[0] = accel.v[1] - accel.v[0];
    accel.e[1] = accel.v[2] - accel.v[1];
    accel.e[2] = accel.v[0] - accel.v[2];
    let cross_e0_e1 = cross(accel.e[0],accel.e[1]);
    let n = normalize(cross_e0_e1);
    accel.inv_area = 1.0 / dot(cross_e0_e1, n);
    accel.n_d = vec4<f32>(n, dot(-accel.v[0], n));

    return accel;
}

fn distance_squared(x: vec3<f32>, y: vec3<f32>) -> f32 {
    let v = y - x;
    return dot(v, v);
}

fn tri_distance(tri: ptr<function, TriAccel>, target_point: vec3<f32>, max_dist_sq: f32) -> f32 {
    var best_dist = max_dist_sq;
    for (var i = 0u; i<3u; i++) {
        // dist to vertices
        best_dist = min(best_dist, distance_squared(target_point, (*tri).v[i]));
        // dist to edges
        let edge_sq = dot((*tri).e[i], (*tri).e[i]); 
        let intercept = dot(target_point - (*tri).v[i], (*tri).e[i]) / edge_sq;
        if intercept > 0.001 && intercept < 0.999 {
            best_dist = min(best_dist, distance_squared(target_point, (*tri).v[i] + (*tri).e[i] * intercept));
        } 
    }
    // dist to face
    let dist_to_plane = dot((*tri).n_d, vec4<f32>(target_point, 1.0));
    let dist_sq = dist_to_plane * dist_to_plane;
    if dist_sq < best_dist { // don't think we need to check this, it'll always be best if we are in the tri plane
        let point_on_plane = target_point - dist_to_plane * (*tri).n_d.xyz;
        // barycoords
        let u = dot(cross((*tri).e[1], point_on_plane - (*tri).v[1]), (*tri).n_d.xyz) * (*tri).inv_area;
        let v = dot(cross((*tri).e[2], point_on_plane - (*tri).v[2]), (*tri).n_d.xyz) * (*tri).inv_area;
        let w = 1.0 - u - v;
        if u >= 0.0 && v >= 0.0 && w > 0.0 {
            best_dist = min(best_dist, dist_sq);
        }
    }

    return best_dist;
}

fn tri_infront(tri: TriAccel, target_point: vec3<f32>) -> f32 {
    if dot(tri.n_d.xyz, target_point - tri.v[0]) > 0.0 {
        return 1.0;
    } else {
        return -1.0;
    }
}

