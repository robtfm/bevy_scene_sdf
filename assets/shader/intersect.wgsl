#define_import_path sdf::intersect

#import sdf::bind as bind
#import sdf::consts as consts

struct TriAccel {
    v: array<vec3<f32>, 3>,
    e: array<vec3<f32>, 3>,
    min: vec3<f32>,
    max: vec3<f32>,
    n: vec3<f32>,
    plane_corners: array<vec3<f32>, 2>,
    h: f32,

    edge_axis: array<vec3<f32>, 3>,
    edge_base: array<vec3<f32>, 3>,
    edge_corners: array<vec3<f32>, 3>,
    edge_dist_sq: array<f32, 3>,
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
    accel.n = normalize(cross_e0_e1);
    accel.h = 0.5 * bind::cascade_info.tile_size / f32(consts::VOXELS_PER_TILE_DIM * consts::SUBVOXELS_PER_VOXEL_DIM);

    accel.min = min(min(accel.v[0], accel.v[1]), accel.v[2]);
    accel.max = max(max(accel.v[0], accel.v[1]), accel.v[2]);

    accel.plane_corners[0] = accel.v[0] + sign(accel.n) * accel.h;
    accel.plane_corners[1] = accel.v[0] - sign(accel.n) * accel.h;

    for (var i=0u; i<3u; i++) {
        let edge = i;
        let vert = (i + 2u) % 3u;

        // project v0->v2 (== -e[2]) onto v0->v1 (== e[0])
        let intercept = -dot(accel.e[vert], accel.e[edge]);
        // find base of axis on the v0->v1 edge
        let base = accel.v[edge] + accel.e[edge] * intercept / dot(accel.e[edge], accel.e[edge]);
        // top of axis is just the vertex
        let tip = accel.v[vert];
        // get the axis (don't normalize)
        accel.edge_axis[i] = tip - base;

        // signs to work out near and far corners
        let axis_dir = sign(accel.edge_axis[i]);
        // box corners
        accel.edge_corners[i] = accel.h * axis_dir;
        accel.edge_base[i] = base;

        // far dist for test
        accel.edge_dist_sq[i] = dot(accel.edge_axis[i], accel.edge_axis[i]);
    }

    return accel;
}

fn voxel_bb_test(target_point: vec3<f32>, tri: ptr<function, TriAccel>) -> bool {
    return all((*tri).min < target_point + (*tri).h && (*tri).max >= target_point - (*tri).h);
}

fn voxel_plane_test(target_point: vec3<f32>, tri: ptr<function, TriAccel>) -> bool {
    let r1 = dot((*tri).n, target_point - (*tri).plane_corners[0]);
    let r2 = dot((*tri).n, target_point - (*tri).plane_corners[1]);

    return sign(r1) != sign(r2);
}

fn voxel_edge_test(target_point: vec3<f32>, tri: ptr<function, TriAccel>) -> bool {
    for (var i=0u; i<3u; i++) {
        let axis = (*tri).edge_axis[i];

        // box corners
        let far_corner = target_point + (*tri).edge_corners[i] - (*tri).edge_base[i];
        let near_corner = target_point - (*tri).edge_corners[i] - (*tri).edge_base[i];

        // project corners onto axis
        let far = dot(far_corner, (*tri).edge_axis[i]);
        let near = dot(near_corner, (*tri).edge_axis[i]);

        // test
        if (max(near, far) < 0.0) || (min(near, far) > (*tri).edge_dist_sq[i]) {
            return false;
        }
    }

    return true;
}

fn voxel_test(target_point: vec3<f32>, tri: ptr<function, TriAccel>) -> bool {
    return
        true
        // && voxel_bb_test(target_point, tri) 
        // && voxel_plane_test(target_point, tri) 
        && voxel_edge_test(target_point, tri)
        ;
}

// todo - base this on primary axis of the triangle for better efficiency
// todo - filter x/y as well
fn subvoxel_z_range(tile_min: vec3<f32>, subvoxel_x: u32, subvoxel_y: u32, subv: f32, tri: ptr<function, TriAccel>) -> vec2<u32> {
    let target_point = tile_min + (vec3<f32>(vec3<u32>(subvoxel_x, subvoxel_y, 0u)) + vec3<f32>(0.5)) * subv;

    if abs((*tri).n.z) < 1e-20 {
        // runs parallel, check one point
        if voxel_plane_test(target_point, tri) {
            return vec2<u32>(0u, consts::SUBVOXELS_PER_VOXEL_DIM * consts::VOXELS_PER_DIM);
        } else {
            return vec2<u32>(1u, 0u);
        }
    }

    // get plane test results for first subvoxel in the z-row
    let r1 = dot((*tri).n, target_point - (*tri).plane_corners[0]);
    let r2 = dot((*tri).n, target_point - (*tri).plane_corners[1]);

    // use the delta per subvoxel to find the subvoxel index (as f32) where they touch the tri plane
    let near_zero = -r2 / ((*tri).n.z * subv);
    let far_zero = -r1 / ((*tri).n.z * subv);

    // clamp our test range to those where the signs will differ for the 2 corners
    let low_point = u32(clamp(i32(ceil(min(far_zero, near_zero))), 0, i32(consts::SUBVOXELS_PER_VOXEL_DIM * consts::VOXELS_PER_DIM)));
    let hi_point = u32(clamp(i32(ceil(max(far_zero, near_zero))), 0, i32(consts::SUBVOXELS_PER_VOXEL_DIM * consts::VOXELS_PER_DIM)));

    return vec2<u32>(low_point, hi_point);
}