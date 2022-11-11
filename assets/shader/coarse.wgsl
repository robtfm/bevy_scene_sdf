// scan triangles and allocate them to tiles for subsequent processing in fine.wgsl
// also multiply meshes by transforms and write out the final vertex positions. this should be done up front instead.

// todo
// - pretransform triangles
//   - no need to write processed
// split and duplicate with subsets of tiles regions?
// parallel prefix sum
// parallel init

#import sdf::consts as consts
#import sdf::bind as bind
#import sdf::types as types

@group(1) @binding(0)
var<storage, read_write> dispatch_indirect: vec3<u32>;

var<workgroup> tile_counts: array<atomic<u32>, consts::TILE_COUNT>;
var<workgroup> tile_write_index: array<atomic<u32>, consts::TILE_COUNT>;

struct TileBbPair {
    min: vec3<u32>,
    max: vec3<u32>,
}

fn tri_bounds(tri_id: u32) -> TileBbPair {
    let epsilon = 1e-5 * bind::cascade_info.tile_size;

    let min_pos_cascade = min(bind::coarse_processed_tris[tri_id].vertex_positions[0], min(bind::coarse_processed_tris[tri_id].vertex_positions[1], bind::coarse_processed_tris[tri_id].vertex_positions[2]));
    let max_pos_cascade = max(bind::coarse_processed_tris[tri_id].vertex_positions[0], max(bind::coarse_processed_tris[tri_id].vertex_positions[1], bind::coarse_processed_tris[tri_id].vertex_positions[2])) + epsilon;

    var res: TileBbPair;
    let min_tile = vec3<i32>(floor(min_pos_cascade.xyz / bind::cascade_info.tile_size));
    let max_tile = vec3<i32>( ceil(max_pos_cascade.xyz / bind::cascade_info.tile_size));
    res.min = vec3<u32>(clamp(min_tile, vec3<i32>(0), vec3<i32>(i32(consts::TILE_DIM_COUNT))));
    res.max = vec3<u32>(clamp(max_tile, vec3<i32>(0), vec3<i32>(i32(consts::TILE_DIM_COUNT))));

    return res;
}

fn process_tri(mesh_index: u32, local_vertex_id: u32, output_tri_id: u32) {
    let origin = vec4<f32>(bind::cascade_info.origin) * bind::cascade_info.tile_size;
    let transform = bind::transforms.t[bind::mesh_headers.index_offset_count[mesh_index].x];
    var processed: types::ProcessedTri;

    processed.vertex_positions[0] = transform * bind::vertex_positions[bind::mesh_headers.index_offset_count[mesh_index].y + local_vertex_id] - origin;
    processed.vertex_positions[1] = transform * bind::vertex_positions[bind::mesh_headers.index_offset_count[mesh_index].y + local_vertex_id + 1u] - origin;
    processed.vertex_positions[2] = transform * bind::vertex_positions[bind::mesh_headers.index_offset_count[mesh_index].y + local_vertex_id + 2u] - origin;

    bind::coarse_processed_tris[output_tri_id] = processed;
}

fn tile_index(x: u32, y: u32, z: u32) -> u32 {
    return (((z * consts::TILE_DIM_COUNT) + y) * consts::TILE_DIM_COUNT) + x;
}

// modify the array to contain running sums
// todo: make this parallel, we are on a gpu ffs
fn prefix_sum(tid: u32) -> u32 {
    var total_tile_writes = 0u;

    if tid == 0u {
        var prev = 0u;
        var next = 0u;
        var index = 0u;
        for (var tile=0u; tile<consts::TILE_COUNT; tile++) {
            next = min(prev + atomicAdd(&tile_counts[tile], prev), consts::MAX_ID_COUNT);

            while prev < next {
                prev = min(next, prev + consts::MAX_IDS_PER_TILE);
                bind::coarse_tile_counts[index].x = tile;
                bind::coarse_tile_counts[index].y = prev;
                index += 1u;
                if index == consts::MAX_TILES {
                    return index;
                } 
            }
        }

        bind::coarse_tile_counts[index].x = 0xFFFFFFFFu;
        return index;
    }

    return 0u;
}

let COARSE_THREADS = 1024u; // must match workgroup_size.x
@compute @workgroup_size(1024, 1, 1)
fn coarse_raster(@builtin(local_invocation_index) invocation_id: u32) {
    // loop to count tris per tile
    var tri_id = invocation_id;
    var local_vertex_id = tri_id * 3u;
    var mesh_index = 0u;

    // init counts
    if invocation_id == 0u {
        for (var i=0u; i<consts::TILE_COUNT; i++) {
            atomicStore(&tile_counts[i], 0u);
            atomicStore(&tile_write_index[i], 0u);
        }
    }

    workgroupBarrier();

    while tri_id < bind::mesh_headers.tri_count {
        while local_vertex_id >= bind::mesh_headers.index_offset_count[mesh_index].z {
            local_vertex_id -= bind::mesh_headers.index_offset_count[mesh_index].z;
            mesh_index += 1u;
        }

        process_tri(mesh_index, local_vertex_id, tri_id);
        let bbs = tri_bounds(tri_id);

        for (var x=bbs.min.x; x<bbs.max.x; x += 1u) {
            for (var y=bbs.min.y; y<bbs.max.y; y += 1u) {
                for (var z=bbs.min.z; z<bbs.max.z; z += 1u) {
                    atomicAdd(&tile_counts[tile_index(x, y, z)], 1u);
                }        
            }
        }

        tri_id += COARSE_THREADS;
        local_vertex_id += COARSE_THREADS * 3u;
    }

    workgroupBarrier();

    if invocation_id == 0u {
        let invocations = prefix_sum(invocation_id);
        dispatch_indirect = vec3<u32>(invocations, 1u, 1u);
    }

    workgroupBarrier();

    // write tri indexes out
    tri_id = invocation_id;
    
    while tri_id < bind::mesh_headers.tri_count {
        let bbs = tri_bounds(tri_id);

        for (var x=bbs.min.x; x<bbs.max.x; x += 1u) {
            for (var y=bbs.min.y; y<bbs.max.y; y += 1u) {
                for (var z=bbs.min.z; z<bbs.max.z; z += 1u) {
                    let tile = tile_index(x,y,z);
                    var start_offset = 0u;
                    if tile > 0u {
                        start_offset = tile_counts[tile - 1u];
                    }
                    let write_index = atomicAdd(&tile_write_index[tile], 1u);
                    if start_offset + write_index < consts::MAX_ID_COUNT {
                        bind::coarse_ids[start_offset + write_index] = tri_id;
                    }
                }        
            }
        }

        tri_id += COARSE_THREADS;
    }
}