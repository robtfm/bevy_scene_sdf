
// max total triangles we can handle in one go (scene can be larger as long as it is not all within a single cascade update)
// we preallocate xxx bytes per triangle of gpu storage buffer space for intermediate data
// this should be removed - we can preprocess triangles on mesh load on cpu then coarse.wgsl doesn't need to process them 
pub const MAX_TRI_COUNT: usize = 1 << 23;

// tile count in each dimension
// the constraint for this is workgroup shared memory - we make heavy use of LDS atomics in coarse.wgsl
pub const TILE_DIM_COUNT: usize = 16;

// number of voxels per tile dimension (so total voxels = tile_dim^3 * vox_per_tile_dim^3)
// this is the resolution of the final SDF
// we need voxels per dim (== voxels_per_tile_dim * tile_dim_count) to be <= 128 to fit our jump offsets into an rgba8sint texture
// if you make this larger you'll need to update the nearest_jfa texture type as well
pub const VOXELS_PER_TILE_DIM: usize = 8;

// subvoxels per voxel
// must be 1-4
// if 1 or 2, the rg32uint jfa seed texture could be r8uint instead of rg32uint
// in theory > 4 would work but texture requirements get larger and the code to write them in fine / fine_blend.wgsl would need updating
// using multiple subvoxels gives a smoother final SDF and allows more granular occlusion
pub const SUBVOXELS_PER_VOXEL_DIM: usize = 4;

// total tiles / buckets
pub const TILE_COUNT: usize = TILE_DIM_COUNT * TILE_DIM_COUNT * TILE_DIM_COUNT;

// max tri/tile intersections. it will screw up visuals if we exceed this, and probably be slow as well. reduce scene complexity / sdf volume
// we need 1 slot for each tile that each triangle touches
// this is quite memory hungry, can def be reduced for lower complexity scenes
pub const MAX_ID_COUNT: usize = 8192 * TILE_COUNT;

// when bucketing in coarse.wgsl to create data to feed to the fine.wgsl stage, when we have more than MAX_IDS_PER_TILE we will create a new tile with the 
// same location that gets blended into a final composite result in fine_blend.wgsl
pub const MAX_IDS_PER_TILE: usize = 128 * 4;

// max number of virtual tiles (tiles output by coarse.wgsl)
pub const MAX_TILES: usize = TILE_COUNT * 16;

pub const SDF_CONSTS_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 9556909940226147828);


mod debug;

pub use debug::{DebugSdfPlugin, DebugSdf};
use wgpu::{util::DispatchIndirect, BufferDescriptor, BufferUsages, QuerySet, ComputePass};

use std::{borrow::Cow, num::NonZeroU64, sync::mpsc::{Receiver, Sender}};

use bevy::{
    core::FrameCount,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::VertexAttributeValues,
        primitives::{Aabb, Frustum, Plane},
        render_asset::ExtractedAssets,
        render_graph::{Node, RenderGraph},
        render_resource::{
            BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
            BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, 
            BufferBindingType, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, DynamicUniformBuffer, Extent3d, PipelineCache, ShaderStages,
            ShaderType, StorageBuffer, StorageTextureAccess, Texture, TextureDescriptor,
            TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, Buffer,
        },
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp, RenderStage,
    },
    utils::{HashMap, HashSet}, reflect::TypeUuid, math::Vec3A,
};

pub struct SceneSdfPlugin;

#[derive(Clone, Debug)]
pub struct CascadeUpdateSchedule {
    pub frequency: u32,
    pub offset: u32,
}

#[derive(Clone, Debug)]
pub struct CascadeSettings {
    pub far_distance: f32,
    pub update_schedule: CascadeUpdateSchedule,
}

pub enum ExtractionFilter {
    Marked,
    Unmarked,
}

#[derive(Resource)]
pub struct SceneSdfSettings {
    pub filter: ExtractionFilter,
    pub cascades: Vec<CascadeSettings>,
}

impl Default for SceneSdfSettings {
    fn default() -> Self {
        Self {
            filter: ExtractionFilter::Unmarked,
            cascades: vec![
                CascadeSettings {
                    far_distance: 3.0,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 3,
                        offset: 0,
                    },
                },
                CascadeSettings {
                    far_distance: 5.0,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 3,
                        offset: 1,
                    },
                },
                CascadeSettings {
                    far_distance: 7.5,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 15,
                        offset: 2,
                    },
                },
                CascadeSettings {
                    far_distance: 11.25,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 15,
                        offset: 5,
                    },
                },
                CascadeSettings {
                    far_distance: 17.0,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 15,
                        offset: 8,
                    },
                },
                CascadeSettings {
                    far_distance: 30.0,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 15,
                        offset: 11,
                    },
                },
                CascadeSettings {
                    far_distance: 60.0,
                    update_schedule: CascadeUpdateSchedule {
                        frequency: 15,
                        offset: 14,
                    },
                },
            ],
        }
    }
}

impl Plugin for SceneSdfPlugin {
    fn build(&self, app: &mut App) {
        app
        .init_resource::<SceneSdfSettings>()
        .init_resource::<SdfState>()
        .init_resource::<SdfCentre>()
        .add_plugin(ExtractResourcePlugin::<SdfState>::default())
        .add_system_to_stage(CoreStage::PostUpdate, init_state_for_settings)
        .add_system_to_stage(CoreStage::PostUpdate, update_cascades.after(init_state_for_settings)) // should be after AABB update
        ;

        assert!(VOXELS_PER_DIM <= 128, "jfa_nearest requires local offset fits into signed byte");

        const VOXELS_PER_DIM: usize = TILE_DIM_COUNT * VOXELS_PER_TILE_DIM;
        const VOXELS_PER_TILE: usize = VOXELS_PER_TILE_DIM * VOXELS_PER_TILE_DIM * VOXELS_PER_TILE_DIM;
        const FINE_OUTPUT_SIZE: usize = VOXELS_PER_TILE * MAX_TILES * 2;
        let consts_str = format!(
            "
            #define_import_path sdf::consts

            let TILE_DIM_COUNT: u32 = {TILE_DIM_COUNT}u;
            let VOXELS_PER_TILE_DIM: u32 = {VOXELS_PER_TILE_DIM}u;
            let VOXELS_PER_DIM: u32 = {VOXELS_PER_DIM}u;
            let SUBVOXELS_PER_VOXEL_DIM: u32 = {SUBVOXELS_PER_VOXEL_DIM}u;

            let TILE_COUNT: u32 = {TILE_COUNT}u;
            let VOXELS_PER_TILE: u32 = {VOXELS_PER_TILE}u;
            let MAX_ID_COUNT: u32 = {MAX_ID_COUNT}u;
            let MAX_IDS_PER_TILE: u32 = {MAX_IDS_PER_TILE}u;
            let MAX_TILES: u32 = {MAX_TILES}u;
            let FINE_OUTPUT_SIZE: u32 = {FINE_OUTPUT_SIZE}u;
            "
        );

        let mut shader_assets = app.world.resource_mut::<Assets<Shader>>();
        shader_assets.set_untracked(SDF_CONSTS_SHADER_HANDLE, Shader::from_wgsl_with_path(consts_str, std::file!()));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SceneSdfPipeline>()
            .init_resource::<SdfWorkingData>()
            .init_resource::<SdfOutputData>()
            .init_non_send_resource::<TimeStampQueue>()
            .add_system_to_stage(RenderStage::Extract, extract_sdf_meshes)
            .add_system_to_stage(RenderStage::Prepare, prepare_sdf_meshes)
            .add_system_to_stage(RenderStage::Queue, queue_sdf_data);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("scene_sdf_node", SceneSdfNode);
        render_graph
            .add_node_edge(
                "scene_sdf_node",
                bevy::render::main_graph::node::CAMERA_DRIVER,
            )
            .unwrap();
    }
}

impl CascadeSettings {
    fn tile_size(&self) -> f32 {
        self.far_distance / TILE_DIM_COUNT as f32 * 2.0
    }
}

#[derive(Clone, Debug)]
struct CascadeState {
    settings: CascadeSettings,
    last_origin: Option<IVec3>,
    redraw: IVec3,
}

#[derive(Default, Resource, ExtractResource, Clone)]
struct SdfState {
    cascades: Vec<CascadeState>,
    visible_entities: HashSet<Entity>,
}

#[derive(Resource, Default)]
pub struct SdfCentre(pub Vec3);

trait AabbIntersect {
    fn intersect_min(&self, with: Vec3) -> Aabb;
    fn intersect_max(&self, with: Vec3) -> Aabb;
}

impl AabbIntersect for Aabb {
    fn intersect_min(&self, with: Vec3) -> Aabb {
        Aabb::from_min_max(Vec3::from(self.min()).max(with), self.max().into())
    }
    fn intersect_max(&self, with: Vec3) -> Aabb {
        Aabb::from_min_max(self.min().into(), Vec3::from(self.max()).min(with))
    }
}

fn init_state_for_settings(settings: Res<SceneSdfSettings>, mut state: ResMut<SdfState>) {
    if settings.is_changed() {
        *state = SdfState {
            cascades: settings
                .cascades
                .iter()
                .map(|settings| CascadeState {
                    settings: settings.clone(),
                    last_origin: None,
                    redraw: IVec3::ZERO,
                })
                .collect(),
            visible_entities: HashSet::default(),
        };
    }
}

#[derive(Component)]
struct StaticObb(Frustum);

fn update_cascades(
    mut commands: Commands,
    settings: Res<SceneSdfSettings>,
    centre: Res<SdfCentre>,
    mut state: ResMut<SdfState>,
    frame: Res<FrameCount>,
    aabbs: Query<(Entity, &Aabb, &GlobalTransform, Option<&StaticObb>, Changed<GlobalTransform>, Changed<Aabb>)>,
) {
    let mut clip_rects = Vec::new();
    let mut updated = false;
    // clip_rects.push(Aabb::from_min_max(Vec3::splat(f32::MIN), Vec3::splat(f32::MAX)));

    for (_i, (cascade, state)) in settings.cascades.iter().zip(&mut state.cascades).enumerate() {
        if frame.0 % cascade.update_schedule.frequency == cascade.update_schedule.offset && !updated {
            let tile_size = cascade.tile_size();
            let origin = ((centre.0 / tile_size) - (TILE_DIM_COUNT as f32 / 2.0))
                .floor()
                .as_ivec3();

            state.redraw = match state.last_origin {
                Some(last_origin) => (origin - last_origin).min(IVec3::splat(TILE_DIM_COUNT as i32)).max(IVec3::splat(TILE_DIM_COUNT as i32 * -1)),
                None => IVec3::new(TILE_DIM_COUNT as i32, 0, 0),

            };

            if state.redraw == IVec3::ZERO {
                continue;
            }

            updated = true;

            match state.last_origin.as_mut() {
                Some(last_origin) => {
                    // update only the biggest axis
                    let abs_redraw = state.redraw.abs();

                    if abs_redraw.max_element() == abs_redraw.x {
                        state.redraw *= IVec3::X;
                        last_origin.x = origin.x;
                    } else if abs_redraw.max_element() == abs_redraw.y {
                        state.redraw *= IVec3::Y;
                        last_origin.y = origin.y;
                    } else {
                        state.redraw *= IVec3::Z;
                        last_origin.z = origin.z;
                    }
                },
                _ => state.last_origin = Some(origin),
            }

            let border_min = origin.as_vec3() * tile_size;
            let border_max = (origin + TILE_DIM_COUNT as i32).as_vec3() * tile_size;

            let border_aabb = Aabb::from_min_max(border_min, border_max);

            if state.redraw.x > 0 {
                let calculated_x_min = border_max.x - tile_size * state.redraw.x as f32;
                clip_rects.push(border_aabb.intersect_min(Vec3::new(
                    calculated_x_min,
                    f32::MIN,
                    f32::MIN,
                )));
            } else if state.redraw.x < 0 {
                let calculated_x_max = border_min.x - tile_size * state.redraw.x as f32;
                clip_rects.push(border_aabb.intersect_max(Vec3::new(
                    calculated_x_max,
                    f32::MAX,
                    f32::MAX,
                )));
            }

            if state.redraw.y > 0 {
                let calculated_y_min = border_max.y - tile_size * state.redraw.y as f32;
                clip_rects.push(border_aabb.intersect_min(Vec3::new(
                    f32::MIN,
                    calculated_y_min,
                    f32::MIN,
                )));
            } else if state.redraw.y < 0 {
                let calculated_y_max = border_min.y - tile_size * state.redraw.y as f32;
                clip_rects.push(border_aabb.intersect_max(Vec3::new(
                    f32::MAX,
                    calculated_y_max,
                    f32::MAX,
                )))
            }

            if state.redraw.z > 0 {
                let calculated_z_min = border_max.z - tile_size * state.redraw.z as f32;
                clip_rects.push(border_aabb.intersect_min(Vec3::new(
                    f32::MIN,
                    f32::MIN,
                    calculated_z_min,
                )))
            } else if state.redraw.z < 0 {
                let calculated_z_max = border_min.z - tile_size * state.redraw.z as f32;
                clip_rects.push(border_aabb.intersect_max(Vec3::new(
                    f32::MAX,
                    f32::MAX,
                    calculated_z_max,
                )))
            }

            // println!("[{}] : origin: {}, redraw: {}", _i, origin, state.redraw);
            // println!("min/max: {:?}", (border_min, border_max));
            // println!("clip rects: {:?}", clip_rects);
        } else {
            state.redraw = IVec3::ZERO;
        }
    }

    state.visible_entities.clear();
    let mut _count = 0;
    let mut _total = 0;
    if !clip_rects.is_empty() {
        for (ent, aabb, g_trans, maybe_obb, changed_trans, changed_aabb) in &aabbs {
            let obb = match (maybe_obb, changed_trans || changed_aabb) {
                (Some(obb), false) => obb.0.clone(),
                _ => {
                    let planes = [-Vec3A::X, Vec3A::X, -Vec3A::Y, Vec3A::Y, -Vec3A::Z, Vec3A::Z]
                        .into_iter()
                        .map(|v| {
                            let matrix = g_trans.compute_matrix();
                            let point_on_plane = aabb.center + v * aabb.half_extents;
                            let transformed_point = (matrix * point_on_plane.extend(1.0)).truncate();
                            let transformed_normal = (matrix * -v.extend(0.0)).truncate();
                            Plane::new(transformed_normal.extend(-transformed_point.dot(transformed_normal)))
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();
                    let frustum = Frustum { planes };

                    commands.entity(ent).insert(StaticObb(frustum.clone()));
                    frustum
                },
            };

            let obb_intersects_aabb = |obb: &Frustum, aabb: &Aabb| -> bool {
                let aabb_center_world = aabb.center.extend(1.0);
                let axes = [Vec3A::X, Vec3A::Y, Vec3A::Z];
        
                for plane in &obb.planes {
                    let p_normal = Vec3A::from(plane.normal_d());
                    let relative_radius = aabb.relative_radius(&p_normal, &axes);
                    if plane.normal_d().dot(aabb_center_world) + relative_radius < 0.0 {
                        return false;
                    }
                }
                true        
            };

            if clip_rects.iter().any(|rect| obb_intersects_aabb(&obb, rect)) {
                state.visible_entities.insert(ent);
                _count += 1;
            }
            _total += 1;
        }
    }

    // if _count > 0 {
    //     println!("inc {}/{}", _count, _total);
    // }
}

// render

#[derive(Component)]
struct SdfSceneData {
    transform: Mat4,
    handle: Handle<Mesh>,
}

fn extract_sdf_meshes(
    mut commands: Commands,
    state: Extract<Res<SdfState>>,
    meshes_query: Extract<Query<(&GlobalTransform, &Handle<Mesh>)>>,
) {
    let mut sdf_mesh_cmds = Vec::new();
    for ent in &state.visible_entities {
        if let Ok((transform, handle)) = meshes_query.get(*ent) {
            let transform = transform.compute_matrix();

            sdf_mesh_cmds.push((
                *ent,
                (SdfSceneData {
                    transform,
                    handle: handle.clone_weak(),
                }),
            ));
        }
    }

    commands.insert_or_spawn_batch(sdf_mesh_cmds);
}

#[derive(Resource)]
struct SdfWorkingData {
    // timestamps
    time_query: Option<(QuerySet, Vec<Buffer>, usize)>,
    record_stats: bool,
    // input buffers
    cascade_header_buffer: DynamicUniformBuffer<SdfCascadeInfo>,
    mesh_header_buffer: StorageBuffer<SdfMeshHeader>,
    transforms_buffer: StorageBuffer<SdfTransforms>,
    tris_buffer: StorageBuffer<Vec<Vec4>>,
    // intermediate buffers
    coarse_tri_buffer: StorageBuffer<Vec<ProcessedTriData>>, // max tri count
    coarse_counts_per_tile: StorageBuffer<Vec<UVec2>>,         // tile count
    coarse_tile_ids: StorageBuffer<Vec<u32>>,                // max tile id count
    dispatch_fine_tiles: Option<Buffer>,
    fine_tile_output: Option<Buffer>,
    jfa_buffers: Option<(Texture, TextureView, Texture, TextureView, usize)>,
    jfa_params: DynamicUniformBuffer<JfaParams>,
    // params
    mesh_offsets: HashMap<Handle<Mesh>, (usize, usize)>,
    jfa_param_offsets: Vec<u32>,
    // bindgroups
    jfa_bindgroup: Option<BindGroup>,
    working_bindgroup: Option<BindGroup>,
    dispatch_bindgroup: Option<BindGroup>,
    output_bindgroups: Vec<BindGroup>,
}

impl Default for SdfWorkingData {
    fn default() -> Self {
        Self {
            time_query: Default::default(),
            record_stats: false,
            cascade_header_buffer: Default::default(),
            mesh_header_buffer: Default::default(),
            transforms_buffer: Default::default(),
            tris_buffer: Default::default(),
            coarse_tri_buffer: StorageBuffer::from(Vec::from_iter(
                std::iter::repeat(Default::default()).take(MAX_TRI_COUNT),
            )),
            coarse_counts_per_tile: StorageBuffer::from(Vec::from_iter(
                std::iter::repeat(Default::default()).take(MAX_TILES),
            )),
            coarse_tile_ids: StorageBuffer::from(Vec::from_iter(
                std::iter::repeat(Default::default()).take(MAX_ID_COUNT),
            )),
            mesh_offsets: Default::default(),
            working_bindgroup: Default::default(),
            dispatch_bindgroup: Default::default(),
            fine_tile_output: Default::default(),
            jfa_buffers: Default::default(),
            jfa_params: Default::default(),
            jfa_param_offsets: Default::default(),
            jfa_bindgroup: Default::default(),
            output_bindgroups: Default::default(),
            dispatch_fine_tiles: None,
        }
    }
}

#[derive(Resource, Default)]
struct SdfOutputData {
    sdf_header_buffer: StorageBuffer<SdfCascadeInfos>,
    output_buffer: Option<(Texture, TextureView, usize)>,
}

#[derive(ShaderType, Default, Clone)]
struct ProcessedTriData {
    vertex_positions: [Vec4; 3],
}

fn prepare_sdf_meshes(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted_assets: Res<ExtractedAssets<Mesh>>,
    mut pos_data: ResMut<SdfWorkingData>,
    mut init: Local<bool>,
    mut state: ResMut<SdfState>,
) {
    if !*init {
        *init = true;
        pos_data.cascade_header_buffer.write_buffer(&device, &queue);
        pos_data.mesh_header_buffer.write_buffer(&device, &queue);
        pos_data.transforms_buffer.write_buffer(&device, &queue);
        pos_data.tris_buffer.write_buffer(&device, &queue);
        pos_data.coarse_tri_buffer.write_buffer(&device, &queue);
        pos_data
            .coarse_counts_per_tile
            .write_buffer(&device, &queue);
        pos_data.coarse_tile_ids.write_buffer(&device, &queue);
    }

    if extracted_assets.removed.is_empty() && extracted_assets.extracted.is_empty() {
        return;
    }

    if !extracted_assets.removed.is_empty() {
        let prev_buffer = std::mem::take(&mut pos_data.tris_buffer);
        let prev_data = prev_buffer.get();

        let mut new_vec = Vec::new();

        let mut running_offset = 0;
        let offsets = &mut pos_data.mesh_offsets;
        offsets.retain(|handle, (offset, count)| {
            if extracted_assets.removed.contains(handle) {
                false
            } else {
                *offset = running_offset;
                running_offset += *count;
                new_vec.extend_from_slice(&prev_data[*offset..*offset + *count]);
                true
            }
        });

        pos_data.tris_buffer.set(new_vec);
    }

    let mut offsets = std::mem::take(&mut pos_data.mesh_offsets);
    let data = pos_data.tris_buffer.get_mut();
    for (handle, mesh) in &extracted_assets.extracted {
        if let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        {
            let offset = data.len();
            let count = match mesh.indices() {
                Some(indices) => {
                    data.extend(
                        indices
                            .iter()
                            .map(|ix| Vec3::from_array(positions[ix]).extend(1.0)),
                    );
                    indices.len()
                }
                None => {
                    data.extend(
                        positions
                            .iter()
                            .map(|pos| Vec3::from_array(*pos).extend(1.0)),
                    );
                    positions.len()
                }
            };

            offsets.insert(handle.clone_weak(), (offset, count));
        }
    }
    pos_data.mesh_offsets = offsets;

    pos_data.tris_buffer.write_buffer(&device, &queue);

    for cascade in &mut state.cascades {
        cascade.last_origin = None;
    }
}

#[derive(Resource)]
struct SceneSdfPipeline {
    #[allow(dead_code)]
    shaders: HashSet<Handle<Shader>>,
    working_layout: BindGroupLayout,
    dispatch_layout: BindGroupLayout,
    coarse_pipeline: CachedComputePipelineId,
    fine_pipeline: CachedComputePipelineId,
    fine_blend_pipeline: CachedComputePipelineId,
    jfa_layout: BindGroupLayout,
    jfa_pipeline: CachedComputePipelineId,
    stitch_pipeline: CachedComputePipelineId,
    output_layout: BindGroupLayout,
    output_pipeline: CachedComputePipelineId,
}

impl FromWorld for SceneSdfPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();

        let entries = [
            // meshes
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(SdfMeshHeader::min_size()),
                },
                count: None,
            },
            // transforms
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(SdfTransforms::min_size()),
                },
                count: None,
            },
            // positions
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(<[f32; 4]>::min_size()),
                },
                count: None,
            },
            // cascade infos
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(SdfCascadeInfo::min_size()),
                },
                count: None,
            },
            // tri data
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        ProcessedTriData::min_size()
                            .saturating_mul(NonZeroU64::try_from(MAX_TRI_COUNT as u64).unwrap()),
                    ),
                },
                count: None,
            },
            // id count per tile
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        UVec2::min_size()
                            .saturating_mul(NonZeroU64::try_from(MAX_TILES as u64).unwrap()),
                    ),
                },
                count: None,
            },
            // tri ids per tile
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        u32::min_size()
                            .saturating_mul(NonZeroU64::try_from(MAX_ID_COUNT as u64).unwrap()),
                    ),
                },
                count: None,
            },
            // jfa seed tex
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rg32Uint,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            },
            // jfa dist tex
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba8Sint,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            },
            // fine output tex 
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new((MAX_TILES * std::mem::size_of::<u32>() * 2 * VOXELS_PER_TILE_DIM * VOXELS_PER_TILE_DIM * VOXELS_PER_TILE_DIM) as u64),
                },
                count: None,
            },
        ];

        let working_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sdf coarse layout"),
            entries: &entries,
        });

        let dispatch_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { 
            label: Some("dispatch layout") ,
            entries: &[
                // dispatch workgroups
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: NonZeroU64::new(std::mem::size_of::<DispatchIndirect>() as u64) },
                    count: None,
                },            
            ] }
        );

        let jfa_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("jfa params layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(JfaParams::min_size()),
                },
                count: None,
            }],
        });

        let output_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sdf output layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            }],
        });

        let asset_server = world.resource::<AssetServer>();
        let mut shaders = HashSet::new();
        // shaders.insert(asset_server.load("shader/consts.wgsl"));
        shaders.insert(asset_server.load("shader/types.wgsl"));
        shaders.insert(asset_server.load("shader/bind.wgsl"));
        shaders.insert(asset_server.load("shader/addressing.wgsl"));
        shaders.insert(asset_server.load("shader/intersect.wgsl"));

        let coarse_shader = world.resource::<AssetServer>().load("shader/coarse.wgsl");
        let fine_shader = world.resource::<AssetServer>().load("shader/fine.wgsl");
        let fine_blend_shader = world.resource::<AssetServer>().load("shader/fine_blend.wgsl");
        let jfa_shader = world.resource::<AssetServer>().load("shader/jfa simple.wgsl");
        let stitch_shader = world.resource::<AssetServer>().load("shader/jfa stitch.wgsl");
        let output_shader = world.resource::<AssetServer>().load("shader/output.wgsl");

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let coarse_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![working_layout.clone(), dispatch_layout.clone()]),
            shader: coarse_shader,
            shader_defs: vec![],
            entry_point: Cow::from("coarse_raster"),
        });

        let fine_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![working_layout.clone()]),
            shader: fine_shader,
            shader_defs: vec![],
            entry_point: Cow::from("fine_raster"),
        });

        let fine_blend_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![working_layout.clone()]),
            shader: fine_blend_shader,
            shader_defs: vec![],
            entry_point: Cow::from("fine_blend"),
        });

        let jfa_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![working_layout.clone(), jfa_layout.clone()]),
            shader: jfa_shader,
            shader_defs: vec![],
            entry_point: Cow::from("jfa"),
        });

        let stitch_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![working_layout.clone(), jfa_layout.clone()]),
            shader: stitch_shader,
            shader_defs: vec![],
            entry_point: Cow::from("stitch"),
        });

        let output_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![working_layout.clone(), output_layout.clone()]),
            shader: output_shader,
            shader_defs: vec![],
            entry_point: Cow::from("output"),
        });

        Self {
            shaders,
            working_layout,
            dispatch_layout,
            coarse_pipeline,
            fine_pipeline,
            fine_blend_pipeline,
            jfa_layout,
            jfa_pipeline,
            stitch_pipeline,
            output_layout,
            output_pipeline,
        }
    }
}

#[derive(ShaderType)]
struct JfaParams {
    jump_size: i32,
}

#[derive(Default)]
struct TimeStampQueue(
    Option<(Sender<usize>, Receiver<usize>)>,
);

#[repr(u8)]
enum StampIndex {
    START = 0,
    COARSE = 1,
    FINE = 2,
    FINEBLEND = 3,
    JFA = 4,
    STITCH = 5,
    OUTPUT = 6,
}

impl StampIndex {
    const LABELS: [&'static str; 7] = ["coarse", "fine", "fine blend", "jump", "stitch", "output", "total"];
}

fn queue_sdf_data(
    sdf_meshes: Query<&SdfSceneData>,
    state: Res<SdfState>,
    mut pos_data: ResMut<SdfWorkingData>,
    mut output_data: ResMut<SdfOutputData>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline: Res<SceneSdfPipeline>,
    mut channel: NonSendMut<TimeStampQueue>,
    mut avgs: Local<Vec<f32>>,
    mut maxs: Local<Vec<f32>>,
    mut count: Local<usize>,
    frame: Res<FrameCount>,
) {
    let (index_offset_counts, transforms): (Vec<UVec4>, Vec<Mat4>) = sdf_meshes
        .iter()
        .enumerate()
        .map(|(ix, scene_data)| {
            let offset_and_count = pos_data.mesh_offsets[&scene_data.handle];
            (
                UVec4::new(
                    ix as u32,
                    offset_and_count.0 as u32,
                    offset_and_count.1 as u32,
                    0,
                ),
                scene_data.transform,
            )
        })
        .unzip();
    let vertex_count: u32 = index_offset_counts.iter().map(|v| v.z).sum();
    // println!("vertex count: {}", vertex_count);
    // println!("mesh count: {}", transforms.len());
    // println!("tri count: {}", vertex_count / 3);

    pos_data.cascade_header_buffer.clear();
    for (i, cascade) in state.cascades.iter().enumerate() {
        if cascade.redraw != IVec3::ZERO {
            pos_data.cascade_header_buffer.push(SdfCascadeInfo {
                tile_size: cascade.settings.tile_size(),
                origin: cascade.last_origin.unwrap_or_default().extend(0),
                redraw: cascade.redraw.extend(0),
                index: i as u32,
            });
        }
    }

    if frame.0 == 10000 {
        pos_data.time_query = None;
    }

    if pos_data.cascade_header_buffer.is_empty() {
        pos_data.working_bindgroup = None;
        return;
    }

    pos_data.cascade_header_buffer.write_buffer(&device, &queue);

    // stats
    let mut record_stats = false;
    match pos_data.time_query.as_mut() {
        None => {
            let (sender, receiver) = std::sync::mpsc::channel();
            channel.0 = Some((sender, receiver));

            let q = device.wgpu_device().create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: 10,
            });
            let bs = (0..10).map(|_| device.create_buffer(&BufferDescriptor {
                label: Some("timestamps"),
                size: 10 * 40,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })).collect();
            pos_data.time_query = Some((q, bs, 0));

            *avgs = std::iter::repeat(0.0).take(StampIndex::OUTPUT as usize + 1).collect();
            *maxs = std::iter::repeat(0.0).take(StampIndex::OUTPUT as usize + 1).collect();
        },
        Some((_, bs, ix)) => {
            if state.cascades.iter().all(|c| c.redraw.abs().max_element() <= 1) {
                record_stats = true;
                *ix = (*ix + 1) % 10;
                let read = (*ix + 5) % 10;
                let slice = bs[read as usize].slice(..);
                let (sender, receiver) = (*channel).0.as_ref().unwrap();
                let sender = sender.clone();
                device.map_buffer(&slice, wgpu::MapMode::Read, move |res| {
                    match res {
                        Ok(_) => {
                            sender.send(read).unwrap();
                        }
                        Err(_) => println!(":("),
                    }
                });
    
                while let Ok(read) = receiver.try_recv()  {
                    let prd = queue.get_timestamp_period();
                    let range = bs[read].slice(..).get_mapped_range();
                    let mut times: Vec<_> = range.chunks(8).take(StampIndex::OUTPUT as usize + 1).map(|chunk| {
                        let mut bytes = [0u8; 8];
                        bytes.clone_from_slice(chunk); 
                        u64::from_le_bytes(bytes)
                    }).collect();
                    let mut prev = times.remove(0);
                    let total = (times.last().unwrap() - prev) as f32 * prd / 1000000.0;
                    let times: Vec<f32> = times.into_iter().map(|t| {
                        let res = (t - prev) as f32 * prd / 1000000.0;
                        prev = t;
                        res
                    }).chain(std::iter::once(total)).collect();
                    (*avgs).iter_mut().zip(&times).map(|(avg, time)| *avg += time).count();
                    (*maxs).iter_mut().zip(&times).map(|(avg, time)| *avg = time.max(*avg)).count();
                    *count += 1;
                    if (*count % 100) == 0 {
                        let avs: Vec<String> = (*avgs).iter().zip(StampIndex::LABELS.iter()).map(|(t, label)| format!("{} {:.2}ms", label, t/ *count as f32)).collect();
                        let avs_overall: Vec<String> = (*avgs).iter().zip(StampIndex::LABELS.iter()).map(|(t, label)| format!("{} {:.2}ms", label, t/ frame.0 as f32)).collect();
                        let mxs: Vec<String> = (*maxs).iter().zip(StampIndex::LABELS.iter()).map(|(t, label)| format!("{} {:.2}ms", label, t)).collect();
                        println!("avg active {:?}", avs);
                        println!("avg frame  {:?}", avs_overall);
                        println!("max        {:?}", mxs);
                    }
                    drop(range);
                    bs[read].unmap();
                }
            }
        }
    } 
    pos_data.record_stats = record_stats;

    // initialize jfa buffers
    if pos_data.jfa_buffers.as_ref().map(|jfa| jfa.4 ) != Some(state.cascades.len()) {
        let vox_per_dim = VOXELS_PER_TILE_DIM * TILE_DIM_COUNT;

        let seed_texture_desc = TextureDescriptor {
            label: Some("sdf output texture"),
            mip_level_count: 1,
            sample_count: 1,
            dimension: bevy::render::render_resource::TextureDimension::D3,
            format: TextureFormat::Rg32Uint,
            size: Extent3d {
                width: vox_per_dim as u32 * state.cascades.len() as u32,
                height: vox_per_dim as u32,
                depth_or_array_layers: vox_per_dim as u32,
            },
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        };

        let seed_texture = device.create_texture(&seed_texture_desc);
        let seed_view = seed_texture.create_view(&TextureViewDescriptor::default());

        let nearest_texture_desc = TextureDescriptor {
            label: Some("sdf output texture"),
            mip_level_count: 1,
            sample_count: 1,
            dimension: bevy::render::render_resource::TextureDimension::D3,
            format: TextureFormat::Rgba8Sint,
            size: Extent3d {
                width: vox_per_dim as u32 * state.cascades.len() as u32,
                height: vox_per_dim as u32,
                depth_or_array_layers: vox_per_dim as u32,
            },
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        };

        let nearest_texture = device.create_texture(&nearest_texture_desc);
        let nearest_view = nearest_texture.create_view(&TextureViewDescriptor::default());

        pos_data.jfa_buffers = Some((
            seed_texture,
            seed_view,
            nearest_texture,
            nearest_view,
            state.cascades.len(),
        ));

        for sz in [64, 32, 16, 8, 4, 2, 1] {
            let offset = pos_data.jfa_params.push(JfaParams {
                jump_size: sz,
            });
            pos_data.jfa_param_offsets.push(offset);
        }
        pos_data.jfa_params.write_buffer(&device, &queue);
        pos_data.jfa_bindgroup = Some(device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.jfa_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: pos_data.jfa_params.binding().unwrap(),
            }],
        }))
    }

    // dispatch buffer
    if pos_data.dispatch_fine_tiles.is_none() {
        pos_data.dispatch_fine_tiles = Some(device.create_buffer(&BufferDescriptor{
            label: Some("dispatch buffer"),
            size: std::mem::size_of::<DispatchIndirect>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
    }

    // fine output buffer
    if pos_data.fine_tile_output.is_none() {
        pos_data.fine_tile_output = Some(device.create_buffer(&BufferDescriptor{
            label: Some("fine output buffer"),
            size: (8 * VOXELS_PER_TILE_DIM * VOXELS_PER_TILE_DIM * VOXELS_PER_TILE_DIM * MAX_TILES) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }

    // init output buffers
    if output_data.output_buffer.is_none() || output_data.output_buffer.as_ref().unwrap().2 != state.cascades.len() {
        let vox_per_dim = VOXELS_PER_TILE_DIM * TILE_DIM_COUNT;
        let width = (vox_per_dim * state.cascades.len()) as u32;
    
        let texture_desc = TextureDescriptor {
            label: Some("sdf output texture"),
            mip_level_count: 1,
            sample_count: 1,
            dimension: bevy::render::render_resource::TextureDimension::D3,
            format: TextureFormat::R32Float,
            size: Extent3d {
                width,
                height: vox_per_dim as u32,
                depth_or_array_layers: vox_per_dim as u32,
            },
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        };

        let texture = device.create_texture(&texture_desc);
        let view = texture.create_view(&TextureViewDescriptor::default());

        pos_data
            .output_bindgroups
            .push(device.create_bind_group(&BindGroupDescriptor {
                label: Some("output bindgroup"),
                layout: &pipeline.output_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            }));
        output_data.output_buffer = Some((texture, view, state.cascades.len()));
    }

    pos_data.mesh_header_buffer.set(SdfMeshHeader {
        mesh_count: index_offset_counts.len() as u32,
        tri_count: vertex_count / 3,
        index_offset_count: index_offset_counts,
    });
    pos_data.mesh_header_buffer.write_buffer(&device, &queue);

    pos_data
        .transforms_buffer
        .set(SdfTransforms { t: transforms });
    pos_data.transforms_buffer.write_buffer(&device, &queue);

    pos_data.working_bindgroup = Some(device.create_bind_group(&BindGroupDescriptor {
        label: Some("sdf coarse bindgroup"),
        layout: &pipeline.working_layout,
        entries: &[
            // mesh header
            BindGroupEntry {
                binding: 0,
                resource: pos_data.mesh_header_buffer.binding().unwrap(),
            },
            // transforms
            BindGroupEntry {
                binding: 1,
                resource: pos_data.transforms_buffer.binding().unwrap(),
            },
            // triangles
            BindGroupEntry {
                binding: 2,
                resource: pos_data.tris_buffer.binding().unwrap(),
            },
            // cascade(s)
            BindGroupEntry {
                binding: 3,
                resource: pos_data.cascade_header_buffer.binding().unwrap(),
            },
            // triangle data out
            BindGroupEntry {
                binding: 4,
                resource: pos_data.coarse_tri_buffer.binding().unwrap(),
            },
            // counts out
            BindGroupEntry {
                binding: 5,
                resource: pos_data.coarse_counts_per_tile.binding().unwrap(),
            },
            // ids out
            BindGroupEntry {
                binding: 6,
                resource: pos_data.coarse_tile_ids.binding().unwrap(),
            },
            // jfa seed tex
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&pos_data.jfa_buffers.as_ref().unwrap().1),
            },
            // jfa dist tex
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(&pos_data.jfa_buffers.as_ref().unwrap().3),
            },
            // fine output tex
            BindGroupEntry {
                binding: 9,
                resource: pos_data.fine_tile_output.as_ref().unwrap().as_entire_binding(),
            }
        ],
    }));

    pos_data.dispatch_bindgroup = Some(device.create_bind_group(&BindGroupDescriptor {
        label: Some("sdf dispatch bindgroup"),
        layout: &pipeline.dispatch_layout,
        entries: &[
            // dispatch 
            BindGroupEntry {
                binding: 0,
                resource: pos_data.dispatch_fine_tiles.as_ref().unwrap().as_entire_binding(),
            }
        ],
    }));
        
    let headers = state.cascades.iter().enumerate().map(|(index, cascade)| SdfCascadeInfo {
        origin: cascade.last_origin.unwrap_or_default().extend(0),
        tile_size: cascade.settings.tile_size(),
        redraw: IVec4::ZERO,
        index: index as u32,
    });

    output_data.sdf_header_buffer.set(SdfCascadeInfos {
        num_cascades: state.cascades.len() as u32,
        cascades: headers.collect(),
    });
    output_data.sdf_header_buffer.write_buffer(&device, &queue);
}

#[derive(ShaderType)]
struct SdfCascadeInfo {
    origin: IVec4,
    redraw: IVec4,
    tile_size: f32,
    index: u32,
}

#[derive(ShaderType, Default)]
struct SdfCascadeInfos {
    num_cascades: u32,
    #[size(runtime)]
    cascades: Vec<SdfCascadeInfo>,
}

#[derive(ShaderType, Default, Debug)]
struct SdfMeshHeader {
    mesh_count: u32,
    tri_count: u32,
    #[size(runtime)]
    index_offset_count: Vec<UVec4>,
}

#[derive(ShaderType, Default)]
struct SdfTransforms {
    #[size(runtime)]
    t: Vec<Mat4>,
}

struct SceneSdfNode;

impl Node for SceneSdfNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pos_data = world.resource::<SdfWorkingData>();
        if pos_data.working_bindgroup.is_none() {
            return Ok(());
        }

        let state = world.resource::<SdfState>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SceneSdfPipeline>();

        let Some(coarse_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.coarse_pipeline) else {
            return Ok(());
        };
        let fine_pipeline = match pipeline_cache.get_compute_pipeline(pipeline.fine_pipeline) {
            Some(p) => p,
            None => {
                // println!("not ready");
                return Ok(());
            }
        };
        let fine_blend_pipeline = match pipeline_cache.get_compute_pipeline(pipeline.fine_blend_pipeline) {
            Some(p) => p,
            None => {
                // println!("not ready");
                return Ok(());
            }
        };
        let jfa_pipeline = match pipeline_cache.get_compute_pipeline(pipeline.jfa_pipeline) {
            Some(p) => p,
            None => {
                // println!("not ready");
                return Ok(());
            }
        };
        let stitch_pipeline = match pipeline_cache.get_compute_pipeline(pipeline.stitch_pipeline) {
            Some(p) => p,
            None => {
                // println!("not ready");
                return Ok(());
            }
        };
        let output_pipeline = match pipeline_cache.get_compute_pipeline(pipeline.output_pipeline) {
            Some(p) => p,
            None => {
                // println!("not ready");
                return Ok(());
            }
        };

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        let stamp = |pass: &mut ComputePass, ix: StampIndex|  {
            match pos_data.time_query.as_ref() {
                Some((q, ..)) => pass.write_timestamp(q, ix as u32),
                None => ()
            }
        };

        for cascade in state.cascades.iter() {
            if cascade.redraw != IVec3::ZERO {
                stamp(&mut pass, StampIndex::START);

                // coarse
                pass.set_pipeline(coarse_pipeline);
                pass.set_bind_group(0, pos_data.working_bindgroup.as_ref().unwrap(), &[0]);
                pass.set_bind_group(1, pos_data.dispatch_bindgroup.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(1, 1, 1);
                stamp(&mut pass, StampIndex::COARSE);

                // fine
                pass.set_pipeline(fine_pipeline);
                pass.dispatch_workgroups_indirect(pos_data.dispatch_fine_tiles.as_ref().unwrap(), 0);
                stamp(&mut pass, StampIndex::FINE);

                // fine blend
                pass.set_pipeline(fine_blend_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
                stamp(&mut pass, StampIndex::FINEBLEND);

                // jfa
                pass.set_pipeline(jfa_pipeline);
                let count = ((cascade.redraw.abs() + TILE_DIM_COUNT as i32 - 1) % TILE_DIM_COUNT as i32 + 1).as_uvec3();
                for offset in &pos_data.jfa_param_offsets {
                    pass.set_bind_group(1, pos_data.jfa_bindgroup.as_ref().unwrap(), &[*offset]);
                    pass.dispatch_workgroups(count.x, count.y, count.z);
                }
                stamp(&mut pass, StampIndex::JFA);

                let count = ((VOXELS_PER_TILE_DIM * TILE_DIM_COUNT) as f32 / 8.0).ceil() as u32;

                // stitch
                pass.set_pipeline(stitch_pipeline);
                pass.dispatch_workgroups(count, count, count);
                stamp(&mut pass, StampIndex::STITCH);

                // output
                pass.set_pipeline(output_pipeline);
                pass.set_bind_group(1, pos_data.output_bindgroups.get(0).as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(count, count, count);
                stamp(&mut pass, StampIndex::OUTPUT);

            }
        }

        drop(pass);

        if pos_data.record_stats {
            if let Some((q, bs, ix)) = pos_data.time_query.as_ref() {
                render_context.command_encoder.resolve_query_set(q, StampIndex::START as u32..StampIndex::OUTPUT as u32 + 1, &bs[*ix as usize], 0)
            }
        }

        Ok(())
    }
}

#[test]
fn how_does_floor_work() {
    println!("{:?}", (0.5f32.floor() as i32, (-0.5f32).floor() as i32));
    println!(
        "{}",
        IVec3::new(20, 49, -66)
            .min(IVec3::splat(25))
            .max(IVec3::splat(-25))
    );
}


#[test]
fn why_cant_i_get_intersection_tests_right() {
    let g_trans = GlobalTransform::from(Transform::from_xyz(10.0, 0.0, 0.0));
    let aabb = Aabb::from_min_max(-Vec3::ONE, Vec3::ONE);

    println!("g trans: {:#?}", g_trans);
    println!("aabb: {:#?}", aabb);

    for v in [-Vec3A::X, Vec3A::X, Vec3A::Y, -Vec3A::Y, -Vec3A::Z, Vec3A::Z] {
        let matrix = g_trans.compute_matrix();
        let point_on_plane = aabb.center + v * aabb.half_extents;
        let transformed_point = (matrix * point_on_plane.extend(1.0)).truncate();
        let transformed_normal = (matrix * -v.extend(0.0)).truncate();

        println!("v: {}", v);
        println!("point on plane: {}", v);
        println!("transformed point: {}", transformed_point);
        println!("transformed_normal: {}", transformed_normal);
    }
}