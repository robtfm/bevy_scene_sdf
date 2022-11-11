use bevy::{
    core_pipeline::{core_3d, fullscreen_vertex_shader::fullscreen_shader_vertex_state},
    prelude::*,
    render::{
        render_graph::{Node, RenderGraph, SlotInfo, SlotType},
        render_resource::{
            BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
            BufferBindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites, 
            FragmentState, MultisampleState, Operations, PipelineCache, PrimitiveState,
            RenderPassDescriptor, RenderPipelineDescriptor,
            ShaderStages, ShaderType,
            BindGroupEntry, BindGroupDescriptor, BindingResource, BindGroup, SamplerDescriptor, AddressMode, FilterMode, SamplerBindingType,
        },
        renderer::RenderDevice,
        view::{ViewTarget, ViewUniform, ViewUniforms, ViewUniformOffset},
        RenderApp, RenderStage, extract_component::{ExtractComponentPlugin, ExtractComponent},
    },
};

use crate::{SdfCascadeInfos, SdfOutputData, queue_sdf_data};

pub const DEBUG_NODE: &str = "sdf_debug_node";

pub struct DebugSdfPlugin;

impl Plugin for DebugSdfPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<DebugSdf>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<DebugSdfPipeline>();
        render_app.add_system_to_stage(RenderStage::Queue, queue_debug_view_bindgroup.after(queue_sdf_data));

        let debug_node = DebugSdfNode::new(&mut render_app.world);
        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        let graph_3d = render_graph
            .get_sub_graph_mut(core_3d::graph::NAME)
            .unwrap();

        graph_3d.add_node(DEBUG_NODE, debug_node);

        graph_3d
            .add_slot_edge(
                graph_3d.input_node().unwrap().id,
                core_3d::graph::input::VIEW_ENTITY,
                DEBUG_NODE,
                DebugSdfNode::IN_VIEW,
            )
            .unwrap();

        graph_3d
            .add_node_edge(core_3d::graph::node::MAIN_PASS, DEBUG_NODE)
            .unwrap();

        graph_3d
            .add_node_edge(DEBUG_NODE, core_3d::graph::node::TONEMAPPING)
            .unwrap();
    }
}

#[derive(Resource)]
struct DebugSdfPipeline {
    layout: BindGroupLayout,
    bind_group: Option<BindGroup>,
    pipeline: CachedRenderPipelineId,
}

impl FromWorld for DebugSdfPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("debug sdf pipeline layout"),
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // cascade infos
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { 
                        ty: BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false, 
                        min_binding_size: Some(SdfCascadeInfos::min_size()),
                    },
                    count: None,
                },
                // sdf texture
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: bevy::render::render_resource::TextureSampleType::Float { filterable: true },
                        view_dimension: bevy::render::render_resource::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // sampler
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let debug_shader = world.resource::<AssetServer>().load("shader/debug.wgsl");

        let descriptor = RenderPipelineDescriptor {
            label: Some("debug sdf pipeline".into()),
            layout: Some(vec![layout.clone()]),
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: debug_shader,
                shader_defs: vec![],
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
        };

        let mut cache = world.resource_mut::<PipelineCache>();

        Self {
            layout,
            pipeline: cache.queue_render_pipeline(descriptor),
            bind_group: None,
        }
    }
}

#[derive(Component, Clone, Copy)]
pub struct DebugSdf;

impl ExtractComponent for DebugSdf {
    type Query = &'static Self;

    type Filter = ();

    fn extract_component(item: bevy::ecs::query::QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

// #[derive(Component)]
// struct DebugSdfBindings {
//     output: CachedTexture,
// }

fn queue_debug_view_bindgroup(
    render_device: Res<RenderDevice>,
    output_data: Res<SdfOutputData>,
    view_uniforms: Res<ViewUniforms>,
    mut pipeline: ResMut<DebugSdfPipeline>,
) {
    pipeline.bind_group = None;

    let header = match output_data.sdf_header_buffer.binding() {
        Some(b) => b,
        None => return,
    };

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let sampler = render_device.create_sampler(&SamplerDescriptor{ 
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
    
        let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("sdf debug view bindgroup"),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: view_binding.clone(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: header,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&output_data.output_buffer.as_ref().unwrap().1),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
            layout: &pipeline.layout,
        });

        pipeline.bind_group = Some(view_bind_group);
    }
}

pub struct DebugSdfNode {
    query: QueryState<(&'static ViewTarget, &'static ViewUniformOffset), With<DebugSdf>>,
}

impl DebugSdfNode {
    pub const IN_VIEW: &'static str = "view";

    fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for DebugSdfNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let debug_pipeline = world.resource::<DebugSdfPipeline>();

        let (target, uniform_offset) = match self.query.get_manual(world, view_entity) {
            Ok(t) => t,
            Err(_) => return Ok(())
        };

        let pipeline = match world
            .resource::<PipelineCache>()
            .get_render_pipeline(debug_pipeline.pipeline)
        {
            Some(p) => p,
            None => return Ok(()),
        };
        let bind_group = match debug_pipeline.bind_group.as_ref() {
            Some(g) => g,
            None => return Ok(()),
        };

        let mut pass = render_context
            .command_encoder
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("debug sdf pass"),
                color_attachments: &[Some(target.get_color_attachment(Operations::default()))],
                depth_stencil_attachment: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[uniform_offset.offset]);
        pass.draw(0..3, 0..1);

        Ok(())
    }
}
