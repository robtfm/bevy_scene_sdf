
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



mod debug;
mod render;

pub use debug::{DebugSdfPlugin, DebugSdf};

use bevy::prelude::*;

pub struct SceneSdfPlugin;

// cascade will try to update when frame.0 % frequency == offset
// if an earlier cascade is also scheduled for the same frame and needs to update then it will take priority
// only 1 cascade update is ever run in a single frame
// todo make it update the most needy cascade automatically
// todo should most of this be async compute ..? would probably need a double/triple buffer on output at least
//  and some thought about cascade info settings used in render loop
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

impl CascadeSettings {
    fn tile_size(&self) -> f32 {
        self.far_distance / TILE_DIM_COUNT as f32 * 2.0
    }
}

pub enum ExtractionFilter {
    Marked,
    Unmarked,
}

// move this to scroll the cascade origins
#[derive(Resource, Default)]
pub struct SdfCentre(pub Vec3);

#[derive(Resource)]
pub struct SceneSdfSettings {
    // not implemented
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
        .init_resource::<SdfCentre>()
        ;

        render::setup_render(app);
    }
}