// `cargo run --release --example simple X`
// to generate a grid of cubes
// `cargo run --release --example simple`
// to try and load bistro.glb
// 9 and 0 to change cascade sizes
// 8 to refresh cascades - need to do this once after bistro loads (i'm too lazy to catch the scene loaded event)

use std::f32::consts::PI;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*, input::mouse::MouseMotion, core::FrameCount,
};
use bevy_scene_sdf::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let args = Args {
        cubes: args.get(1).map(|a| a.parse::<usize>().unwrap()).unwrap_or(0),
        stopat: args.get(2).map(|a| a.parse::<usize>().unwrap()).unwrap_or(0),
    };

    App::new()
        // .insert_resource(LogSettings {
        //     level: Level::DEBUG,
        //     filter: "wgpu=info".to_string(),
        // })
        .insert_resource(Msaa { samples: 1 })
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 0.5,
        })
        .insert_resource(args)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                present_mode: bevy::window::PresentMode::Immediate,
                ..default()
            },
            ..default()
        }))
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(SceneSdfPlugin)
        .add_plugin(DebugSdfPlugin)
        .add_startup_system(setup)
        .add_system(camera_controller)
        .add_system(toggle)
        .run()
}

#[derive(Resource)]
struct Args{
    cubes: usize,
    #[allow(dead_code)]
    stopat: usize,
}

fn setup(
    mut commands: Commands,
    #[allow(unused_variables)]
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<Args>,
    mut centre: ResMut<SdfCentre>,
) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-5.0, 0.0, 0.0)
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        ..default()
    })
    .insert(DebugSdf)
    .insert(CameraController::default());

    centre.0 = Vec3::new(-5.0, 0.0, 0.0);

    // Plane
    // commands.spawn(PbrBundle {
    //     mesh: meshes.add(Mesh::from(shape::Plane { size: 100.0 })),
    //     material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
    //     ..default()
    // })
    //     .insert(Name::new("Ground"));

    let cube_range = args.cubes as i32;

    // lil cubes
    let mat = materials.add(Color::rgb(0.7, 0.5, 1.0).into());
    let mesh = meshes.add(shape::Cube::new(10.0).into());
    commands
        .spawn(SpatialBundle {
            transform: Transform::from_xyz(1.0, 1.0, 1.0).with_rotation(Quat::from_euler(
                EulerRot::XYZ,
                20.0,
                30.0,
                40.0,
            )),
            // transform: Transform::from_xyz(0.01, 0.01, 0.01),
            // transform: Transform::from_xyz(1.0, 1.0, 1.0),
            ..Default::default()
        })
        .insert(Name::new("grid_squares"))
        .with_children(|s| {
            for i in -cube_range+1..cube_range {
                for j in -cube_range+1..cube_range {
                    let k = 0;
                    // for k in -cube_range+1..cube_range {
                        s.spawn(PbrBundle {
                            mesh: mesh.clone(),
                            material: mat.clone(),
                            transform: Transform::from_translation(IVec3::new(i, j, k).as_vec3() * 25.0),
                            ..Default::default()
                        })
                        .insert(Cube);
                    // }
                }
            }
        });

    // Light
    commands.spawn(PointLightBundle {
        transform: Transform::from_xyz(0.0, 2.0, 0.0),
        point_light: PointLight {
            shadows_enabled: true,
            ..Default::default()
        },
        ..default()
    });

    // scene
    if args.cubes == 0 {
        commands.spawn(SceneBundle {
            scene: asset_server.load("gltf/bistro.glb#Scene0"),
            // scene: asset_server.load("gltf/sponza/gltf/sponza.gltf#Scene0"),
            ..default()
        })
        .insert(Name::new("scene"))
        .insert(Visibility::INVISIBLE);
    }
}


#[derive(Component)]
struct CameraController {
    pub enabled: bool,
    pub initialized: bool,
    pub sensitivity: f32,
    pub key_forward: KeyCode,
    pub key_back: KeyCode,
    pub key_left: KeyCode,
    pub key_right: KeyCode,
    pub key_up: KeyCode,
    pub key_down: KeyCode,
    pub key_run: KeyCode,
    pub mouse_key_enable_mouse: MouseButton,
    pub keyboard_key_enable_mouse: KeyCode,
    pub walk_speed: f32,
    pub run_speed: f32,
    pub friction: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub velocity: Vec3,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            enabled: true,
            initialized: false,
            sensitivity: 0.5 / 100.0,
            key_forward: KeyCode::W,
            key_back: KeyCode::S,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_up: KeyCode::E,
            key_down: KeyCode::Q,
            key_run: KeyCode::LShift,
            mouse_key_enable_mouse: MouseButton::Left,
            keyboard_key_enable_mouse: KeyCode::M,
            walk_speed: 5.0,
            run_speed: 15.0,
            friction: 0.5,
            pitch: 0.0,
            yaw: 0.0,
            velocity: Vec3::ZERO,
        }
    }
}

#[derive(Component)]
struct Cube;

fn toggle(
    mut commands: Commands,
    mut centre: ResMut<SdfCentre>,
    key_input: Res<Input<KeyCode>>,
    mut q: Query<(Entity, Option<&DebugSdf>, &GlobalTransform, &mut Transform), With<Camera3d>>,
    mut base: Query<&mut Visibility, With<Handle<Scene>>>,
    mut state: ResMut<SceneSdfSettings>,
    mut still: Local<bool>,
    #[allow(unused_variables)]
    frame: Res<FrameCount>,
    #[allow(unused_variables)]
    args: Res<Args>,
) {
    #[allow(unused_mut, unused_variables)]
    let (e, maybe_debug, trans, mut m_trans) = q.single_mut();
    if key_input.just_pressed(KeyCode::P) {
        if maybe_debug.is_some() {
            commands.entity(e).remove::<DebugSdf>();
            centre.0 = Vec3::splat(-500.0);
            if let Ok(mut vis) = base.get_single_mut() {
                vis.is_visible = true;
            }
        } else {
            commands.entity(e).insert(DebugSdf);
            if let Ok(mut vis) = base.get_single_mut() {
                vis.is_visible = false;
            }
        }
    }

    if key_input.just_pressed(KeyCode::Key8) {
        for cascade in &mut state.cascades {
            let f = cascade.far_distance;
            cascade.far_distance = 1.0;
            cascade.far_distance = f;
        }
    }

    if key_input.just_pressed(KeyCode::Key9) {
        for cascade in &mut state.cascades {
            cascade.far_distance *= 1.1;
        }
    }

    if key_input.just_pressed(KeyCode::Key0) {
        for cascade in &mut state.cascades {
            cascade.far_distance /= 1.1;
        }
    }

    if key_input.just_pressed(KeyCode::F) {
        *still = !*still;
    }

    if !*still {
        // some code for automating movement and freezing after some time, for grabbing consistent renderdocs

        // 1 107
        // let f = match args.stopat {
        //     0 => frame.0,
        //     stopat => ((frame.0.max(100)) - 100).min(stopat as u32),
        // };

        // if frame.0 == 100 {
        //     // just bump it
        //     state.filter = ExtractionFilter::Unmarked;
        // }
        // let mut angle = f as f32 / 2000.0;
        // if angle % std::f32::consts::PI * 4.0 > std::f32::consts::PI * 2.0 {
        //     angle = std::f32::consts::PI * 2.0 - angle % std::f32::consts::PI * 2.0;
        // } else {
        //     angle = angle % std::f32::consts::PI * 2.0;
        // }
        // m_trans.translation = Vec3::new(angle.sin(), 0.0, angle.cos()) * 20.0;
        // // m_trans.translation = Vec3::new(0.0 - (f as f32 / 2000.0).min(15.0), 0.0, 3.0 - 5.0 * (f as f32 / 5000.0).sin());
        // *m_trans = m_trans.looking_at(Vec3::ZERO, Vec3::Y);
        centre.0 = trans.translation();
    }
}

fn camera_controller(
    time: Res<Time>,
    mut mouse_events: EventReader<MouseMotion>,
    mouse_button_input: Res<Input<MouseButton>>,
    key_input: Res<Input<KeyCode>>,
    mut move_toggled: Local<bool>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
) {
    let dt = time.delta_seconds();

    if let Ok((mut transform, mut options)) = query.get_single_mut() {
        if !options.initialized {
            let (yaw, pitch, _roll) = transform.rotation.to_euler(EulerRot::YXZ);
            options.yaw = yaw;
            options.pitch = pitch;
            options.initialized = true;
        }
        if !options.enabled {
            return;
        }

        // Handle key input
        let mut axis_input = Vec3::ZERO;
        if key_input.pressed(options.key_forward) {
            axis_input.z += 1.0;
        }
        if key_input.pressed(options.key_back) {
            axis_input.z -= 1.0;
        }
        if key_input.pressed(options.key_right) {
            axis_input.x += 1.0;
        }
        if key_input.pressed(options.key_left) {
            axis_input.x -= 1.0;
        }
        if key_input.pressed(options.key_up) {
            axis_input.y += 1.0;
        }
        if key_input.pressed(options.key_down) {
            axis_input.y -= 1.0;
        }
        if key_input.just_pressed(options.keyboard_key_enable_mouse) {
            *move_toggled = !*move_toggled;
        }

        // Apply movement update
        if axis_input != Vec3::ZERO {
            let max_speed = if key_input.pressed(options.key_run) {
                options.run_speed
            } else {
                options.walk_speed
            };
            options.velocity = axis_input.normalize() * max_speed;
        } else {
            let friction = options.friction.clamp(0.0, 1.0);
            options.velocity *= 1.0 - friction;
            if options.velocity.length_squared() < 1e-6 {
                options.velocity = Vec3::ZERO;
            }
        }
        let forward = transform.forward();
        let right = transform.right();
        transform.translation += options.velocity.x * dt * right
            + options.velocity.y * dt * Vec3::Y
            + options.velocity.z * dt * forward;

        // Handle mouse input
        let mut mouse_delta = Vec2::ZERO;
        if mouse_button_input.pressed(options.mouse_key_enable_mouse) || *move_toggled {
            for mouse_event in mouse_events.iter() {
                mouse_delta += mouse_event.delta;
            }
        } else {
            mouse_events.clear();
        }

        if mouse_delta != Vec2::ZERO {
            // Apply look update
            options.pitch = (options.pitch - mouse_delta.y * 0.5 * options.sensitivity)
                .clamp(-PI / 2., PI / 2.);
            options.yaw -= mouse_delta.x * options.sensitivity;
            transform.rotation = Quat::from_euler(EulerRot::ZYX, 0.0, options.yaw, options.pitch);
        }
    }
}
