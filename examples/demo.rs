use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

use bevy_procedural_terrain_gen::{TerrainPlugin, hud::CameraDebugHud};

fn main() {
	App::new()
		.add_plugins(DefaultPlugins)
		.add_plugins(EguiPlugin::default())
		.add_plugins(PanOrbitCameraPlugin)
		.add_plugins(TerrainPlugin)
		.add_plugins(CameraDebugHud)
		.add_systems(Startup, setup_demo_scene)
		.run();
}

fn setup_demo_scene(mut commands: Commands) {
	// Simple orbit camera
	commands.spawn((
		Camera { ..default() },
		Transform::from_xyz(0.0, 600.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y),
		bevy_panorbit_camera::PanOrbitCamera::default(),
	));

	// UI camera for HUD overlay
	commands.spawn((
		Camera2d,
		Camera {
			order: 1,
			clear_color: ClearColorConfig::None,
			..default()
		},
	));

	// Directional light
	commands.spawn((
		DirectionalLight {
			illuminance: light_consts::lux::OVERCAST_DAY,
			shadows_enabled: true,
			..default()
		},
		Transform::from_xyz(500.0, 1000.0, 500.0).looking_at(Vec3::ZERO, Vec3::Y),
	));
}
