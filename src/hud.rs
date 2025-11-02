use bevy::prelude::*;

#[derive(Component)]
pub struct HudText;

pub struct CameraDebugHud;

impl Plugin for CameraDebugHud {
	fn build(&self, app: &mut App) {
		app
			.add_systems(Startup, setup_hud)
			.add_systems(Update, update_hud);
	}
}

fn setup_hud(mut commands: Commands) {
	commands
		.spawn((Node {
			padding: UiRect::all(Val::Px(10.0)),
			justify_self: JustifySelf::Start,
			align_self: AlignSelf::Start,
			..default()
		},))
		.with_child((
			HudText,
			Text::new("Camera Transform: Loading..."),
			TextFont {
				font_size: 18.0,
				..default()
			},
			TextColor(Color::WHITE),
			TextLayout::new_with_justify(Justify::Left),
			Node { ..default() },
		));
	debug!("HUD setup complete");
}

fn update_hud(
	camera_query: Single<&Transform, With<bevy_panorbit_camera::PanOrbitCamera>>,
	mut hud_text: Single<&mut Text, With<HudText>>,
) {
	let camera_transform = camera_query;

	let translation = camera_transform.translation;
	let rotation = camera_transform.rotation;

	// Convert rotation to euler angles for display
	let euler = rotation.to_euler(EulerRot::YXZ);

	let text = format!(
		"Camera Transform:\n\
            Position: ({:.2}, {:.2}, {:.2})\n\
            Rotation: ({:.2}, {:.2}, {:.2}) deg",
		translation.x,
		translation.y,
		translation.z,
		euler.0.to_degrees(),
		euler.1.to_degrees(),
		euler.2.to_degrees()
	);
	hud_text.0 = text;
}
