use crate::Settings;
use bevy::prelude::*;

/// Convert world coordinates to grid coordinates
pub fn world_to_grid(world_pos: Vec3, settings: &Settings) -> (u32, u32) {
	let world_x = settings.world_x();
	let world_z = settings.world_z();
	let grid_x = settings.grid_x();
	let grid_z = settings.grid_z();

	let grid_x_f = (world_pos.x + world_x / 2.0) / world_x * grid_x as f32;
	let grid_z_f = (world_pos.z + world_z / 2.0) / world_z * grid_z as f32;

	let grid_x_clamped = (grid_x_f as u32).min(grid_x.saturating_sub(1));
	let grid_z_clamped = (grid_z_f as u32).min(grid_z.saturating_sub(1));

	(grid_x_clamped, grid_z_clamped)
}

/// Convert grid coordinates to world coordinates
pub fn grid_to_world(grid_x: u32, grid_z: u32, settings: &Settings) -> Vec3 {
	let world_x = settings.world_x();
	let world_z = settings.world_z();
	let grid_x_count = settings.grid_x();
	let grid_z_count = settings.grid_z();

	let x_step = world_x / grid_x_count as f32;
	let z_step = world_z / grid_z_count as f32;

	let x_pos = (grid_x as f32).mul_add(x_step, -(world_x / 2.0));
	let z_pos = (grid_z as f32).mul_add(z_step, -(world_z / 2.0));

	Vec3::new(x_pos, 0.0, z_pos)
}

/// Calculate terrain height at given world coordinates
pub fn calculate_terrain_height(
	world_pos: Vec3,
	heightmap: &crate::HeightMap,
	settings: &Settings,
) -> f32 {
	let (grid_x, grid_z) = world_to_grid(world_pos, settings);
	let base_height = heightmap.get(grid_x, grid_z);

	base_height * world_size_for_height(settings) * settings.height_multiplier
}

/// Get the world size (commonly used calculation)
pub fn world_size(settings: &Settings) -> f32 {
	settings.world_x().max(settings.world_z())
}

/// Get the world size for height calculations
pub fn world_size_for_height(settings: &Settings) -> f32 {
	settings.world_x().min(settings.world_z())
}

/// Clamp world position to terrain bounds
pub fn clamp_to_terrain_bounds(world_pos: Vec3, settings: &Settings) -> Vec3 {
	let half_world_x = settings.world_x() / 2.0;
	let half_world_z = settings.world_z() / 2.0;

	Vec3::new(
		world_pos.x.clamp(-half_world_x, half_world_x),
		world_pos.y,
		world_pos.z.clamp(-half_world_z, half_world_z),
	)
}
