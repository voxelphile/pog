//!include src/base.wgsl

@group(1) @binding(0) var world_storage: texture_storage_3d<r32uint, write>;

fn world_gen(position: vec3<i32>) -> u32 {
	var pos = vec3<f32>(position);
	var seed = 400;
	var alpha = fbm(pos, seed);
	var beta = fbm(pos, seed + 1);
	var gamma = fbm(pos, seed + 2);
	var csample = sample_buffer.continentalness[u32(f32(alpha) * f32(max_samples))];
	var esample = sample_buffer.erosion[u32(f32(beta) * f32(max_samples))];
	var pandvsample = sample_buffer.pandv[u32(f32(gamma) * f32(max_samples))];
	var base_height = 60.0;
	var c_height = mix(base_height / 2.0, base_height * 2.0, csample);
	var e_height = mix(c_height, base_height, esample);
	var height = mix(mix(base_height / 4.0, base_height * 4.0, pandvsample), base_height, 1.0 - esample);

	if(position.z == i32(height)) {
		return u32(BLOCK_ID_GRASS);
	} else if(position.z < i32(height) && position.z > i32(height - 10.0)) {
		return u32(BLOCK_ID_DIRT);
	} else if(position.z < i32(height)) {
		return u32(BLOCK_ID_STONE);
	}

	return u32(BLOCK_ID_AIR);
}

@compute
@workgroup_size(4,4,4)
fn cs_world(
	@builtin(global_invocation_id) gid: vec3<u32>,
) {
	var pos = vec3<i32>(gid) + global_buffer.floating_origin - textureDimensions(world_storage) / 2;

	textureStore(world_storage, vec3<i32>(gid), vec4<u32>(
		world_gen(pos)
	));
}

