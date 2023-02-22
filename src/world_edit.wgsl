//!include src/base.wgsl

@group(1) @binding(0) var world_storage: texture_storage_3d<r32uint, write>;

fn world_gen_base(position: vec3<i32>) -> u32 {
	var pos = vec3<f32>(position);
	var seed = 400;
	
	var alpha = saturate(fbm(pos, seed));
	var beta = saturate(fbm(pos, seed + 1));
	var gamma = saturate(fbm(pos, seed + 2));
	
	var csample = sample_buffer.continentalness[u32(f32(alpha) * f32(max_samples))];
	var esample = sample_buffer.erosion[u32(f32(beta) * f32(max_samples))];
	var pandvsample = sample_buffer.pandv[u32(f32(gamma) * f32(max_samples))];

	var density = csample + pandvsample - esample;

	var height_offset = 60;
	var squashing_factor = 0.05;

	density -= f32(position.z - height_offset) / (1.0 / squashing_factor);

	if(density > 0.0) {
		return u32(BLOCK_ID_STONE);
	}

	return u32(BLOCK_ID_AIR);
}

fn world_gen(position: vec3<i32>) -> u32 {
	if(world_gen_base(position) == u32(BLOCK_ID_AIR)) {
		return u32(BLOCK_ID_AIR);
	}

	var air_above = 0;
	var max_air = 5;

	for(var i = 1; i < max_air; i++) {
		if(world_gen_base(position + vec3(0, 0, i)) == u32(BLOCK_ID_AIR)) {
			air_above++;
		}
	}

	if(air_above > max_air - 2) {
		return u32(BLOCK_ID_GRASS);
	} else if(air_above > 0) {
		return u32(BLOCK_ID_DIRT);
	}
	
	return u32(BLOCK_ID_STONE);
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

