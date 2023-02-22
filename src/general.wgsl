//!include src/base.wgsl

@group(1) @binding(0) var world_texture: texture_3d<u32>;

@compute
@workgroup_size(1)
fn cs_perframe() {
	var size: vec3<i32> = vec3<i32>(region_size);
	
	var dst = distance(vec3<f32>(global_buffer.floating_origin), perframe_buffer.camera.position.xyz);
	
	if(global_buffer.load == 0 || dst > view_distance) {
		indirect_buffer.create_chunk_x = u32(size.x / chunk_size);
		indirect_buffer.create_chunk_y = u32(size.y / chunk_size);
		indirect_buffer.create_chunk_z = u32(size.z / chunk_size);

		batch_buffer.batch_count = u32(0);
		for(var i = 0; i <= max_batches; i++) {
			batch_buffer.batches[i].batch_vertex_count = u32(36);
			batch_buffer.batches[i].batch_instance_count = u32(0);
			batch_buffer.batches[i].batch_base_instance = u32(i * 100000);
		}
		
		global_buffer.floating_origin = vec3<i32>(perframe_buffer.camera.position.xyz);
		var size: vec3<i32> = vec3<i32>(textureDimensions(world_texture).xyz);
		indirect_buffer.create_noise_x = u32(size.x / 4);
		indirect_buffer.create_noise_y = u32(size.y / 4);
		indirect_buffer.create_noise_z = u32(size.z / 4);
	} else {
		indirect_buffer.create_chunk_x = u32(0);
		indirect_buffer.create_chunk_y = u32(0);
		indirect_buffer.create_chunk_z = u32(0);
		indirect_buffer.create_noise_x = u32(0);
		indirect_buffer.create_noise_y = u32(0);
		indirect_buffer.create_noise_z = u32(0);
	}

	global_buffer.load = 1;
}

var<workgroup> solid_blocks: atomic<i32>;

@compute
@workgroup_size(4, 4, 4)
fn cs_create_chunks(
	@builtin(global_invocation_id) gid: vec3<u32>,
	@builtin(local_invocation_index) lid : u32
) {
	var voxel_pos = vec3<i32>(gid);

	if(voxel_query(voxel_pos)) {
		var up_block = !voxel_query(
			voxel_pos
			+ vec3<i32>(0, 0, 1)
		);
		var down_block = !voxel_query(
			voxel_pos
			+ vec3<i32>(0, 0, -1)
		);
		var left_block = !voxel_query(
			voxel_pos
			+ vec3<i32>(0, -1, 0)
		);
		var right_block = !voxel_query(
			voxel_pos
			+ vec3<i32>(0, 1, 0)
		);
		var forward_block = !voxel_query(
			voxel_pos
			+ vec3<i32>(1, 0, 0)
		);
		var backward_block = !voxel_query(
			voxel_pos
			+ vec3<i32>(-1, 0, 0)
		);
		
		var exposed = up_block 
			|| down_block 
			|| left_block 
			|| right_block 
			|| forward_block 
			|| backward_block;

		if(exposed) {
			atomicAdd(&solid_blocks, 1);
		}
	}

	workgroupBarrier();

	if(lid == u32(0) && solid_blocks != 0) {
		var batch_count: u32;
		var instance: u32;
	
		{
			var pos = perframe_buffer.camera.transform[3].xyz;
			var min = vec3<f32>(voxel_pos) - 1.0;
			var max = vec3<f32>(voxel_pos) + 1.0 + f32(chunk_size);

			var near_chunk = pos.x >= min.x
				&& pos.y >= min.y
				&& pos.z >= min.z
				&& pos.x < max.x
				&& pos.y < max.y
				&& pos.z < max.z;

			if(near_chunk) {
				//put this chunk in two batches
				//the last batch draws back faces,
				//which we draw in addition to the front faces
				var instance = atomicAdd(&batch_buffer.batches[max_batches].batch_instance_count, u32(1));
				batch_buffer.batches[max_batches].batch_chunks[instance].position = vec4(voxel_pos / chunk_size, 1); 
				
			}
		}

		
		loop {
			batch_count = atomicLoad(&batch_buffer.batch_count);
			instance = atomicLoad(&batch_buffer.batches[batch_count].batch_instance_count);
			if(batch_count >= u32(max_batches)) {
				return;
			}

			if(instance >= u32(100000)) {
				if(atomicCompareExchangeWeak(&batch_buffer.batch_count, batch_count, batch_count + u32(1)).exchanged) {
					batch_count += u32(1);
				}
				continue;
			}

			if(atomicCompareExchangeWeak(&batch_buffer.batches[batch_count].batch_instance_count, instance, instance + u32(1)).exchanged) {
				break;
			}
		}
		
				
		batch_buffer.batches[batch_count].batch_chunks[instance].position = vec4(voxel_pos / chunk_size, 1); 
	}
}


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) @interpolate(flat) chunk_position: vec4<f32>,
    @location(3) local_position: vec4<f32>,
    @location(4) @interpolate(flat) chunk_normal: vec4<f32>,
};

@vertex
fn vs_main(
   	@builtin(instance_index) i: u32,
    	@builtin(vertex_index) j: u32,
) -> VertexOutput {
	var indices = array<i32, 36>(
		1, 0, 3, 1, 3, 2,
		4, 5, 6, 4, 6, 7, 
		2, 3, 7, 2, 7, 6, 
		5, 4, 0, 5, 0, 1, 
		6, 5, 1, 6, 1, 2, 
		3, 0, 4, 3, 4, 7
	);

	var offsets = array<vec3<f32>, 8>(
		vec3<f32>(0.0, 0.0, 1.0),
        	vec3<f32>(0.0, 1.0, 1.0),
        	vec3<f32>(1.0, 1.0, 1.0),
        	vec3<f32>(1.0, 0.0, 1.0),
        	vec3<f32>(0.0, 0.0, 0.0),
        	vec3<f32>(0.0, 1.0, 0.0),
        	vec3<f32>(1.0, 1.0, 0.0),
		vec3<f32>(1.0, 0.0, 0.0)
	);

	var normals = array<vec3<f32>, 6>(
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, -1.0),
		vec3(1.0, 0.0, 0.0),
		vec3(-1.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, -1.0, 0.0),
	);

	var x = i / u32(100000);
	var y = i % u32(100000);
	
	var out: VertexOutput;

	out.chunk_normal = vec4(normals[j / u32(6)], 0.0);
	out.local_position = vec4(f32(chunk_size) * offsets[indices[j]], 1.0);

	out.chunk_position = vec4<f32>(f32(chunk_size) * vec3<f32>(batch_buffer.batches[x].batch_chunks[y].position.xyz), 1.0);

	out.world_position = vec4<f32>(out.local_position.xyz + out.chunk_position.xyz, 1.0);
    	out.clip_position = perframe_buffer.camera.projection * perframe_buffer.camera.view * out.world_position;
    	return out;
}

const MAX_STEP_COUNT = 1024;
const RAY_STATE_INITIAL = 0;
const RAY_STATE_OUT_OF_BOUNDS = 1;
const RAY_STATE_MAX_DIST_REACHED = 2;
const RAY_STATE_MAX_STEP_REACHED = 3;
const RAY_STATE_VOXEL_FOUND = 4;

struct Ray {
	origin: vec3<f32>,
	offset: vec3<f32>,
	direction: vec3<f32>,
	minimum: vec3<i32>,
	maximum: vec3<i32>,
	max_distance: f32,
}

struct RayState {
	ray: Ray,
	id: i32,
	position: vec3<i32>,
	dist: f32,
	mask: vec3<bool>,
	side_dist: vec3<f32>,
	delta_dist: vec3<f32>,
	ray_step: vec3<i32>,
	step_count: i32,
	block_id: u32,
}

struct RayHit {
	ray: Ray,
	dist: f32,
	normal: vec3<i32>,
	back_step: vec3<i32>,
	mask: vec3<bool>,
	destination: vec3<f32>,
	uv: vec2<f32>,
	block_id: u32,
}

fn ray_cast_start(ray: Ray) -> RayState {
	var state: RayState;

	state.id = RAY_STATE_INITIAL;
	state.ray = ray;
	state.ray.direction = normalize(ray.direction);
	state.position = vec3<i32>(floor(ray.origin + 0.));
	state.dist = 0.0;
	state.mask = vec3(false);
	state.delta_dist = 1.0 / abs(state.ray.direction);
	state.ray_step = vec3<i32>(sign(state.ray.direction));
	state.side_dist = (sign(state.ray.direction) * (vec3<f32>(state.position) - state.ray.origin) + (sign(state.ray.direction) * 0.5) + 0.5) * state.delta_dist;
	state.step_count = 0;

	return state;
}

fn ray_cast_complete(state: RayState) -> RayHit {
	var destination = state.ray.origin + state.ray.direction * state.dist;
	var back_step = vec3<i32>(state.position - state.ray_step * vec3<i32>(state.mask));
	var uv = vec2(dot(vec3<f32>(state.mask) * destination.yzx, vec3(1.0)), dot(vec3<f32>(state.mask) * destination.zxy, vec3(1.0))) % vec2(1.0);
	var normal = vec3<i32>(vec3<i32>(state.mask) * vec3<i32>(sign(-state.ray.direction)));

	var hit: RayHit;
	hit.destination = destination;
	hit.block_id = state.block_id;
	hit.mask = state.mask;
	hit.back_step = back_step;
	hit.uv = uv;
	hit.normal = normal;
	hit.dist = state.dist;
	hit.ray = state.ray;
	
	return hit;
}

fn ray_cast_check_over_step_count(state: ptr<function, RayState>) -> bool {
	if((*state).step_count > MAX_STEP_COUNT) {
		(*state).id = RAY_STATE_MAX_STEP_REACHED;
		return true;
	}
	(*state).step_count += 1;
	return false;
}

fn ray_cast_check_over_dist(state: ptr<function, RayState>) -> bool {
	if((*state).dist > (*state).ray.max_distance) {
		(*state).id = RAY_STATE_MAX_DIST_REACHED;
		return true;
	}
	return false;
}

fn ray_cast_check_out_of_bounds(state: ptr<function, RayState>, fluff: i32) -> bool {
	var in_bounds = (*state).position.x >= (*state).ray.minimum.x - fluff 
		&& (*state).position.y >= (*state).ray.minimum.y - fluff
		&& (*state).position.z >= (*state).ray.minimum.z - fluff
		&& (*state).position.x < (*state).ray.maximum.x + fluff
		&& (*state).position.y < (*state).ray.maximum.y + fluff
		&& (*state).position.z < (*state).ray.maximum.z + fluff;

	if(!in_bounds) {
		(*state).id = RAY_STATE_OUT_OF_BOUNDS;
		return true;
	}

	return false;
}

fn ray_cast_check_failure(state: ptr<function, RayState>) -> bool {
	return ray_cast_check_over_dist(state) 
		|| ray_cast_check_out_of_bounds(state, 0) 
		|| ray_cast_check_over_step_count(state);
}

fn ray_cast_check_success(state: ptr<function, RayState>) -> bool {
	if(voxel_query((*state).position)) {
		(*state).id = RAY_STATE_VOXEL_FOUND;
		(*state).block_id = voxel_id((*state).position);
		return true;
	}

	return false;
}

fn ray_cast_body(state: ptr<function, RayState>) {
	(*state).mask.x = (*state).side_dist.x <= min((*state).side_dist.y, (*state).side_dist.z);
	(*state).mask.y = (*state).side_dist.y <= min((*state).side_dist.z, (*state).side_dist.x);
	(*state).mask.z = (*state).side_dist.z <= min((*state).side_dist.x, (*state).side_dist.y);
        (*state).side_dist += vec3<f32>((*state).mask) * (*state).delta_dist;
        (*state).position += vec3<i32>((*state).mask) * (*state).ray_step;
	(*state).dist = length(vec3<f32>((*state).mask) * ((*state).side_dist - (*state).delta_dist)) / length((*state).ray.direction);
}

fn ray_cast_drive(state: ptr<function, RayState>) -> bool {
	if((*state).id != RAY_STATE_INITIAL) {
		return true;
	}
	
	if(ray_cast_check_failure(state)) {
		return true;
	}
	
	if(ray_cast_check_success(state)) {
		return true;
	}

	ray_cast_body(state);
	
	return false;
}

fn vertex_ao(side: vec2<f32>, corner: f32) -> f32 {
	return (side.x + side.y + max(corner, side.x * side.y)) / 3.0;
}

fn voxel_id(position: vec3<i32>) -> u32 {
	var texture_position = position
		- global_buffer.floating_origin
		+ textureDimensions(world_texture) / 2;
		return textureLoad(world_texture, texture_position, 0).r;
}

fn voxel_query(position: vec3<i32>) -> bool {
	return voxel_id(position) != u32(BLOCK_ID_AIR); 
}

fn voxel_ao(position: vec3<i32>, d1: vec3<i32>, d2: vec3<i32>) -> vec4<f32> {
	var voxel_position: vec3<i32>;
	var side: vec4<f32>;

	voxel_position = position + d1;
	side.x = f32(voxel_query(voxel_position)); 
	voxel_position = position + d2;
	side.y = f32(voxel_query(voxel_position)); 
	voxel_position = position - d1;
	side.z = f32(voxel_query(voxel_position)); 
	voxel_position = position - d2;
	side.w = f32(voxel_query(voxel_position));

	var corner: vec4<f32>;

	voxel_position = position + d1 + d2;
	corner.x = f32(voxel_query(voxel_position)); 
	voxel_position = position - d1 + d2;
	corner.y = f32(voxel_query(voxel_position)); 
	voxel_position = position - d1 - d2;
	corner.z = f32(voxel_query(voxel_position)); 
	voxel_position = position + d1 - d2;
	corner.w = f32(voxel_query(voxel_position));

	var ret: vec4<f32>;
	ret.x = vertex_ao(side.xy, corner.x);
	ret.y = vertex_ao(side.yz, corner.y);
	ret.z = vertex_ao(side.zw, corner.z);
	ret.w = vertex_ao(side.wx, corner.w);
	return 1.0 - ret;
}

struct FragmentOutput {
	@builtin(frag_depth) depth: f32,
	@location(0) display: vec4<f32>,
}

fn map_range(s: f32, a1: f32, a2: f32, b1: f32, b2: f32) -> f32
{
    return b1 + (s-a1)*(b2-b1)/(a2-a1);
}

fn map_range3d(s: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, b1: vec3<f32>, b2: vec3<f32>) -> vec3<f32>
{
	return vec3<f32>(
		map_range(s.x, a1.x, a2.x, b1.x, b2.x),
		map_range(s.y, a1.y, a2.y, b1.y, b2.y),
		map_range(s.z, a1.z, a2.z, b1.z, b2.z),
	);
}

@fragment
fn fs_main(
	in: VertexOutput,
	@builtin(front_facing) front: bool,
) -> FragmentOutput {
	var output: FragmentOutput;

	var v_position = in.chunk_position.xyz 
			+ in.local_position.xyz
			+ 1e-3 * vec3<f32>(in.chunk_normal.xyz);
	var o_position = perframe_buffer.camera.transform[3].xyz;

	var origin: vec3<f32>;
	
	if(front) {
		origin = v_position;
	} else {
		origin = o_position;
	}

	var ray: Ray;
	ray.origin = origin;
	ray.direction = normalize(v_position - o_position);
	ray.max_distance = 1000.0;
	// TODO see if I can eliminate the fluff?
	ray.minimum = vec3<i32>(in.chunk_position.xyz) - 1;
	ray.maximum = vec3<i32>(in.chunk_position.xyz) + chunk_size + 1;

	var ray_state = ray_cast_start(ray);

	loop {
		if(ray_cast_drive(&ray_state)) {
			break;
		}
	}

	var ray_hit = ray_cast_complete(ray_state);

	if(ray_state.id == RAY_STATE_OUT_OF_BOUNDS) {
		discard;
	}

	var color = vec3(1.0);

/*
	var border = (textureDimensions(world_texture).x - region_size) / 2;
	var pos = ray_hit.destination 
		+ floor(perframe_buffer.camera.position.xyz)
		- vec3<f32>(global_buffer.floating_origin)
		+ vec3<f32>(f32(border)); 
*/
	var noise_factor = 0.5; /* saturate(
		map_range(
			textureLoad(world_texture, vec3<i32>(pos), 0).r, 
			-sqrt(3.0) / 2.0, 
			sqrt(3.0) / 2.0, 
			-1.5, 
			2.5
		)
	);*/

	if(ray_hit.block_id == u32(BLOCK_ID_GRASS)) {
		color *= mix(vec3<f32>(84.0, 127.0, 68.0) / 256.0, vec3<f32>(34.0, 139.0, 34.0) / 256.0, noise_factor);

	}
	if(ray_hit.block_id == u32(BLOCK_ID_STONE)) {
		color *= mix(vec3<f32>(135.0) / 256.0, vec3<f32>(20.0) / 256.0, noise_factor);
	}

	if(ray_hit.block_id == u32(BLOCK_ID_DIRT)) {
		color *= mix(vec3<f32>(107.0, 84.0, 40.0) / 256.0, vec3<f32>(64.0, 41.0, 5.0) / 256.0, noise_factor);
	}

	var ambient = voxel_ao(
		ray_hit.back_step,
		abs(ray_hit.normal.zxy), 
		abs(ray_hit.normal.yzx)
	);
	
	color *= vec3(0.75 + 0.25 * mix(mix(ambient.z, ambient.w, ray_hit.uv.x), mix(ambient.y, ambient.x, ray_hit.uv.x), ray_hit.uv.y));
	
	var result = vec4(color, 1.0);

	output.display = result;

	var v_clip_coord = perframe_buffer.camera.projection 
		* perframe_buffer.camera.view 
		* vec4(ray_hit.destination, 1.0);
	var f_ndc_depth = v_clip_coord.z / v_clip_coord.w;
	output.depth = (f_ndc_depth + 1.0) * 0.4;

	return output;
}
