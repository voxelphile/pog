const chunk_size = 4;
const region_size = 256;

struct Camera {
	transform: mat4x4<f32>,
	view: mat4x4<f32>,
	projection: mat4x4<f32>,
	inv_projection: mat4x4<f32>,
	position: vec4<f32>,
	rotation: vec4<f32>,
	resolution: vec2<f32>,
}

struct PerframeData {
	camera: Camera,
	up: u32,
	down: u32,
	left: u32,
	right: u32,
	forward: u32,
	backward: u32,
	action1: u32,
	action2: u32,
	look_x: f32,
	look_y: f32,
};

@group(0) @binding(0) var<storage> perframe_buffer : PerframeData;

struct IndirectData {
	draw_vertex_count: u32,
    	draw_instance_count: atomic<u32>,
    	draw_base_vertex: u32,
    	draw_base_instance: u32,
	create_chunk_x: u32,
	create_chunk_y: u32,
	create_chunk_z: u32,
}

@group(0) @binding(1) var<storage, read_write> indirect_buffer : IndirectData;

struct GlobalData {
	load: i32,
}

@group(0) @binding(2) var<storage, read_write> global_buffer : GlobalData;

struct Chunk {
	position: vec4<i32>,
}

struct ChunkData {
	chunks: array<Chunk>,
};

@group(0) @binding(3) var<storage, read_write> chunk_buffer : ChunkData;

fn world_gen(position: vec3<f32>) -> u32 {
	if(position.z < 20.0) {
		return u32(1);
	}
	return u32(0);
}

@compute
@workgroup_size(1)
fn cs_perframe() {
	var size: vec3<i32> = vec3<i32>(region_size);
	
	if(global_buffer.load == 0 || true) {
		indirect_buffer.draw_vertex_count = u32(36);
		indirect_buffer.draw_instance_count = u32(0);
		indirect_buffer.draw_base_vertex = u32(0);
		indirect_buffer.draw_base_instance = u32(0);

		indirect_buffer.create_chunk_x = u32(size.x / chunk_size);
		indirect_buffer.create_chunk_y = u32(size.y / chunk_size);
		indirect_buffer.create_chunk_z = u32(size.z / chunk_size);
	} else {
		indirect_buffer.create_chunk_x = u32(0);
		indirect_buffer.create_chunk_y = u32(0);
		indirect_buffer.create_chunk_z = u32(0);
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
	var size: vec3<i32> = vec3<i32>(region_size / chunk_size);
	
	var empty = true;

	var voxel_pos = vec3<i32>(gid);

	var block_id = world_gen(
		vec3<f32>(voxel_pos)
		+ perframe_buffer.camera.position.xyz
	);

	if(block_id != u32(0)) {
		solid_blocks += 1;
	}

	workgroupBarrier();

	if(lid == u32(0) && solid_blocks != 0) {
		var instance = atomicAdd(&indirect_buffer.draw_instance_count, u32(1));
		chunk_buffer.chunks[instance].position = vec4(voxel_pos, 1); 
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
	
	var out: VertexOutput;

	out.chunk_normal = vec4(normals[j / u32(6)], 0.0);
	out.local_position = vec4(f32(chunk_size) * offsets[indices[j]], 1.0);

	out.chunk_position = vec4<f32>(f32(chunk_size) * vec3<f32>(chunk_buffer.chunks[i].position.xyz), 1.0);

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
	
	var block_id = world_gen(vec3<f32>((*state).position)
		+ perframe_buffer.camera.position.xyz
	);

	if(block_id == u32(0)) {
		(*state).id = RAY_STATE_VOXEL_FOUND;
		(*state).block_id = block_id;
		return true;
	}

/*
	var block_id = textureLoad(region_texture, (*state).position, 0).x;

*/

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

fn voxel_query(position: vec3<i32>) -> bool {
	var block_id = u32(0); //world_gen(position);

	if(block_id != u32(0)) {
		return true;
	}

	return false;
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

@fragment
fn fs_main(
	in: VertexOutput,
) -> @location(0) vec4<f32> {
	var ray: Ray;
	ray.origin = in.chunk_position.xyz + in.local_position.xyz + 1e-2 * vec3<f32>(in.chunk_normal.xyz);
	ray.offset = perframe_buffer.camera.position.xyz;
	ray.direction = normalize(ray.origin - vec3<f32>(f32(region_size) / 2.0));
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

	var result = vec4(1.0);

	if(ray_hit.block_id == u32(1)) {
		result *= vec4(1.0, 0.0, 0.0, 1.0);	
	}

/*
	var ambient = voxel_ao(
		ray_hit.back_step, 
		abs(ray_hit.normal.zxy), 
		abs(ray_hit.normal.yzx)
	);
	
	result *= vec4(vec3(0.75 + 0.25 * mix(mix(ambient.z, ambient.w, ray_hit.uv.x), mix(ambient.y, ambient.x, ray_hit.uv.x), ray_hit.uv.y)), 1.0);*/

	return vec4(ray_hit.destination % f32(chunk_size) / f32(chunk_size), 1.0);	
}
