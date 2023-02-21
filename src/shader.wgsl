const chunk_size = 4;
const region_size = 128;
const max_batches = 10;
const max_samples = 1000;

const BLOCK_ID_AIR = 1;
const BLOCK_ID_GRASS = 2;
const BLOCK_ID_DIRT = 3;
const BLOCK_ID_STONE = 4;

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
	create_chunk_x: u32,
	create_chunk_y: u32,
	create_chunk_z: u32,
	create_noise_x: u32,
	create_noise_y: u32,
	create_noise_z: u32,
}

@group(0) @binding(1) var<storage, read_write> indirect_buffer : IndirectData;

struct GlobalData {
	load: i32,
	dst: f32,
	floating_origin: vec3<i32>,
}

@group(0) @binding(2) var<storage, read_write> global_buffer : GlobalData;

struct BatchChunk {
	position: vec4<i32>,
}

struct Batch {
	batch_vertex_count: u32,
    	batch_instance_count: atomic<u32>,
    	batch_base_vertex: u32,
    	batch_base_instance: u32,
	batch_chunks: array<BatchChunk, 100000>,
}

struct BatchData {
	batch_count: atomic<u32>,
	batches: array<Batch>,
};

@group(0) @binding(3) var<storage, read_write> batch_buffer : BatchData;

struct SampleData {
	continentalness: array<f32, 1000>,
	erosion: array<f32, 1000>,
	pandv: array<f32, 1000>,
};

@group(0) @binding(4) var<storage> sample_buffer : SampleData;

@group(1) @binding(0) var perlin_texture: texture_3d<f32>;
@group(1) @binding(0) var perlin_storage: texture_storage_3d<rgba32float, write>;

fn hash1d(a: u32) -> u32 {
	var x = a;
	x += ( x << 10u );
    	x ^= ( x >>  6u );
    	x += ( x <<  3u );
    	x ^= ( x >> 11u );
    	x += ( x << 15u );
    	return x;
}

fn hash2d(v: vec2<u32>) -> u32 {
	return hash1d(v.x ^ hash1d(v.y));
}

fn hash3d(v: vec3<u32>) -> u32 {
	return hash1d(v.x ^ hash1d(v.y) ^ hash1d(v.z));
}

fn hash4d(v: vec4<u32>) -> u32 {
	return hash1d(v.x ^ hash1d(v.y) ^ hash1d(v.z) ^ hash1d(v.w));
}

fn float_construct(m: u32) -> f32 {
	var x = m;

	var ieee_mantissa = 0x007FFFFFu;
	var ieee_one = 0x3F800000u;

	x &= ieee_mantissa;
	x |= ieee_one;

	var f = bitcast<f32>(x);
	return f - 1.0;
}

fn random1d(x: f32) -> f32 {
	return float_construct(hash1d(bitcast<u32>(x)));
}

fn random2d(v: vec2<f32>) -> f32 {
	return float_construct(hash2d(vec2<u32>(
		bitcast<u32>(v.x),
		bitcast<u32>(v.y)
	)));
}

fn random3d(v: vec3<f32>) -> f32 {
	return float_construct(hash3d(vec3<u32>(
		bitcast<u32>(v.x),
		bitcast<u32>(v.y),
		bitcast<u32>(v.z)
	)));
}

fn random4d(v: vec4<f32>) -> f32 {
	return float_construct(hash4d(vec4<u32>(
		bitcast<u32>(v.x),
		bitcast<u32>(v.y),
		bitcast<u32>(v.z),
		bitcast<u32>(v.w),
	)));
}

fn random_gradient(position: vec3<i32>, seed: i32) -> vec3<f32> {
	var alpha = random4d(vec4<f32>(vec3<f32>(position.xyz), f32(seed)));
	var beta = random4d(vec4<f32>(vec3<f32>(position.xyz), f32(seed) + 1.0));

	return normalize(vec3<f32>(
		cos(alpha) * cos(beta),
		sin(beta),
		sin(alpha) * cos(beta)
	));
}

fn dot_grid_gradient(i: vec3<i32>, p: vec3<f32>, seed: i32) -> f32 {
	return dot(random_gradient(i, seed), p - vec3<f32>(i));
}

fn perlin(position: vec3<f32>, seed: i32) -> f32 {
	var m0 = vec3<i32>(floor(position));

	var m1 = m0 + 1;

	var s = position - vec3<f32>(m0);

	var n0: f32;
	var n1: f32;
	var ix0: f32;
	var ix1: f32;
	var jx0: f32;
	var jx1: f32;
	var k: f32;

	n0 = dot_grid_gradient(vec3<i32>(m0.x, m0.y, m0.z), position, seed);
	n1 = dot_grid_gradient(vec3<i32>(m1.x, m0.y, m0.z), position, seed);
	ix0 = mix(n0, n1, s.x);

	n0 = dot_grid_gradient(vec3<i32>(m0.x, m1.y, m0.z), position, seed);
	n1 = dot_grid_gradient(vec3<i32>(m1.x, m1.y, m0.z), position, seed);
	ix1 = mix(n0, n1, s.x);

	jx0 = mix(ix0, ix1, s.y); 
	
	n0 = dot_grid_gradient(vec3<i32>(m0.x, m0.y, m1.z), position, seed);
	n1 = dot_grid_gradient(vec3<i32>(m1.x, m0.y, m1.z), position, seed);
	ix0 = mix(n0, n1, s.x);

	n0 = dot_grid_gradient(vec3<i32>(m0.x, m1.y, m1.z), position, seed);
	n1 = dot_grid_gradient(vec3<i32>(m1.x, m1.y, m1.z), position, seed);
	ix1 = mix(n0, n1, s.x);

	jx1 = mix(ix0, ix1, s.y); 

	k = mix(jx0, jx1, s.z);

	var value = (k + 1.0) / 2.0;	

	return value;
}

fn fbm(position: vec3<f32>, seed: i32) -> f32 {
	var octaves = 10;
	var lacunarity = 2.0;
	var gain = 0.3;
	var amplitude = 1.0;
	var frequency = 0.05;
	var height = 0.0;

	for(var i = 0; i < octaves; i++) {
		height += amplitude * perlin(frequency * position, seed);
		frequency *= lacunarity;
		amplitude *= gain;
	}

	return height;
}

fn world_gen(position: vec3<i32>) -> u32 {
	var border = (textureDimensions(perlin_texture).x - region_size) / 2;
	var pos = position - global_buffer.floating_origin + border; 
	pos.z = 0;
	var perlin = textureLoad(perlin_texture, pos, 0).rgb;
	var csample = sample_buffer.continentalness[u32(f32(perlin.r) * f32(max_samples))];
	var esample = sample_buffer.erosion[u32(f32(perlin.g) * f32(max_samples))];
	var pandvsample = sample_buffer.pandv[u32(f32(perlin.b) * f32(max_samples))];
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
@workgroup_size(1)
fn cs_perframe() {
	var size: vec3<i32> = vec3<i32>(region_size);
	
	if(global_buffer.load == 0 || true) {
		indirect_buffer.create_chunk_x = u32(size.x / chunk_size);
		indirect_buffer.create_chunk_y = u32(size.y / chunk_size);
		indirect_buffer.create_chunk_z = u32(size.z / chunk_size);

		batch_buffer.batch_count = u32(0);
		for(var i = 0; i < max_batches; i++) {
			batch_buffer.batches[i].batch_vertex_count = u32(36);
			batch_buffer.batches[i].batch_instance_count = u32(0);
			batch_buffer.batches[i].batch_base_instance = u32(i * 100000);
		}
	} else {
		indirect_buffer.create_chunk_x = u32(0);
		indirect_buffer.create_chunk_y = u32(0);
		indirect_buffer.create_chunk_z = u32(0);
	}

	var dst = distance(vec3<f32>(global_buffer.floating_origin), perframe_buffer.camera.position.xyz);

	var border = (textureDimensions(perlin_texture).x - region_size) / 2;
	
	if(global_buffer.load == 0 || dst > f32(border)) {
		global_buffer.floating_origin = vec3<i32>(perframe_buffer.camera.position.xyz);
		var size: vec3<i32> = vec3<i32>(textureDimensions(perlin_texture).xyz);
		indirect_buffer.create_noise_x = u32(size.x / 4);
		indirect_buffer.create_noise_y = u32(size.y / 4);
		indirect_buffer.create_noise_z = u32(size.z / 4);
	} else {
		indirect_buffer.create_noise_x = u32(0);
		indirect_buffer.create_noise_y = u32(0);
		indirect_buffer.create_noise_z = u32(0);
	}

	global_buffer.load = 1;
}

@compute
@workgroup_size(4,4,4)
fn cs_noise(
	@builtin(global_invocation_id) gid: vec3<u32>,
) {
	var seed = 46029;

	var pos = vec3<f32>(gid) + vec3<f32>(global_buffer.floating_origin);

	var alpha = fbm(pos, seed);
	var beta = fbm(pos, seed + 1);
	var gamma = fbm(pos, seed + 2);
	var delta = fbm(pos, seed + 3);

	textureStore(perlin_storage, vec3<i32>(gid), vec4<f32>(
		alpha,
		beta,
		gamma,
		delta
	));
}

var<workgroup> solid_blocks: atomic<i32>;

@compute
@workgroup_size(4, 4, 4)
fn cs_create_chunks(
	@builtin(global_invocation_id) gid: vec3<u32>,
	@builtin(local_invocation_index) lid : u32
) {
	var voxel_pos = vec3<i32>(gid);

	var block_id = world_gen(
		voxel_pos
		+ vec3<i32>(perframe_buffer.camera.position.xyz)
	);

	if(block_id != u32(BLOCK_ID_AIR)) {
		var up_block = world_gen(
			voxel_pos
			+ vec3<i32>(perframe_buffer.camera.position.xyz)
			+ vec3<i32>(0, 0, 1)
		) == u32(BLOCK_ID_AIR);
		var down_block = world_gen(
			voxel_pos
			+ vec3<i32>(perframe_buffer.camera.position.xyz)
			+ vec3<i32>(0, 0, -1)
		) == u32(BLOCK_ID_AIR);
		var left_block = world_gen(
			voxel_pos
			+ vec3<i32>(perframe_buffer.camera.position.xyz)
			+ vec3<i32>(0, -1, 0)
		) == u32(BLOCK_ID_AIR);
		var right_block = world_gen(
			voxel_pos
			+ vec3<i32>(perframe_buffer.camera.position.xyz)
			+ vec3<i32>(0, 1, 0)
		) == u32(BLOCK_ID_AIR);
		var forward_block = world_gen(
			voxel_pos
			+ vec3<i32>(perframe_buffer.camera.position.xyz)
			+ vec3<i32>(1, 0, 0)
		) == u32(BLOCK_ID_AIR);
		var backward_block = world_gen(
			voxel_pos
			+ vec3<i32>(perframe_buffer.camera.position.xyz)
			+ vec3<i32>(-1, 0, 0)
		) == u32(BLOCK_ID_AIR);
		
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

		loop {
			batch_count = atomicLoad(&batch_buffer.batch_count);
			instance = atomicLoad(&batch_buffer.batches[batch_count].batch_instance_count);

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
	
	var block_id = world_gen(
		(*state).position
		+ vec3<i32>(perframe_buffer.camera.position.xyz)
	);

	if(block_id != u32(BLOCK_ID_AIR)) {
		(*state).id = RAY_STATE_VOXEL_FOUND;
		(*state).block_id = block_id;
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

fn voxel_query(position: vec3<i32>) -> bool {
	var block_id = world_gen(position);

	if(block_id != u32(BLOCK_ID_AIR)) {
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

struct FragmentOutput {
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
) -> FragmentOutput {
	var output: FragmentOutput;

	var origin = in.chunk_position.xyz + in.local_position.xyz + 1e-2 * vec3<f32>(in.chunk_normal.xyz);
	var offset = perframe_buffer.camera.position.xyz;

	var ray: Ray;
	ray.origin = origin;
	ray.offset = offset;
	ray.direction = normalize(ray.origin - vec3<f32>(floor(f32(region_size) / 2.0) + fract(ray.offset)));
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

	var border = (textureDimensions(perlin_texture).x - region_size) / 2;
	var pos = ray_hit.destination - vec3<f32>(global_buffer.floating_origin + border); 
	var noise_factor = map_range3d(
		textureLoad(perlin_texture, vec3<i32>(pos), 0).rgb,
		vec3<f32>(0.5),
		vec3<f32>(0.8),
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	).r;


	if(ray_hit.block_id == u32(BLOCK_ID_GRASS)) {
		color *= mix(vec3<f32>(170.0, 255.0, 21.0) / 256.0, vec3<f32>(34.0, 139.0, 34.0) / 256.0, noise_factor);
	}
	if(ray_hit.block_id == u32(BLOCK_ID_STONE)) {
		color *= mix(vec3<f32>(135.0) / 256.0, vec3<f32>(20.0) / 256.0, noise_factor);
	}

	if(ray_hit.block_id == u32(BLOCK_ID_DIRT)) {
		color *= mix(vec3<f32>(107.0, 84.0, 40.0) / 256.0, vec3<f32>(64.0, 41.0, 5.0) / 256.0, noise_factor);
	}

	var ambient = voxel_ao(
		ray_hit.back_step
			+ vec3<i32>(perframe_buffer.camera.position.xyz),
		abs(ray_hit.normal.zxy), 
		abs(ray_hit.normal.yzx)
	);
	
	color *= vec3(0.75 + 0.25 * mix(mix(ambient.z, ambient.w, ray_hit.uv.x), mix(ambient.y, ambient.x, ray_hit.uv.x), ray_hit.uv.y));
	
	var result = vec4(color, 1.0);

	output.display = result;

	return output;
}
