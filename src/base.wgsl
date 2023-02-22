const chunk_size = 4;
const region_size = 128;
const max_batches = 10;
const max_samples = 1000;
const view_distance = 128.0;

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

	return k;
}

fn fbm(position: vec3<f32>, seed: i32) -> f32 {
	var octaves = 10;
	var lacunarity = 2.0;
	var gain = 0.5;
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

