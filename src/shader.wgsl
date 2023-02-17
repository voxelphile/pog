@group(0) @binding(0)
var region_texture: texture_3d<u32>;

@group(0) @binding(0)
var region_storage: texture_storage_3d<r32uint, write>;

struct Indirect {
	draw_vertex_count: u32,
    	draw_instance_count: atomic<u32>,
    	draw_base_vertex: u32,
    	draw_base_instance: u32,
	build_x: u32,
	build_y: u32,
	build_z: u32,
	setup_x: u32,
	setup_y: u32,
	setup_z: u32,
}

struct Camera {
	transform: mat4x4<f32>,
	view: mat4x4<f32>,
	projection: mat4x4<f32>,
	inv_projection: mat4x4<f32>,
	position: vec4<f32>,
	rotation: vec4<f32>,
	aspect_ratio: f32,
}

struct GlobalData {
	indirect: Indirect,
	load: u32,
	instance_ids: array<vec3<u32>>,
};

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

@group(0) @binding(1) var<storage, read_write> data_buffer : GlobalData;
@group(0) @binding(2) var<storage> perframe_buffer : PerframeData;

@compute
@workgroup_size(1)
fn cs_perframe() {
	var size: vec3<i32> = textureDimensions(region_texture);

	if(data_buffer.load == u32(0)) {
		data_buffer.indirect.draw_vertex_count = u32(36);
		data_buffer.indirect.draw_instance_count = u32(0);
		data_buffer.indirect.build_x = u32(size.x / 4);
		data_buffer.indirect.build_y = u32(size.y / 4);
		data_buffer.indirect.build_z = u32(size.z / 4);
		data_buffer.indirect.setup_x = u32(size.x / 8)
			* u32(size.y / 8)
			* u32(size.z / 8);
		data_buffer.indirect.setup_y = u32(1);
		data_buffer.indirect.setup_z = u32(1);
	} else {
		data_buffer.indirect.build_x = u32(0);
		data_buffer.indirect.build_y = u32(0);
		data_buffer.indirect.build_z = u32(0);
		data_buffer.indirect.setup_x = u32(0);
		data_buffer.indirect.setup_y = u32(0);
		data_buffer.indirect.setup_z = u32(0);
	}

	data_buffer.load = u32(1);
	
}

@compute
@workgroup_size(4,4,4)
fn cs_build(@builtin(global_invocation_id) gid: vec3<u32>) {
		textureStore(region_storage, gid, vec4<u32>(u32(1)));
}


@compute
@workgroup_size(1,1,1)
fn cs_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
	var size: vec3<i32> = textureDimensions(region_texture) / 8;
	
	var empty = true;

	var id = gid.x;		
	
	var idx = id;
	var chunk_pos = vec3<u32>(u32(0));
	chunk_pos.z = idx / u32(size.x * size.y);
	idx -= chunk_pos.z * u32(size.x * size.y);
	chunk_pos.y = idx / u32(size.x);
	chunk_pos.x = idx % u32(size.x);

	for(var x = 0; x < 8 && empty; x++) {
		for(var y = 0; y < 8 && empty; y++) {
			for(var z = 0; z < 8 && empty; z++) {
				var voxel_pos = chunk_pos * u32(8) + vec3<u32>(u32(x),u32(y),u32(z));
				var block_id = textureLoad(region_texture, voxel_pos, 0).x;

				if(block_id != u32(0)) {
					empty = false;
					break;
				}
			}
		}
	}

	if(!empty) {
		var instance = atomicAdd(&data_buffer.indirect.draw_instance_count, u32(1));
		data_buffer.instance_ids[instance] = chunk_pos; 
	}
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
   	@builtin(instance_index) i: u32,
    	@builtin(vertex_index) j: u32,
) -> VertexOutput {
	var indices = array<i32, 36>(1, 0, 3, 1, 3, 2, 4, 5, 6, 4, 6, 7, 2, 3, 7, 2, 7, 6, 5, 4, 0, 5, 0, 1, 6, 5, 1, 6, 1, 2, 3, 0, 4, 3, 4, 7);

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

	var local_position = offsets[indices[j]];

	var chunk_position = vec3<f32>(data_buffer.instance_ids[i].xyz);

	var world_position = local_position + chunk_position;


	var out: VertexOutput;
    	out.clip_position = perframe_buffer.camera.projection * perframe_buffer.camera.view * vec4<f32>(world_position, 1.0);
    	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}
