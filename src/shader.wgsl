@group(0) @binding(0)
var region_texture: texture_3d<u32>;

struct Indirect {
	draw_vertex_count: u32,
    	draw_instance_count: u32,
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
}

struct GlobalData {
	indirect: Indirect,
	camera: Camera,
	load: u32,
};

struct PerframeData {
    	fov: f32,
    	near: f32,
    	far: f32,
    	aspect_ratio: f32,
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
		data_buffer.indirect.build_x = u32(size.x / 8);
		data_buffer.indirect.build_y = u32(size.y / 8);
		data_buffer.indirect.build_z = u32(size.z / 8);
		data_buffer.indirect.setup_x = u32(size.x / 8);
		data_buffer.indirect.setup_y = u32(size.y / 8);
		data_buffer.indirect.setup_z = u32(size.z / 8);
	} else {
		data_buffer.indirect.build_x = u32(0);
		data_buffer.indirect.build_y = u32(0);
		data_buffer.indirect.build_z = u32(0);
		data_buffer.indirect.setup_x = u32(0);
		data_buffer.indirect.setup_y = u32(0);
		data_buffer.indirect.setup_z = u32(0);
	}

	data_buffer.load = u32(1);
	
	data_buffer.indirect.draw_vertex_count = u32(3);
	data_buffer.indirect.draw_instance_count = u32(1);
}

@compute
@workgroup_size(8)
fn cs_build() {

}


@compute
@workgroup_size(8)
fn cs_setup(@builtin(global_invocation_id) id: vec3<u32>) {
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}
