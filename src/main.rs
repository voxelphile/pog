use bytemuck::{Pod, Zeroable};
use cgmath::{
    Deg, Euler, InnerSpace, Matrix4, PerspectiveFov, Quaternion, Rad, Rotation3, SquareMatrix,
    Vector2, Vector3, Vector4,
};
use legion::*;
use noise::{NoiseFn, Seedable, Simplex};
use std::{
    future::Future,
    mem,
    num::{NonZeroU32, NonZeroU64},
    pin::Pin,
    task::{Context, Poll},
    time,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wgpu::util::DeviceExt;
use wgsl_preprocessor::ShaderBuilder;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

pub const REGION_SIZE: u32 = 64;
pub const VIEW_DISTANCE: f32 = 128.0;
pub const WORLD_SIZE: u32 = 300;
pub const CHUNK_SIZE: u32 = 4;
pub const SAMPLE_COUNT: u32 = 1000;
pub const MAX_BATCHES: u32 = 10;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let state = State::new().await;

    State::run(state);

    Ok(())
}

pub struct Time {
    delta_time: f32,
}

#[system]
fn ui(
    #[resource] egui_context: &mut EguiContext,
    #[resource] egui_io: &mut EguiIo,
) {
    egui_io.full_output = Some(egui_context.run(egui_io.window_input.take().unwrap(), |ctx| {
        egui::Window::new("Settings").show(ctx, |ui| {});
    }));
}

#[system]
fn camera(
    #[resource] time: &Time, 
    #[resource] camera: &mut Camera,
) {
    camera.position[3] = 1.0;

    camera.transform = camera.rotation.into();
    camera.transform[3] = Vector4 {
        x: (REGION_SIZE as f32 / 2.0) + (f32::fract(camera.position.x)),
        y: (REGION_SIZE as f32 / 2.0) + (f32::fract(camera.position.y)),
        z: (REGION_SIZE as f32 / 2.0) + (f32::fract(camera.position.z)),
        w: 1.0,
    };

    camera.view = camera.transform.invert().unwrap();

    camera.projection = OPENGL_TO_WGPU_MATRIX
        * Matrix4::from(
            cgmath::PerspectiveFov::<f32> {
                fovy: Deg(90.0).into(),
                aspect: camera.resolution.x / camera.resolution.y,
                near: 0.1,
                far: 1000.0,
            }
            .to_perspective(),
        );

    camera.inv_projection = camera.projection.invert().unwrap();
}
#[system]
fn input(
    #[resource] time: &Time, 
    #[resource] camera: &mut Camera,
    #[resource] input: &mut Input
) {
    const sens: f32 = 0.0002;

    let Time { delta_time } = time;

    let rot_z = Quaternion::from_angle_z(Rad(input.rot_z));

    camera.rotation = rot_z;

    let rot_x = Quaternion::from_angle_x(Rad(input.rot_x));

    camera.rotation = camera.rotation * rot_x;

    let mut dx = input.right as f32 - input.left as f32;
    let mut dy = input.forward as f32 - input.backward as f32;
    let mut dz = input.up as f32 - input.down as f32;

    let mut adjusted_movement = (rot_z
        * Vector3 {
            x: dx,
            y: dy,
            z: 0.0,
        });

    if adjusted_movement.dot(adjusted_movement) != 0.0 {
        adjusted_movement = adjusted_movement.normalize();
    }

    dx = adjusted_movement.x;
    dy = adjusted_movement.y;

    const SPEED: f32 = 20.0;

    dx *= SPEED * delta_time;
    dy *= SPEED * delta_time;
    dz *= SPEED * delta_time;

    camera.position += Vector4 {
        x: dx,
        y: dy,
        z: dz,
        w: 0.0,
    };
}

mod world_gen {
    use splines::*;

    pub fn continentalness() -> splines::Spline<f32, f32> {
        Spline::from_vec(vec![
            Key::new(0.0, 1.0, Interpolation::Cosine),
            Key::new(1.0 / 11.0, 0.1, Interpolation::Cosine),
            Key::new(3.5 / 11.0, 0.1, Interpolation::Cosine),
            Key::new(4.0 / 11.0, 0.4, Interpolation::Cosine),
            Key::new(5.1 / 11.0, 0.4, Interpolation::Cosine),
            Key::new(5.2 / 11.0, 0.8, Interpolation::Cosine),
            Key::new(5.5 / 11.0, 0.8, Interpolation::Cosine),
            Key::new(7.0 / 11.0, 0.9, Interpolation::Cosine),
            Key::new(1.0, 1.0, Interpolation::default()),
        ])
    }

    pub fn erosion() -> splines::Spline<f32, f32> {
        Spline::from_vec(vec![
            Key::new(0.0, 1.0, Interpolation::Cosine),
            Key::new(1.5 / 9.0, 5.5 / 8.0, Interpolation::Cosine),
            Key::new(3.0 / 9.0, 4.0 / 8.0, Interpolation::Cosine),
            Key::new(3.3 / 9.0, 4.5 / 8.0, Interpolation::Cosine),
            Key::new(4.5 / 9.0, 1.2 / 8.0, Interpolation::Cosine),
            Key::new(6.0 / 9.0, 1.0 / 8.0, Interpolation::Cosine),
            Key::new(7.0 / 9.0, 1.0 / 8.0, Interpolation::Cosine),
            Key::new(7.2 / 9.0, 3.0 / 8.0, Interpolation::Cosine),
            Key::new(7.8 / 9.0, 3.0 / 8.0, Interpolation::Cosine),
            Key::new(8.0 / 9.0, 1.0 / 8.0, Interpolation::Cosine),
            Key::new(1.0, 0.5, Interpolation::default()),
        ])
    }

    pub fn pandv() -> splines::Spline<f32, f32> {
        Spline::from_vec(vec![
            Key::new(0.0, 0.0, Interpolation::Cosine),
            Key::new(1.0 / 5.0, 1.0 / 7.0, Interpolation::Cosine),
            Key::new(2.2 / 5.0, 2.0 / 7.0, Interpolation::Cosine),
            Key::new(3.0 / 5.0, 2.0 / 7.0, Interpolation::Cosine),
            Key::new(3.8 / 5.0, 5.5 / 7.0, Interpolation::Cosine),
            Key::new(4.2 / 5.0, 6.5 / 7.0, Interpolation::Cosine),
            Key::new(1.0, 0.8, Interpolation::default()),
        ])
    }
}

struct State {
    event_loop: EventLoop<()>,
    graphics: Graphics,
    world: World,
    resources: Resources,
}

impl State {
    async fn new() -> Self {
        let event_loop = EventLoop::new();

        let mut resources = Resources::default();

        resources.insert(
        Camera {
                transform: Matrix4::<f32>::identity(),
                view: Matrix4::<f32>::identity(),
                projection: Matrix4::<f32>::identity(),
                inv_projection: Matrix4::<f32>::identity(),
                position: Vector4::<f32>::new(0.0, 0.0, 60.0, 1.0),
                rotation: Quaternion::<f32>::new(0.0, 0.0, 0.0, 0.0),
                resolution: Vector2::<f32>::new(0.0, 0.0),
        });

        resources.insert(Input {
            up: 0,
            down: 0,
            left: 0,
            right: 0,
            forward: 0,
            backward: 0,
            action1: 0,
            action2: 0,
            rot_x: 0.0,
            rot_z: 0.0,
        });

        let world = World::default();

        let graphics = Graphics::new(&event_loop).await;

        resources.insert(egui_wgpu::Renderer::new(
            &graphics.device,
            graphics.surface_format,
            Some(wgpu::TextureFormat::Depth32Float),
            1,
        ));

        resources.insert(egui_winit::State::new(&event_loop));

        resources.insert(egui::Context::default());

        resources.insert(EguiIo { 
            full_output: None,
            window_input: None,
        });

        Self {
            graphics,
            world,
            resources,
            event_loop,
        }
    }

    fn run(mut self) {
        let Self { event_loop, .. } = self;

        let mut cursor_captured = false;

        let mut last_instant = time::Instant::now();

        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == self.graphics.window.id() => {
                let mut egui_state = self.resources.get_mut::<EguiState>().unwrap();
                let mut egui_context = self.resources.get::<EguiContext>().unwrap();

                let egui_response = egui_state
                    .on_event(&egui_context, event);

                if (egui_response.consumed) {
                    return;
                }

                let mut camera = self
                    .resources
                    .get_mut::<Camera>()
                    .expect("Camera not added");
                
                let mut input = self
                    .resources
                    .get_mut::<Input>()
                    .expect("Input not added");

                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        self.graphics.window_size = *physical_size;
                        camera.resolution = Vector2 {
                            x: physical_size.width as f32,
                            y: physical_size.height as f32,
                        };
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        self.graphics.window_size = **new_inner_size;
                        camera.resolution = Vector2 {
                            x: new_inner_size.width as f32,
                            y: new_inner_size.height as f32,
                        };
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if cursor_captured {
                            let winit::dpi::PhysicalPosition { x, y } = position;

                            let winit::dpi::PhysicalSize { width, height } =
                                self.graphics.window.inner_size();

                            let x_diff = x - width as f64 / 2.0;
                            let y_diff = y - height as f64 / 2.0;

                            self.graphics.window.set_cursor_position(
                                winit::dpi::PhysicalPosition::new(
                                    width as i32 / 2,
                                    height as i32 / 2,
                                ),
                            );

                            const SENS: f32 = 0.0002;

                            input.rot_x -= SENS * y_diff as f32;
                            input.rot_x =
                                f32::clamp(input.rot_x, 0.0, 2.0 * std::f32::consts::PI);
                            input.rot_z -= SENS * x_diff as f32;
                        }
                    }
                    WindowEvent::MouseInput { button, .. } => {
                        use winit::event::MouseButton::*;

                        match button {
                            Left => {
                                cursor_captured = true;
                                self.graphics
                                    .window
                                    .set_cursor_icon(winit::window::CursorIcon::Crosshair);
                                self.graphics
                                    .window
                                    .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                    .expect("could not grab mouse cursor");
                                input.action1 = true as _;
                            }
                            Right => {
                                input.action2 = true as _;
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::KeyboardInput { input: key_input, .. } => {
                        let Some(key_code) = key_input.virtual_keycode else {
                    return;
                };

                        use winit::event::VirtualKeyCode::*;

                        match key_code {
                            W => {
                                input.forward =
                                    (key_input.state == winit::event::ElementState::Pressed) as _
                            }
                            A => {
                                input.left =
                                    (key_input.state == winit::event::ElementState::Pressed) as _
                            }
                            S => {
                                input.backward =
                                    (key_input.state == winit::event::ElementState::Pressed) as _
                            }
                            D => {
                                input.right =
                                    (key_input.state == winit::event::ElementState::Pressed) as _
                            }
                            Space => {
                                input.up =
                                    (key_input.state == winit::event::ElementState::Pressed) as _
                            }
                            LShift => {
                                input.down =
                                    (key_input.state == winit::event::ElementState::Pressed) as _
                            }
                            Escape => {
                                cursor_captured = false;
                                self.graphics
                                    .window
                                    .set_cursor_icon(winit::window::CursorIcon::Default);
                                self.graphics
                                    .window
                                    .set_cursor_grab(winit::window::CursorGrabMode::None)
                                    .expect("could not grab mouse cursor");
                            }
                            _ => {}
                        };
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == self.graphics.window.id() => {
                let current_instant = time::Instant::now();
                let delta_time = current_instant.duration_since(last_instant).as_secs_f32();
                last_instant = current_instant;

                self.resources.insert(Time { delta_time });

                {
                let mut egui_io = self.resources.get_mut::<EguiIo>().unwrap();
                let mut egui_state = self.resources.get_mut::<EguiState>().unwrap();

                egui_io.window_input =
                    Some(egui_state.take_egui_input(&self.graphics.window));
                }

                let mut schedule = Schedule::builder()
                    .add_system(input_system())
                    .add_system(camera_system())
                    .add_system(ui_system())
                    .build();

                schedule.execute(&mut self.world, &mut self.resources);

                self.graphics.window.set_title(&format!(
                    "Game | Frame time: {} ms",
                    (delta_time * 1000.0) as u32
                ));

                let camera = *self
                    .resources
                    .get::<Camera>()
                    .expect("Perframe data not added");
                
                let input = *self
                    .resources
                    .get::<Input>()
                    .expect("Perframe data not added");

                let mut egui_renderer = self.resources.get_mut::<EguiRenderer>().expect("Egui not addded");
                let mut egui_context = self.resources.get_mut::<EguiContext>().expect("Egui not addded");
                let mut egui_io = self.resources.get_mut::<EguiIo>().expect("Egui not addded");

                match self.graphics.render(FrameData {
                    perframe_data: PerframeData {
                        camera,
                        input,
                    },
                    egui_renderer: &mut egui_renderer,
                    egui_context: &mut egui_context,
                    egui_io: &mut egui_io,
                }) {
                    Err(wgpu::SurfaceError::Outdated) | Ok(_)
                        if self.graphics.config.width != self.graphics.window_size.width
                            || self.graphics.config.height != self.graphics.window_size.height =>
                    {
                        self.graphics.resize()
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    _ => {}
                }
            }
            Event::MainEventsCleared => {
                self.graphics.window.request_redraw();
            }
            _ => {}
        });
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Camera {
    transform: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
    inv_projection: Matrix4<f32>,
    position: Vector4<f32>,
    rotation: Quaternion<f32>,
    resolution: Vector2<f32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Input {
    up: u32,
    down: u32,
    left: u32,
    right: u32,
    forward: u32,
    backward: u32,
    action1: u32,
    action2: u32,
    rot_x: f32,
    rot_z: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PerframeData {
    camera: Camera,
    input: Input
}

unsafe impl bytemuck::Zeroable for PerframeData {}
unsafe impl bytemuck::Pod for PerframeData {}

struct FrameData<'a> {
    perframe_data: PerframeData,
    egui_renderer: &'a mut EguiRenderer,
    egui_context: &'a mut EguiContext,
    egui_io: &'a mut EguiIo,
}

type EguiState = egui_winit::State;
type EguiRenderer = egui_wgpu::Renderer;
type EguiContext = egui::Context;

struct EguiIo {
    full_output: Option<egui::FullOutput>,
    window_input: Option<egui::RawInput>,
}

struct Graphics {
    surface: wgpu::Surface,
    surface_format: wgpu::TextureFormat,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window_size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    front_render_pipeline: wgpu::RenderPipeline,
    back_render_pipeline: wgpu::RenderPipeline,
    perframe_pipeline: wgpu::ComputePipeline,
    create_chunk_pipeline: wgpu::ComputePipeline,
    batch_buffer: wgpu::Buffer,
    global_buffer: wgpu::Buffer,
    modify_indirect_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    perframe_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    depth_texture_sampler: wgpu::Sampler,
    world_texture: wgpu::Texture,
    world_texture_view: wgpu::TextureView,
    world_pipeline: wgpu::ComputePipeline,
    world_group: wgpu::BindGroup,
    world_storage_group: wgpu::BindGroup,
}

impl Graphics {
    async fn new(event_loop: &EventLoop<()>) -> Self {
        let window = WindowBuilder::new()
            .with_title("Game | Frame time: 0 ms")
            .build(&event_loop)
            .unwrap();

        let window_size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            dx12_shader_compiler: Default::default(),
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .filter(|adapter| adapter.is_surface_supported(&surface))
            .next()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Immediate, //surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let perframe_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Data Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size: 1_000_000,
            mapped_at_creation: false,
        });

        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Buffer"),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            size: 1_000_000,
            mapped_at_creation: false,
        });

        let modify_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modify Indirect Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: 1_000_000,
            mapped_at_creation: false,
        });

        let global_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Buffer"),
            usage: wgpu::BufferUsages::STORAGE,
            size: 1_000_000,
            mapped_at_creation: false,
        });

        let batch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: 134217728,
            mapped_at_creation: false,
        });

        let mut samples = vec![];
        let continentalness = world_gen::continentalness();
        let erosion = world_gen::erosion();
        let pandv = world_gen::pandv();

        for i in 0..SAMPLE_COUNT {
            let x = i as f64 * (1.0 / SAMPLE_COUNT as f64);
            let y = continentalness.sample(x as f32).unwrap() as f32;
            samples.push(y);
        }

        for i in 0..SAMPLE_COUNT {
            let x = i as f64 * (1.0 / SAMPLE_COUNT as f64);
            let y = erosion.sample(x as f32).unwrap() as f32;
            samples.push(y);
        }

        for i in 0..SAMPLE_COUNT {
            let x = i as f64 * (1.0 / SAMPLE_COUNT as f64);
            let y = pandv.sample(x as f32).unwrap() as f32;
            samples.push(y);
        }

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sample buffer"),
            usage: wgpu::BufferUsages::STORAGE,
            contents: bytemuck::cast_slice(&samples),
        });

        let world_texture = device.create_texture(&wgpu::TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: wgpu::Extent3d {
                width: WORLD_SIZE,
                height: WORLD_SIZE,
                depth_or_array_layers: WORLD_SIZE,
            },
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            // Most images are stored using sRGB so we need to reflect that here.
            format: wgpu::TextureFormat::R32Uint,
            // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            label: Some("Perlin texture"),
            view_formats: &[],
        });

        let world_texture_view = world_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let world_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                }],
                label: Some("bind_group_layout2"),
            });

        let world_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &world_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&world_texture_view),
            }],
            label: Some("bind_group"),
        });

        let world_storage_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                }],
                label: Some("bind_group_layout2"),
            });

        let world_storage_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &world_storage_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&world_texture_view),
            }],
            label: Some("bind_group"),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(134217728).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(1000 * 12).unwrap()),
                    },
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &perframe_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &modify_indirect_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &global_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &batch_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(134217728).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &sample_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(1000 * 12).unwrap()),
                    }),
                },
            ],
            label: Some("bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &world_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_layout2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &world_storage_group_layout],
            push_constant_ranges: &[],
        });

        let world_edit_shader = Self::shader_module(&device, "world_edit.wgsl");
        let general_shader = Self::shader_module(&device, "general.wgsl");

        let perframe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &general_shader,
            entry_point: "cs_perframe",
        });

        let world_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout2),
            module: &world_edit_shader,
            entry_point: "cs_world",
        });

        let create_chunk_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &general_shader,
                entry_point: "cs_create_chunks",
            });

        let depth_size = wgpu::Extent3d {
            // 2.
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let depth_desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: depth_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let depth_texture = device.create_texture(&depth_desc);

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            // 4.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual), // 5.
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        let front_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &general_shader,
                    entry_point: "vs_main", // 1.
                    buffers: &[],           // 2.
                },
                fragment: Some(wgpu::FragmentState {
                    // 3.
                    module: &general_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        // 4.
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw, // 2.
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less, // 1.
                    stencil: wgpu::StencilState::default(),     // 2.
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,                         // 2.
                    mask: !0,                         // 3.
                    alpha_to_coverage_enabled: false, // 4.
                },
                multiview: None, // 5.
            });

        let back_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &general_shader,
                entry_point: "vs_main", // 1.
                buffers: &[],           // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &general_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Front),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always, // 1.
                stencil: wgpu::StencilState::default(),       // 2.
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });

        Self {
            window,
            surface,
            surface_format,
            device,
            queue,
            config,
            window_size,
            front_render_pipeline,
            back_render_pipeline,
            perframe_pipeline,
            create_chunk_pipeline,
            perframe_buffer,
            indirect_buffer,
            modify_indirect_buffer,
            global_buffer,
            batch_buffer,
            bind_group,
            depth_texture,
            depth_texture_view,
            depth_texture_sampler,
            world_pipeline,
            world_group,
            world_storage_group,
            world_texture,
            world_texture_view,
        }
    }

    fn resize(&mut self) {
        if self.window_size.width > 0 && self.window_size.height > 0 {
            self.config.width = self.window_size.width;
            self.config.height = self.window_size.height;
            self.surface.configure(&self.device, &self.config);
            let depth_size = wgpu::Extent3d {
                // 2.
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            };
            let depth_desc = wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: depth_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Depth32Float],
            };
            self.depth_texture = self.device.create_texture(&depth_desc);

            self.depth_texture_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_texture_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                // 4.
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual), // 5.
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            });
        }
    }

    fn render(&mut self, mut frame_data: FrameData<'_>) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.write_buffer(
            &self.perframe_buffer,
            0,
            bytemuck::cast_slice(&[frame_data.perframe_data]),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.world_group, &[]);

            compute_pass.set_pipeline(&self.perframe_pipeline);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.modify_indirect_buffer,
            0,
            &self.indirect_buffer,
            0,
            2 * mem::size_of::<wgpu::util::DispatchIndirect>() as u64,
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.world_storage_group, &[]);

            compute_pass.set_pipeline(&self.world_pipeline);
            compute_pass.dispatch_workgroups_indirect(
                &self.indirect_buffer,
                mem::size_of::<wgpu::util::DispatchIndirect>() as u64,
            );
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.world_group, &[]);

            compute_pass.set_pipeline(&self.create_chunk_pipeline);
            compute_pass.dispatch_workgroups_indirect(&self.indirect_buffer, 0);
        }

        for i in 0..=MAX_BATCHES {
            encoder.copy_buffer_to_buffer(
                &self.batch_buffer,
                4 * mem::size_of::<u32>() as u64
                    + i as u64
                        * (4 * mem::size_of::<u32>() + 100_000 * 4 * mem::size_of::<f32>()) as u64,
                &self.indirect_buffer,
                (2 * mem::size_of::<wgpu::util::DispatchIndirect>()) as u64
                    + (i as usize * 4 * mem::size_of::<u32>()) as u64,
                4 * mem::size_of::<u32>() as u64,
            );
        }

        //Draw the ui. Note this is not where the ui is rendered.
        //the ui is rendered after the world
        frame_data.egui_context.request_repaint();

        let full_output = frame_data.egui_io.full_output.take().unwrap();

        let clipped_primitives = frame_data.egui_context.tessellate(full_output.shapes);

        let screen_descriptor = egui_wgpu::renderer::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: 1.0,
        };

        for (id, delta) in full_output.textures_delta.set {
            frame_data
                .egui_renderer
                .update_texture(&self.device, &self.queue, id, &delta);
        }

        frame_data.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.8,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &self.world_group, &[]);

            render_pass.set_pipeline(&self.front_render_pipeline);

            for i in 0..MAX_BATCHES {
                render_pass.draw_indirect(
                    &self.indirect_buffer,
                    (2 * mem::size_of::<wgpu::util::DispatchIndirect>()) as u64
                        + (i as usize * mem::size_of::<wgpu::util::DrawIndirect>()) as u64,
                );
            }

            render_pass.set_pipeline(&self.back_render_pipeline);

            render_pass.draw_indirect(
                &self.indirect_buffer,
                (2 * mem::size_of::<wgpu::util::DispatchIndirect>()) as u64
                    + (MAX_BATCHES as usize * mem::size_of::<wgpu::util::DrawIndirect>()) as u64,
            );

            //Draw the ui after the world
            frame_data.egui_renderer.render(
                &mut render_pass,
                &clipped_primitives,
                &screen_descriptor,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn shader_module(device: &wgpu::Device, s: &'static str) -> wgpu::ShaderModule {
        device.create_shader_module(ShaderBuilder::new(&format!("src/{}", s)).unwrap().build())
    }
}
