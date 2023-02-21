use bytemuck::{Pod, Zeroable};
use cgmath::{
    Deg, Euler, InnerSpace, Matrix4, PerspectiveFov, Quaternion, Rad, Rotation3, SquareMatrix,
    Vector2, Vector3, Vector4,
};
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
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};
use splines::{Interpolation, Key, Spline};

pub const REGION_SIZE: u32 = 128;
pub const FIELD_SIZE: u32 = 256;
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

struct World;

impl World {
    fn continentalness() -> splines::Spline<f32, f32> {
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
    
    fn erosion() -> splines::Spline<f32, f32> {
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
    
    fn pandv() -> splines::Spline<f32, f32> {
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

    fn update(delta_time: f32, perframe_data: &mut PerframeData) {
        const sens: f32 = 0.0002;

        let rot_z = Quaternion::from_angle_z(Rad(perframe_data.rot_z));

        perframe_data.camera.rotation = rot_z;

        let rot_x = Quaternion::from_angle_x(Rad(perframe_data.rot_x));

        perframe_data.camera.rotation = perframe_data.camera.rotation * rot_x;

        let mut dx = perframe_data.right as f32 - perframe_data.left as f32;
        let mut dy = perframe_data.forward as f32 - perframe_data.backward as f32;
        let mut dz = perframe_data.up as f32 - perframe_data.down as f32;

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

        perframe_data.camera.position += Vector4 {
            x: dx,
            y: dy,
            z: dz,
            w: 0.0,
        };

        perframe_data.camera.position[3] = 1.0;

        perframe_data.camera.transform = perframe_data.camera.rotation.into();
        perframe_data.camera.transform[3] = Vector4 {
            x: (REGION_SIZE as f32 / 2.0)
                + (f32::fract(perframe_data.camera.position.x)),
            y: (REGION_SIZE as f32 / 2.0)
                + (f32::fract(perframe_data.camera.position.y)),
            z: (REGION_SIZE as f32 / 2.0)
                + (f32::fract(perframe_data.camera.position.z)),
            w: 1.0,
        };

        perframe_data.camera.view = perframe_data.camera.transform.invert().unwrap();

        perframe_data.camera.projection = OPENGL_TO_WGPU_MATRIX
            * Matrix4::from(
                cgmath::PerspectiveFov::<f32> {
                    fovy: Deg(90.0).into(),
                    aspect: perframe_data.camera.resolution.x / perframe_data.camera.resolution.y,
                    near: 0.1,
                    far: 1000.0,
                }
                .to_perspective(),
            );

        perframe_data.camera.inv_projection = perframe_data.camera.projection.invert().unwrap();
    }
}

struct State {
    event_loop: EventLoop<()>,
    graphics: Graphics,
    world: World,
}

impl State {
    async fn new() -> Self {
        let event_loop = EventLoop::new();
        Self {
            graphics: Graphics::new(&event_loop).await,
            world: World,
            event_loop,
        }
    }

    fn run(mut self) {
        let Self { event_loop, .. } = self;

        let mut cursor_captured = false;
        let mut perframe_data = PerframeData {
            camera: Camera {
                transform: Matrix4::<f32>::identity(),
                view: Matrix4::<f32>::identity(),
                projection: Matrix4::<f32>::identity(),
                inv_projection: Matrix4::<f32>::identity(),
                position: Vector4::<f32>::new(0.0, 0.0, 0.0, 1.0),
                rotation: Quaternion::<f32>::new(0.0, 0.0, 0.0, 0.0),
                resolution: Vector2::<f32>::new(0.0, 0.0),
            },
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
        };

        let mut last_instant = time::Instant::now();

        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == self.graphics.window.id() => {
                let egui_response = self
                    .graphics
                    .egui
                    .state
                    .on_event(&self.graphics.egui.context, event);

                if (egui_response.consumed) {
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        self.graphics.window_size = *physical_size;
                        perframe_data.camera.resolution = Vector2 {
                            x: physical_size.width as f32,
                            y: physical_size.height as f32,
                        };
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        self.graphics.window_size = **new_inner_size;
                        perframe_data.camera.resolution = Vector2 {
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

                            perframe_data.rot_x -= SENS * y_diff as f32;
                            perframe_data.rot_x =
                                f32::clamp(perframe_data.rot_x, 0.0, 2.0 * std::f32::consts::PI);
                            perframe_data.rot_z -= SENS * x_diff as f32;
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
                                perframe_data.action1 = true as _;
                            }
                            Right => {
                                perframe_data.action2 = true as _;
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        let Some(key_code) = input.virtual_keycode else {
                    return;
                };

                        use winit::event::VirtualKeyCode::*;

                        match key_code {
                            W => {
                                perframe_data.forward =
                                    (input.state == winit::event::ElementState::Pressed) as _
                            }
                            A => {
                                perframe_data.left =
                                    (input.state == winit::event::ElementState::Pressed) as _
                            }
                            S => {
                                perframe_data.backward =
                                    (input.state == winit::event::ElementState::Pressed) as _
                            }
                            D => {
                                perframe_data.right =
                                    (input.state == winit::event::ElementState::Pressed) as _
                            }
                            Space => {
                                perframe_data.up =
                                    (input.state == winit::event::ElementState::Pressed) as _
                            }
                            LShift => {
                                perframe_data.down =
                                    (input.state == winit::event::ElementState::Pressed) as _
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
                World::update(delta_time, &mut perframe_data);

                self.graphics.window.set_title(&format!(
                    "Game | Frame time: {} ms",
                    (delta_time * 1000.0) as u32
                ));

                match self.graphics.render(perframe_data) {
                    Err(wgpu::SurfaceError::Outdated)
                    | Ok(_) if self.graphics.config.width != self.graphics.window_size.width
                        || self.graphics.config.height != self.graphics.window_size.height
                        => self.graphics.resize(),
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
    rot_x: f32,
    rot_z: f32,
}

unsafe impl bytemuck::Zeroable for PerframeData {}
unsafe impl bytemuck::Pod for PerframeData {}

struct Egui {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    context: egui::Context,
}

struct Graphics {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window_size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
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
    perlin_texture: wgpu::Texture,
    perlin_texture_view: wgpu::TextureView,
    noise_pipeline: wgpu::ComputePipeline,
    noise_group: wgpu::BindGroup,
    noise_storage_group: wgpu::BindGroup,
    egui: Egui,
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
        let continentalness = World::continentalness();
        let erosion = World::erosion();
        let pandv = World::pandv();

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

        let perlin_texture = device.create_texture(&wgpu::TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: wgpu::Extent3d {
                width: FIELD_SIZE,
                height: FIELD_SIZE,
                depth_or_array_layers: FIELD_SIZE,
            },
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            // Most images are stored using sRGB so we need to reflect that here.
            format: wgpu::TextureFormat::Rgba32Float,
            // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            label: Some("Perlin texture"),
            view_formats: &[],
        });

        let perlin_texture_view =
            perlin_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let noise_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                }],
                label: Some("bind_group_layout2"),
            });

        let noise_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &noise_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&perlin_texture_view),
            }],
            label: Some("bind_group"),
        });

        let noise_storage_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                }],
                label: Some("bind_group_layout2"),
            });

        let noise_storage_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &noise_storage_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&perlin_texture_view),
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
            bind_group_layouts: &[&bind_group_layout, &noise_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_layout2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &noise_storage_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let perframe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_perframe",
        });

        let noise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout2),
            module: &shader,
            entry_point: "cs_noise",
        });

        let create_chunk_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[],           // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
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

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            Some(wgpu::TextureFormat::Depth32Float),
            1,
        );

        let egui_state = egui_winit::State::new(&event_loop);

        let egui_context = egui::Context::default();

        let egui = Egui {
            renderer: egui_renderer,
            state: egui_state,
            context: egui_context,
        };

        Self {
            window,
            surface,
            device,
            queue,
            config,
            window_size,
            render_pipeline,
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
            noise_pipeline,
            noise_group,
            noise_storage_group,
            perlin_texture,
            perlin_texture_view,
            egui,
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

    fn ui(ctx: &egui::Context) {
        egui::Window::new("My Window")
            .default_open(true)
            .show(ctx, |ui| {
                use egui::plot::{Line, Plot, PlotPoints};
                let resolution = 1000;
                let spline = World::pandv();
                let sin: PlotPoints = (0..resolution)
                    .map(|i| {
                        let x = i as f64 * (1.0 / resolution as f64);
                        let y = spline.sample(x as f32).unwrap();
                        [x, y as f64]
                    })
                    .collect();
                let line = Line::new(sin);
                Plot::new("my_plot")
                    .view_aspect(2.0)
                    .show(ui, |plot_ui| plot_ui.line(line));
            });
    }

    fn render(&mut self, perframe_data: PerframeData) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.write_buffer(
            &self.perframe_buffer,
            0,
            bytemuck::cast_slice(&[perframe_data]),
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
            compute_pass.set_bind_group(1, &self.noise_group, &[]);

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
            compute_pass.set_bind_group(1, &self.noise_storage_group, &[]);

            compute_pass.set_pipeline(&self.noise_pipeline);
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
            compute_pass.set_bind_group(1, &self.noise_group, &[]);

            compute_pass.set_pipeline(&self.create_chunk_pipeline);
            compute_pass.dispatch_workgroups_indirect(&self.indirect_buffer, 0);
        }

        for i in 0..MAX_BATCHES {
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
        self.egui.context.request_repaint();

        let full_output = self
            .egui
            .context
            .run(self.egui.state.take_egui_input(&self.window), Self::ui);

        let clipped_primitives = self.egui.context.tessellate(full_output.shapes);

        let screen_descriptor = egui_wgpu::renderer::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: 1.0,
        };

        for (id, delta) in full_output.textures_delta.set {
            self.egui
                .renderer
                .update_texture(&self.device, &self.queue, id, &delta);
        }

        self.egui.renderer.update_buffers(
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
            render_pass.set_bind_group(1, &self.noise_group, &[]);

            render_pass.set_pipeline(&self.render_pipeline);

            for i in 0..MAX_BATCHES {
                render_pass.draw_indirect(
                    &self.indirect_buffer,
                    (2 * mem::size_of::<wgpu::util::DispatchIndirect>()) as u64
                        + (4 * i as usize * mem::size_of::<u32>()) as u64,
                );
            }

            //Draw the ui after the world
            self.egui
                .renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
