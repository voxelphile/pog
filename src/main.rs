use std::{
    mem,
    num::NonZeroU64,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let state = State::new().await;

    State::run(state);

    Ok(())
}
/*
type Identifier = u32;

struct Chunk {}

impl Entity for Chunk {}

impl Spawnable for Chunk {}

trait Entity {}

trait Spawnable: Entity {}

struct Spawn<T: Spawnable>(T);

struct Spawner {}

impl Future for Spawner {
    type Output = Identifier;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

enum EntityState {
    Free,
    Spawn,
    Active(Box<dyn Entity>),
    Despawn,
}*/

struct State {
    event_loop: EventLoop<()>,
    graphics: Graphics,
}

impl State {
    async fn new() -> Self {
        let event_loop = EventLoop::new();
        Self {
            graphics: Graphics::new(&event_loop).await,
            //entities: vec![],
            event_loop,
        }
    }

    fn run(mut self) {
        let Self { event_loop, .. } = self;

        let mut cursor_captured = false;
        let mut perframe_data = PerframeData::default();

        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == self.graphics.window.id() => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    self.graphics.window_size = *physical_size;
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    self.graphics.window_size = **new_inner_size;
                }
                _ => {}
            },
             Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                window_id,
            } => {
                if cursor_captured {
                    let winit::dpi::PhysicalPosition { x, y } = position;

                    let winit::dpi::PhysicalSize { width, height } = self.graphics.window.inner_size();

                    let x_diff = x - width as f64 / 2.0;
                    let y_diff = y - height as f64 / 2.0;

                    self.graphics.window.set_cursor_position(winit::dpi::PhysicalPosition::new(
                        width as i32 / 2,
                        height as i32 / 2,
                    ));

                    perframe_data.look_x += x_diff as f32;
                    perframe_data.look_y += y_diff as f32;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { button, .. },
                window_id,
            } => {
                use winit::event::MouseButton::*;
                
                match button {
                    Left => {
                        cursor_captured = true;
                        self.graphics.window.set_cursor_icon(winit::window::CursorIcon::Crosshair);
                        self.graphics.window
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
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                window_id,
            } => {
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
                        perframe_data.up = (input.state == winit::event::ElementState::Pressed) as _
                    }
                    LShift => {
                        perframe_data.down =
                            (input.state == winit::event::ElementState::Pressed) as _
                    }
                    Escape => {
                        cursor_captured = false;
                        self.graphics.window.set_cursor_icon(winit::window::CursorIcon::Default);
                        self.graphics.window
                            .set_cursor_grab(winit::window::CursorGrabMode::None)
                            .expect("could not grab mouse cursor");
                    }
                    _ => {}
                };
            }
            Event::RedrawRequested(window_id) if window_id == self.graphics.window.id() => {
                match self.graphics.render(perframe_data) {
                    Ok(_) => {
                        perframe_data.look_x = 0.0;
                        perframe_data.look_y = 0.0;
                    },
                    Err(wgpu::SurfaceError::Outdated) => self.graphics.resize(),
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
#[derive(Default, Copy, Clone)]
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
}

unsafe impl bytemuck::Zeroable for PerframeData {}
unsafe impl bytemuck::Pod for PerframeData {}

struct Graphics {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window_size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    setup_pipeline: wgpu::ComputePipeline,
    perframe_pipeline: wgpu::ComputePipeline,
    build_pipeline: wgpu::ComputePipeline,
    data_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    perframe_buffer: wgpu::Buffer,
    region_texture: wgpu::Texture,
    region_texture_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl Graphics {
    async fn new(event_loop: &EventLoop<()>) -> Self {
        let window = WindowBuilder::new().build(&event_loop).unwrap();

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
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Buffer"),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            size: 1_000_000,
            mapped_at_creation: false,
        });

        let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Data Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            size: 1_000_000,
            mapped_at_creation: false,
        });
        
        let perframe_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Data Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size: 1_000_000,
            mapped_at_creation: false,
        });


        let region_texture = device.create_texture(
    &wgpu::TextureDescriptor {
        // All textures are stored as 3D, we represent our 2D texture
        // by setting depth to 1.
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 256,
        },
        mip_level_count: 1, // We'll talk about this a little later
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        // Most images are stored using sRGB so we need to reflect that here.
        format: wgpu::TextureFormat::R32Uint,
        // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
        // COPY_DST means that we want to copy data to this texture
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        label: Some("Region texture"),
        // This is the same as with the SurfaceConfig. It
        // specifies what texture formats can be used to
        // create TextureViews for this texture. The base
        // texture format (Rgba8UnormSrgb in this case) is
        // always supported. Note that using a different
        // texture format is not supported on the WebGL2
        // backend.
        view_formats: &[],
    }
    );

    let region_texture_view = region_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
            wgpu::BindGroupLayoutEntry {
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(1_000_000).unwrap()),
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
                    resource: wgpu::BindingResource::TextureView(&region_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &data_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &perframe_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(1_000_000).unwrap()),
                    }),
                },
            ],
            label: Some("bind_group"),
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        
        let perframe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_perframe",
        });
        
        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_build",
        });
        
        let setup_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_setup",
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
            depth_stencil: None, // 1.
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
            device,
            queue,
            config,
            window_size,
            render_pipeline,
            perframe_pipeline,
            setup_pipeline,
            build_pipeline,
            perframe_buffer,
            data_buffer,
            indirect_buffer,
            region_texture,
            region_texture_view,
            bind_group,
        }
    }

    fn resize(&mut self) {
        if self.window_size.width > 0 && self.window_size.height > 0 {
            self.config.width = self.window_size.width;
            self.config.height = self.window_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self, perframe_data: PerframeData) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.write_buffer(&self.perframe_buffer, 0, bytemuck::cast_slice(&[perframe_data]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass") });

            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            compute_pass.set_pipeline(&self.perframe_pipeline);
            compute_pass.dispatch_workgroups(1,1,1);

        }

        encoder.copy_buffer_to_buffer(
            &self.data_buffer, 
            0, 
            &self.indirect_buffer, 
            mem::size_of::<wgpu::util::DrawIndirect>() as u64,
            2 * mem::size_of::<wgpu::util::DispatchIndirect>() as u64
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass") });

            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            let mut offset = mem::size_of::<wgpu::util::DrawIndirect>() as u64;
            compute_pass.set_pipeline(&self.build_pipeline);
            compute_pass.dispatch_workgroups_indirect(&self.indirect_buffer, offset); 
            
            offset += mem::size_of::<wgpu::util::DispatchIndirect>() as u64;
            compute_pass.set_pipeline(&self.setup_pipeline);
            compute_pass.dispatch_workgroups_indirect(&self.indirect_buffer, offset);
        }

        encoder.copy_buffer_to_buffer(
            &self.data_buffer, 
            0, 
            &self.indirect_buffer, 
            0,
            mem::size_of::<wgpu::util::DrawIndirect>() as u64,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_indirect(&self.indirect_buffer, 0);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
