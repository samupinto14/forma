//! GPU side: uploads meshes and renders the viewport into an offscreen texture that egui shows as an image.

use crate::loader::{LoadResult, TextureImage};
use eframe::egui;
use eframe::egui_wgpu;
use eframe::egui::mutex::RwLock;
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

pub const SAMPLES: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub mvp: [[f32; 4]; 4],
    pub vp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub key_dir: [f32; 4],
    pub key: [f32; 4],
    pub params: [f32; 4],
    pub shape_col: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatUniform {
    base: [f32; 4],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub col: [f32; 4],
}

pub struct GpuChunk {
    pos: wgpu::Buffer,
    nrm: wgpu::Buffer,
    uv: Option<wgpu::Buffer>,
    idx: wgpu::Buffer,
    count: u32,
    material: usize,
}

pub struct GpuMaterial {
    bind: wgpu::BindGroup,
}

pub struct GpuMesh {
    chunks: Vec<GpuChunk>,
    materials: Vec<GpuMaterial>,
    pub geometry_bytes: u64,
    pub texture_bytes: u64,
}

struct Targets {
    size: [u32; 2],
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::TextureView,
    resolve_tex: wgpu::Texture,
}

pub struct Renderer {
    mesh_uv: wgpu::RenderPipeline,
    mesh_nouv: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    mat_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white: wgpu::TextureView,
    targets: Option<Targets>,
    pub texture_id: Option<egui::TextureId>,
    line_buf: Option<(wgpu::Buffer, u32)>,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
pub fn hex_linear(rgb: [u8; 3], a: f32) -> [f32; 4] {
    [srgb_to_linear(rgb[0] as f32 / 255.0), srgb_to_linear(rgb[1] as f32 / 255.0), srgb_to_linear(rgb[2] as f32 / 255.0), a]
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forma shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let glob_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let mat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh layout"),
            bind_group_layouts: &[&glob_layout, &mat_layout],
            push_constant_ranges: &[],
        });
        let line_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line layout"),
            bind_group_layouts: &[&glob_layout],
            push_constant_ranges: &[],
        });

        let pos_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }],
        };
        let nrm_layout = wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Snorm16x4, offset: 0, shader_location: 1 }],
        };
        let uv_layout = wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 2 }],
        };
        let depth_mesh = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let ms = wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false };
        let target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let mk_mesh = |entry: &str, buffers: &[wgpu::VertexBufferLayout]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&mesh_layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: entry, buffers, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &target,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: Some(depth_mesh.clone()),
                multisample: ms,
                multiview: None,
                cache: None,
            })
        };
        let mesh_uv = mk_mesh("vs_uv", &[pos_layout.clone(), nrm_layout.clone(), uv_layout]);
        let mesh_nouv = mk_mesh("vs_nouv", &[pos_layout, nrm_layout]);
        let line_target = [Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let lines = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&line_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_line",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 12, shader_location: 1 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_line", targets: &line_target, compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: ms,
            multiview: None,
            cache: None,
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &glob_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });
        let max_aniso = 16;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("texture sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: max_aniso,
            ..Default::default()
        });
        let white_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        Renderer {
            mesh_uv,
            mesh_nouv,
            lines,
            globals,
            globals_bind,
            mat_layout,
            sampler,
            white: white_tex.create_view(&Default::default()),
            targets: None,
            texture_id: None,
            line_buf: None,
        }
    }

    /// Moves the mesh to the graphics card. CPU copies are released piece by piece, and the queue is flushed
    /// regularly so the driver's staging copies never pile up into a second full copy of the model.
    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue, data: &mut LoadResult) -> GpuMesh {
        let flush = || {
            queue.submit(std::iter::empty());
            let _ = device.poll(wgpu::Maintain::Wait);
        };
        let mut geometry_bytes = 0u64;
        let mut texture_bytes = 0u64;
        let mut materials = Vec::with_capacity(data.materials.len());
        for m in data.materials.iter_mut() {
            let owned;
            let (view, has_tex): (&wgpu::TextureView, bool) = match &m.texture {
                Some(t) => {
                    texture_bytes += t.mips.iter().map(|l| l.len() as u64).sum::<u64>();
                    owned = upload_texture(device, queue, t, &m.name);
                    flush();
                    (&owned, true)
                }
                None => (&self.white, false),
            };
            let base = if has_tex { [1.0, 1.0, 1.0] } else { m.base };
            let u = MatUniform { base: [base[0], base[1], base[2], 1.0], flags: [if has_tex { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0] };
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material"),
                layout: &self.mat_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            materials.push(GpuMaterial { bind });
            if let Some(t) = m.texture.as_mut() {
                t.mips = Vec::new();
            }
        }
        let mut chunks = Vec::with_capacity(data.chunks.len());
        let mut pending = 0u64;
        for c in std::mem::take(&mut data.chunks) {
            let init = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
            };
            let pos = init("pos", bytemuck::cast_slice(&c.pos), wgpu::BufferUsages::VERTEX);
            let nrm = init("nrm", bytemuck::cast_slice(&c.nrm), wgpu::BufferUsages::VERTEX);
            let uv = c.uv.as_ref().map(|u| init("uv", bytemuck::cast_slice(u), wgpu::BufferUsages::VERTEX));
            let idx = init("idx", bytemuck::cast_slice(&c.idx), wgpu::BufferUsages::INDEX);
            geometry_bytes += (c.pos.len() * 12 + c.nrm.len() * 8 + c.uv.as_ref().map(|u| u.len() * 8).unwrap_or(0) + c.idx.len() * 4) as u64;
            pending += (c.pos.len() * 28 + c.idx.len() * 4) as u64;
            chunks.push(GpuChunk { pos, nrm, uv, idx, count: c.idx.len() as u32, material: c.material.min(materials.len().saturating_sub(1)) });
            drop(c);
            if pending > (192 << 20) {
                flush();
                pending = 0;
            }
        }
        flush();
        GpuMesh { chunks, materials, geometry_bytes, texture_bytes }
    }

    pub fn set_lines(&mut self, device: &wgpu::Device, verts: &[LineVertex]) {
        if verts.is_empty() {
            self.line_buf = None;
            return;
        }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lines"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.line_buf = Some((buf, verts.len() as u32));
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, egui_r: Option<&RwLock<egui_wgpu::Renderer>>, size: [u32; 2]) {
        let size = [size[0].max(1), size[1].max(1)];
        if self.targets.as_ref().map(|t| t.size == size).unwrap_or(false) {
            return;
        }
        let mk = |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let msaa = mk("msaa", COLOR_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = mk("depth", DEPTH_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let resolve = resolve_tex.create_view(&Default::default());
        if let Some(er) = egui_r {
            let mut r = er.write();
            match self.texture_id {
                Some(id) => r.update_egui_texture_from_wgpu_texture(device, &resolve, wgpu::FilterMode::Linear, id),
                None => self.texture_id = Some(r.register_native_texture(device, &resolve, wgpu::FilterMode::Linear)),
            }
        }
        self.targets = Some(Targets { size, msaa, depth, resolve, resolve_tex });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_r: Option<&RwLock<egui_wgpu::Renderer>>,
        size: [u32; 2],
        g: &Globals,
        clear: [f32; 4],
        mesh: Option<&GpuMesh>,
        show_lines: bool,
    ) {
        self.ensure_targets(device, egui_r, size);
        let t = self.targets.as_ref().unwrap();
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(g));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewport") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa,
                    resolve_target: Some(&t.resolve),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: clear[0] as f64, g: clear[1] as f64, b: clear[2] as f64, a: 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.globals_bind, &[]);
            if let Some(mesh) = mesh {
                for c in &mesh.chunks {
                    match &c.uv {
                        Some(uv) => {
                            pass.set_pipeline(&self.mesh_uv);
                            pass.set_vertex_buffer(2, uv.slice(..));
                        }
                        None => pass.set_pipeline(&self.mesh_nouv),
                    }
                    pass.set_bind_group(0, &self.globals_bind, &[]);
                    pass.set_bind_group(1, &mesh.materials[c.material].bind, &[]);
                    pass.set_vertex_buffer(0, c.pos.slice(..));
                    pass.set_vertex_buffer(1, c.nrm.slice(..));
                    pass.set_index_buffer(c.idx.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..c.count, 0, 0..1);
                }
            }
            if show_lines {
                if let Some((buf, n)) = &self.line_buf {
                    pass.set_pipeline(&self.lines);
                    pass.set_bind_group(0, &self.globals_bind, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..*n, 0..1);
                }
            }
        }
        queue.submit(Some(enc.finish()));
    }

    /// Read the last rendered frame back as tightly packed RGBA8 (used by the self-test).
    pub fn read_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<(u32, u32, Vec<u8>)> {
        let t = self.targets.as_ref()?;
        let (w, h) = (t.size[0], t.size[1]);
        let row = (w * 4 + 255) / 256 * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: &t.resolve_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let s = (y * row) as usize;
            out.extend_from_slice(&data[s..s + (w * 4) as usize]);
        }
        drop(data);
        buf.unmap();
        Some((w, h, out))
    }
}

fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, t: &TextureImage, label: &str) -> wgpu::TextureView {
    let max = device.limits().max_texture_dimension_2d;
    // skip mip levels that are bigger than the GPU allows
    let mut skip = 0usize;
    while skip + 1 < t.mips.len() && ((t.width >> skip).max(1) > max || (t.height >> skip).max(1) > max) {
        skip += 1;
    }
    let (w, h) = ((t.width >> skip).max(1), (t.height >> skip).max(1));
    let levels = &t.mips[skip..];
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, data) in levels.iter().enumerate() {
        let (lw, lh) = ((w >> i).max(1), (h >> i).max(1));
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: i as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * lw), rows_per_image: Some(lh) },
            wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
        );
    }
    tex.create_view(&Default::default())
}
