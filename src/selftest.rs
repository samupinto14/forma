//! `forma.exe --selftest <file.obj> <out prefix>`: loads a model without opening a window, renders it in
//! several viewing modes to PNG files and appends load statistics to `<out prefix>_report.txt`.
//! Used by the build on GitHub to check that loading and drawing really work.

use crate::loader::{self, Progress};
use crate::render::{hex_linear, Globals, LineVertex, Renderer};
use eframe::wgpu;
use glam::{Mat4, Vec3};
use std::io::Write;
use std::path::Path;

pub fn run(obj: &str, out: &str) -> Result<(), String> {
    let mut report = String::new();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
    let all = instance.enumerate_adapters(wgpu::Backends::all());
    for a in &all {
        let i = a.get_info();
        report += &format!("found adapter: {} ({:?}, {:?})\n", i.name, i.backend, i.device_type);
    }
    // prefer a real GPU, otherwise take whatever exists (a software renderer on build machines)
    let pick = all.iter().position(|a| matches!(a.get_info().device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu))
        .or_else(|| all.iter().position(|a| a.get_info().backend != wgpu::Backend::Gl))
        .or(if all.is_empty() { None } else { Some(0) });
    let adapter = match pick {
        Some(i) => all.into_iter().nth(i).unwrap(),
        None => {
            let _ = std::fs::write(format!("{out}_report.txt"), &report);
            return Err("no graphics adapter".into());
        }
    };
    let info = adapter.get_info();
    report += &format!("adapter: {} ({:?}, {:?})\n", info.name, info.backend, info.device_type);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("selftest"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("device: {e}"))?;

    let progress = Progress::default();
    let mut data = loader::load(Path::new(obj), &progress, loader::TexLimit { budget: std::env::var("FORMA_TEX_BUDGET").ok().and_then(|v| v.parse().ok()), ..Default::default() }, 1_500_000)?;
    let st = &data.stats;
    report += &format!(
        "file: {} ({} bytes)\nload: {:.2} s\ntriangles: {}\nvertices: {}\ngpu vertices: {}\npieces: {}\nuvs: {} normals: {}\ndegenerate: {} invalid: {}\npieces without uv: {}\nmaterials: {}\n",
        data.name, st.file_bytes, st.seconds, st.tris, st.verts, st.gpu_verts, st.pieces, st.uvs, st.normals, st.degenerate, st.invalid, st.pieces_without_uv, data.materials.len()
    );
    for m in &data.materials {
        report += &format!(
            "  material '{}' kd={:?} texture={:?} {}\n",
            m.name,
            m.base,
            m.texture_file,
            match (&m.texture, &m.texture_problem) {
                (Some(t), _) => format!("{}x{} {} mips", t.width, t.height, t.mips.len()),
                (None, Some(p)) => p.clone(),
                _ => String::new(),
            }
        );
    }

    report += &format!("up axis: {:?}\nlight version: {} triangles\n", data.up, data.proxy.iter().map(|c| c.idx.len() / 3).sum::<usize>());
    let t0 = std::time::Instant::now();
    let mut r = Renderer::new(&device, &queue);
    let mesh = r.upload(&device, &queue, &mut data);
    report += &format!("upload: {:.2} s, geometry {} bytes, textures {} bytes\n", t0.elapsed().as_secs_f32(), mesh.geometry_bytes, mesh.texture_bytes);

    // frame the model the same way the app does for its starting view
    let (mn, mx) = (Vec3::from(data.min), Vec3::from(data.max));
    let center = (mn + mx) * 0.5;
    let size = mx - mn;
    let radius = (size.length() * 0.5).max(1e-6);
    let fov = 35f32.to_radians();
    let dist = radius / (fov * 0.5).sin() * 1.15;
    let eye = center + Vec3::new(0.75, 0.5, 1.0).normalize() * dist;
    let view = Mat4::look_at_rh(eye, center, Vec3::Y);
    let (w, h) = (960u32, 600u32);
    let proj = Mat4::perspective_rh(fov, w as f32 / h as f32, dist * 0.002, dist + radius * 12.0);
    let vp = proj * view;

    let cell = {
        let raw = size.x.max(size.z).max(1e-6) * 3.0 / 20.0;
        let p10 = 10f32.powf(raw.log10().floor());
        let f = raw / p10;
        (if f <= 1.0 { 1.0 } else if f <= 2.0 { 2.0 } else if f <= 5.0 { 5.0 } else { 10.0 }) * p10
    };
    let half = cell * 10.0;
    let mut lines = Vec::new();
    let gc = hex_linear([0x8a, 0x8f, 0x96], 0.4);
    for i in -10..=10 {
        let o = i as f32 * cell;
        lines.push(LineVertex { pos: [center.x - half, mn.y, center.z + o], col: gc });
        lines.push(LineVertex { pos: [center.x + half, mn.y, center.z + o], col: gc });
        lines.push(LineVertex { pos: [center.x + o, mn.y, center.z - half], col: gc });
        lines.push(LineVertex { pos: [center.x + o, mn.y, center.z + half], col: gc });
    }
    r.set_lines(&device, &lines);

    let (az, el) = (35f32.to_radians(), 40f32.to_radians());
    let kd = Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos());
    for (mode, name) in [(1.0, "grey"), (3.0, "textured"), (4.0, "checker"), (5.0, "normals"), (0.0, "shape")] {
        let g = Globals {
            mvp: vp.to_cols_array_2d(),
            vp: vp.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            key_dir: [kd.x, kd.y, kd.z, 0.0],
            key: [1.6, 0.6, 0.0, 0.0],
            params: [mode, 8.0, 0.0, 0.0],
            shape_col: hex_linear([0x11, 0x13, 0x15], 1.0),
        };
        let t1 = std::time::Instant::now();
        r.render(&device, &queue, None, [w, h], &g, hex_linear([0x3a, 0x3d, 0x42], 1.0), Some(&mesh), true, false);
        let (pw, ph, px) = r.read_pixels(&device, &queue).ok_or("no frame")?;
        report += &format!("render {name}: {:.0} ms\n", t1.elapsed().as_secs_f32() * 1000.0);
        image::save_buffer(format!("{out}_{name}.png"), &px, pw, ph, image::ExtendedColorType::Rgba8).map_err(|e| e.to_string())?;
    }

    {
        let g = Globals {
            mvp: vp.to_cols_array_2d(),
            vp: vp.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            key_dir: [kd.x, kd.y, kd.z, 0.0],
            key: [1.6, 0.6, 0.0, 0.0],
            params: [1.0, 8.0, 0.0, 0.0],
            shape_col: hex_linear([0x11, 0x13, 0x15], 1.0),
        };
        r.render(&device, &queue, None, [w, h], &g, hex_linear([0x3a, 0x3d, 0x42], 1.0), Some(&mesh), true, true);
        let (pw, ph, px) = r.read_pixels(&device, &queue).ok_or("no frame")?;
        image::save_buffer(format!("{out}_light.png"), &px, pw, ph, image::ExtendedColorType::Rgba8).map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{out}_report.txt")).map_err(|e| e.to_string())?;
    f.write_all(report.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}
