use crate::camera::Camera;
use crate::loader::{self, LoadResult, MeshStats, Progress};
use crate::render::{hex_linear, Globals, GpuMesh, LineVertex, Renderer};
use eframe::egui::{self, Color32, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use eframe::egui_wgpu::RenderState;
use glam::{Mat3, Mat4, Quat, Vec3};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------- look

const ACCENT: Color32 = Color32::from_rgb(0x7f, 0xae, 0xff);
const BG: Color32 = Color32::from_rgb(0x12, 0x14, 0x17);
const PANEL: Color32 = Color32::from_rgb(0x19, 0x1c, 0x20);
const RAISED: Color32 = Color32::from_rgb(0x22, 0x26, 0x2b);
const LINE: Color32 = Color32::from_rgb(0x2c, 0x31, 0x37);
const FG: Color32 = Color32::from_rgb(0xe6, 0xe8, 0xeb);
const MUTED: Color32 = Color32::from_rgb(0x8d, 0x94, 0x9c);
const OK: Color32 = Color32::from_rgb(0x5f, 0xc4, 0x8a);
const WARN: Color32 = Color32::from_rgb(0xe9, 0xa2, 0x3b);
const BAD: Color32 = Color32::from_rgb(0xef, 0x6a, 0x5e);

const ROT: f32 = 0.006;
const TIERS: [&str; 3] = ["High", "Mid", "Low"];

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum Mode {
    Shape,
    Grey,
    Unlit,
    Textured,
    Checker,
    Normals,
}
const MODES: [(Mode, &str); 6] = [
    (Mode::Shape, "Shape"),
    (Mode::Grey, "Grey shaded"),
    (Mode::Unlit, "Self-illuminated"),
    (Mode::Textured, "Textured"),
    (Mode::Checker, "Checker"),
    (Mode::Normals, "Normals"),
];

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum TexSize {
    Auto,
    Full,
    Half,
    Quarter,
    Off,
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum Orient {
    Flat,
    File,
    ZUp,
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    budgets: [[u64; 2]; 3],
    gpu_gb: u32,
    swap_xy: bool,
    orient: Orient,
    mode: Mode,
    grid: bool,
    axes: bool,
    key_az: f32,
    key_el: f32,
    key_int: f32,
    fill: f32,
    bg: usize,
    checker: f32,
    tex_size: TexSize,
}
impl Default for Settings {
    fn default() -> Self {
        Settings {
            budgets: [[35_000_000, 8192], [5_000_000, 8192], [500_000, 2048]],
            gpu_gb: 24,
            swap_xy: true,
            orient: Orient::Flat,
            mode: Mode::Grey,
            grid: true,
            axes: true,
            key_az: 35.0,
            key_el: 40.0,
            key_int: 1.6,
            fill: 0.6,
            bg: 0,
            checker: 8.0,
            tex_size: TexSize::Auto,
        }
    }
}
const BGS: [(&str, [u8; 3]); 3] = [("Mid grey", [0x3a, 0x3d, 0x42]), ("Dark", [0x14, 0x16, 0x18]), ("Light", [0xd9, 0xdc, 0xdf])];

struct TexInfo {
    name: String,
    w: u32,
    h: u32,
    ow: u32,
    bytes: u64,
    problem: Option<String>,
}

struct Version {
    n: u32,
    name: String,
    stats: MeshStats,
    gpu: GpuMesh,
    min: Vec3,
    max: Vec3,
    textures: Vec<TexInfo>,
    materials: usize,
    complete: bool,
    shown_tris: u64,
}

#[derive(Default)]
struct Tier {
    versions: Vec<Version>,
    cur: usize,
}

struct Job {
    tier: usize,
    rx: Receiver<Result<LoadResult, String>>,
    progress: Arc<Progress>,
    name: String,
}

#[derive(Clone, Copy)]
struct Framing {
    center: Vec3,
    radius: f32,
    size: Vec3,
    min: Vec3,
    cell: f32,
    half: f32,
}
impl Default for Framing {
    fn default() -> Self {
        Framing { center: Vec3::ZERO, radius: 1.0, size: Vec3::ONE, min: Vec3::ZERO, cell: 1.0, half: 10.0 }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Lvl {
    Ok,
    Warn,
    Bad,
}
struct Check {
    lvl: Lvl,
    label: &'static str,
    pill: String,
    detail: String,
}

pub struct FormaApp {
    s: Settings,
    tiers: [Tier; 3],
    active: usize,
    cam: Camera,
    framing: Framing,
    model: Mat4,
    renderer: Option<Renderer>,
    job: Option<Job>,
    toast: Option<(String, Instant, f32)>,
    dirty: bool,
    logo: Option<egui::TextureHandle>,
    gpu_error: Arc<AtomicBool>,
    last_size: [u32; 2],
    last_t: Instant,
    drag_t: Instant,
    rendered_key: Option<(usize, usize)>,
    gpu_kind: Option<eframe::wgpu::DeviceType>,
}

impl FormaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        style(&cc.egui_ctx);
        let s: Settings = cc.storage.and_then(|st| eframe::get_value(st, eframe::APP_KEY)).unwrap_or_default();
        let gpu_error = Arc::new(AtomicBool::new(false));
        let renderer = cc.wgpu_render_state.as_ref().map(|rs| {
            let flag = gpu_error.clone();
            let info = rs.adapter.get_info();
            crate::log::line(&format!("Forma {} on {} ({:?}, {:?}), memory {:?}", crate::VERSION, info.name, info.backend, info.device_type, crate::log::memory().map(|m| (crate::log::gb(m.0), crate::log::gb(m.2)))));
            rs.device.on_uncaptured_error(Box::new(move |e| {
                static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 25 {
                    crate::log::line(&format!("graphics error: {e}"));
                }
                flag.store(true, Ordering::Relaxed);
                // a failing allocation repeats the same error many times: keep the log readable
                static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                if COUNT.fetch_add(1, Ordering::Relaxed) == 25 {
                    crate::log::line("more graphics errors follow, not logged");
                }
            }));
            Renderer::new(&rs.device, &rs.queue)
        });
        let logo = image::load_from_memory(include_bytes!("../assets/logo.png")).ok().map(|img| {
            let img = img.resize(96, 96, image::imageops::FilterType::Lanczos3).to_rgba8();
            let (w, h) = img.dimensions();
            cc.egui_ctx.load_texture("logo", egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img), egui::TextureOptions::LINEAR)
        });
        FormaApp {
            s,
            tiers: Default::default(),
            active: 0,
            cam: Camera::default(),
            framing: Framing::default(),
            model: Mat4::IDENTITY,
            renderer,
            job: None,
            toast: None,
            dirty: true,
            logo,
            gpu_error,
            last_size: [0, 0],
            last_t: Instant::now(),
            drag_t: Instant::now(),
            rendered_key: None,
            gpu_kind: cc.wgpu_render_state.as_ref().map(|rs| rs.adapter.get_info().device_type),
        }
    }

    /// Texture detail for the next load. Auto keeps all textures together within a share of graphics memory
    /// that suits this card: integrated graphics borrow system memory, so they get a smaller share.
    fn tex_limit(&self) -> loader::TexLimit {
        let integrated = matches!(self.gpu_kind, Some(eframe::wgpu::DeviceType::IntegratedGpu) | Some(eframe::wgpu::DeviceType::Cpu));
        let ram = crate::log::memory().map(|m| m.2).unwrap_or(8_000_000_000);
        let auto = if integrated { ((ram as f64 * 0.12) as u64).clamp(500_000_000, 2_500_000_000) } else { 6_000_000_000 };
        match self.s.tex_size {
            TexSize::Auto => loader::TexLimit { budget: Some(auto), min_skip: 0, off: false },
            TexSize::Full => loader::TexLimit { budget: None, min_skip: 0, off: false },
            TexSize::Half => loader::TexLimit { budget: None, min_skip: 1, off: false },
            TexSize::Quarter => loader::TexLimit { budget: None, min_skip: 2, off: false },
            TexSize::Off => loader::TexLimit { budget: None, min_skip: 0, off: true },
        }
    }

    fn current(&self) -> Option<&Version> {
        let t = &self.tiers[self.active];
        t.versions.get(t.cur)
    }

    fn toast(&mut self, msg: impl Into<String>, secs: f32) {
        self.toast = Some((msg.into(), Instant::now(), secs));
    }

    // ------------------------------------------------------------ loading

    fn pick_and_load(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("OBJ mesh", &["obj", "OBJ"]).set_title("Open OBJ").pick_file() {
            self.start_load(p, self.active);
        }
    }

    fn start_load(&mut self, path: PathBuf, tier: usize) {
        if self.job.is_some() {
            self.toast("A file is still loading. Wait for it to finish.", 3.0);
            return;
        }
        let progress = Arc::new(Progress::default());
        let (tx, rx) = channel();
        let p2 = progress.clone();
        let path2 = path.clone();
        let limit = self.tex_limit();
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loader::load(&path2, &p2, limit)))
                .unwrap_or_else(|_| Err("The loader stopped unexpectedly. The file may be damaged or too large for this computer's memory.".into()));
            let _ = tx.send(r);
        });
        let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        self.job = Some(Job { tier, rx, progress, name });
    }

    fn poll_job(&mut self, rs: Option<&RenderState>) {
        let Some(job) = &self.job else { return };
        let Ok(res) = job.rx.try_recv() else { return };
        let job = self.job.take().unwrap();
        match res {
            Err(e) => {
                crate::log::line(&format!("load failed: {e}"));
                self.toast(format!("Could not load {}. {e}", job.name), 9.0)
            }
            Ok(mut data) => {
                let (Some(rs), Some(r)) = (rs, self.renderer.as_ref()) else { return };

                crate::log::line(&format!("upload start: {} pieces, memory {:?}", data.chunks.len(), crate::log::memory().map(|m| crate::log::gb(m.0))));
                let gpu = r.upload(&rs.device, &rs.queue, &mut data);
                let (gpu_complete, shown_tris) = (gpu.complete, gpu.shown_tris);
                let textures: Vec<TexInfo> = data
                    .materials
                    .iter()
                    .filter_map(|m| {
                        let name = m.texture_file.clone()?;
                        let name = name.rsplit('/').next().unwrap_or(&name).to_string();
                        Some(match &m.texture {
                            Some(t) => TexInfo { name, w: t.width, h: t.height, ow: t.orig_w.max(t.orig_h), bytes: t.bytes, problem: None },
                            None => TexInfo { name, w: 0, h: 0, ow: 0, bytes: 0, problem: m.texture_problem.clone() },
                        })
                    })
                    .collect();
                crate::log::line(&format!("upload done: geometry {} textures {}", gpu.geometry_bytes, gpu.texture_bytes));
                let tier = &mut self.tiers[job.tier];
                let n = tier.versions.len() as u32 + 1;
                let tris = data.stats.tris;
                let secs = data.stats.seconds;
                tier.versions.push(Version {
                    n,
                    name: data.name.clone(),
                    stats: data.stats.clone(),
                    gpu,
                    min: Vec3::from(data.min),
                    max: Vec3::from(data.max),
                    textures,
                    materials: data.materials.len().saturating_sub(1),
                    complete: gpu_complete,
                    shown_tris,
                });
                tier.cur = tier.versions.len() - 1;
                drop(data);
                if job.tier == self.active {
                    self.reframe(rs, false);
                }
                if gpu_complete {
                    self.toast(format!("{} v{n} loaded · {} triangles in {secs:.1} s", TIERS[job.tier], fmt(tris)), 5.0);
                } else {
                    self.toast(
                        format!("The graphics card ran out of memory. Showing {} of {} triangles. Close other apps, or set Texture size lower and open it again.", fmt(shown_tris), fmt(tris)),
                        12.0,
                    );
                }
                self.dirty = true;
            }
        }
    }

    // ------------------------------------------------------------ framing and camera

    fn update_model(&mut self) {
        let swap = if self.s.swap_xy {
            // swap X and Y as a 180° turn (never a mirror)
            Mat3::from_cols(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0))
        } else {
            Mat3::IDENTITY
        };
        let (mn, mx) = match self.current() {
            Some(v) => (v.min, v.max),
            None => (Vec3::splat(-1.0), Vec3::splat(1.0)),
        };
        let size_after = aabb(Mat4::from_mat3(swap), mn, mx).1 - aabb(Mat4::from_mat3(swap), mn, mx).0;
        let rot = match self.s.orient {
            Orient::File => Mat3::IDENTITY,
            Orient::ZUp => Mat3::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            Orient::Flat => {
                let s = size_after;
                if s.x < s.y && s.x <= s.z {
                    Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2)
                } else if s.z < s.y && s.z < s.x {
                    Mat3::from_rotation_x(-std::f32::consts::FRAC_PI_2)
                } else {
                    Mat3::IDENTITY
                }
            }
        };
        self.model = Mat4::from_mat3(rot * swap);
    }

    fn reframe(&mut self, rs: &RenderState, keep_angle: bool) {
        self.update_model();
        let (mn, mx) = match self.current() {
            Some(v) => (v.min, v.max),
            None => (Vec3::splat(-1.0), Vec3::splat(1.0)),
        };
        let (a, b) = aabb(self.model, mn, mx);
        let size = b - a;
        let center = (a + b) * 0.5;
        let radius = (size.length() * 0.5).max(1e-6);
        let raw = size.x.max(size.z).max(1e-6) * 3.0 / 20.0;
        let p10 = 10f32.powf(raw.log10().floor());
        let f = raw / p10;
        let cell = (if f <= 1.0 { 1.0 } else if f <= 2.0 { 2.0 } else if f <= 5.0 { 5.0 } else { 10.0 }) * p10;
        self.framing = Framing { center, radius, size, min: a, cell, half: cell * 10.0 };
        self.cam.pivot = center;
        if keep_angle {
            let d = self.fit_dist();
            self.cam.pos = center - self.cam.forward() * d;
            self.cam.spin = (0.0, 0.0);
        } else {
            self.set_view(Vec3::new(0.75, 0.5, 1.0));
        }
        self.build_lines(rs);
        self.dirty = true;
    }

    fn fit_dist(&self) -> f32 {
        self.framing.radius / (self.cam.fov_y * 0.5).sin() * 1.15
    }

    fn set_view(&mut self, dir: Vec3) {
        let d = self.fit_dist();
        let c = self.framing.center;
        self.cam.look_at(c + dir.normalize() * d, c);
        self.cam.pivot = c;
        self.dirty = true;
    }

    fn build_lines(&mut self, rs: &RenderState) {
        let f = self.framing;
        let y = f.min.y;
        let (cx, cz, h) = (f.center.x, f.center.z, f.half);
        let mut v = Vec::new();
        let grid = hex_linear([0x8a, 0x8f, 0x96], 0.40);
        let gridc = hex_linear([0x8a, 0x8f, 0x96], 0.55);
        if self.s.grid {
            for i in -10..=10 {
                if i == 0 && self.s.axes {
                    continue;
                }
                let o = i as f32 * f.cell;
                let c = if i == 0 { gridc } else { grid };
                v.push(LineVertex { pos: [cx - h, y, cz + o], col: c });
                v.push(LineVertex { pos: [cx + h, y, cz + o], col: c });
                v.push(LineVertex { pos: [cx + o, y, cz - h], col: c });
                v.push(LineVertex { pos: [cx + o, y, cz + h], col: c });
            }
        }
        if self.s.axes {
            let lift = h * 1e-4;
            let xr = hex_linear([0xe5, 0x53, 0x4b], 1.0);
            let zb = hex_linear([0x53, 0x9b, 0xf5], 1.0);
            let yg = hex_linear([0x57, 0xab, 0x5a], 1.0);
            let yl = (f.size.y * 1.2).max(f.cell);
            v.push(LineVertex { pos: [cx - h, y + lift, cz], col: xr });
            v.push(LineVertex { pos: [cx + h, y + lift, cz], col: xr });
            v.push(LineVertex { pos: [cx, y + lift, cz - h], col: zb });
            v.push(LineVertex { pos: [cx, y + lift, cz + h], col: zb });
            v.push(LineVertex { pos: [cx, y, cz], col: yg });
            v.push(LineVertex { pos: [cx, y + yl, cz], col: yg });
        }
        if let Some(r) = self.renderer.as_mut() {
            r.set_lines(&rs.device, &v);
        }
    }

    fn zoom_at(&mut self, ndc: Vec2, aspect: f32, scale: f32) {
        let r = self.framing.radius;
        let (o, d) = self.cam.ray(ndc.x, ndc.y, aspect);
        let fwd = self.cam.forward();
        let ahead = (self.framing.center - self.cam.pos).dot(fwd);
        let plane_p = if ahead > r * 0.05 { self.framing.center } else { self.cam.pos + fwd * r };
        let denom = d.dot(fwd);
        if denom.abs() < 1e-6 {
            return;
        }
        let t = (plane_p - o).dot(fwd) / denom;
        if t <= 0.0 {
            return;
        }
        let p = o + d * t;
        let mut s = scale;
        let near = self.cam.pos.distance(p);
        if s < 1.0 && near * s < r * 0.002 {
            s = (r * 0.002 / near).max(1.0);
        }
        let to_c = self.cam.pos.distance(self.framing.center);
        if s > 1.0 && to_c * s > r * 40.0 {
            s = (r * 40.0 / to_c).max(1.0);
        }
        if s != 1.0 {
            self.cam.pos = p + (self.cam.pos - p) * s;
            self.dirty = true;
        }
    }

    // ------------------------------------------------------------ checks

    fn checks(&self, tier: usize) -> Vec<Check> {
        let Some(v) = self.tiers[tier].versions.get(self.tiers[tier].cur) else { return vec![] };
        let b = self.s.budgets[tier];
        let st = &v.stats;
        let mut out = Vec::new();
        if b[0] > 0 {
            let r = st.tris as f64 / b[0] as f64;
            out.push(Check {
                lvl: if r > 1.0 { Lvl::Bad } else if r > 0.85 { Lvl::Warn } else { Lvl::Ok },
                label: "Triangles",
                pill: (if r > 1.0 { "Over budget" } else if r > 0.85 { "Near limit" } else { "Within budget" }).into(),
                detail: format!("{} of {} ({:.0}%)", fmt(st.tris), fmt(b[0]), r * 100.0),
            });
        }
        let loaded: Vec<&TexInfo> = v.textures.iter().filter(|t| t.problem.is_none()).collect();
        let missing: Vec<&TexInfo> = v.textures.iter().filter(|t| t.problem.is_some()).collect();
        if let Some(maxt) = loaded.iter().map(|t| t.ow).max() {
            let over = b[1] > 0 && maxt as u64 > b[1];
            out.push(Check {
                lvl: if over { Lvl::Bad } else { Lvl::Ok },
                label: "Largest texture",
                pill: (if over { "Too large" } else { "Within budget" }).into(),
                detail: format!("{maxt} px of {} px · {} map{}", b[1], loaded.len(), if loaded.len() == 1 { "" } else { "s" }),
            });
        } else if missing.is_empty() {
            out.push(Check { lvl: Lvl::Warn, label: "Textures", pill: "None".into(), detail: "Keep the .mtl and textures next to the .obj".into() });
        }
        if !missing.is_empty() {
            let names: Vec<String> = missing.iter().take(3).map(|t| t.name.clone()).collect();
            out.push(Check {
                lvl: Lvl::Warn,
                label: "Textures not shown",
                pill: format!("{} file{}", missing.len(), if missing.len() == 1 { "" } else { "s" }),
                detail: names.join(", ") + if missing.len() > 3 { " and more" } else { "" },
            });
        }
        let mem = v.gpu.geometry_bytes + v.gpu.texture_bytes;
        let rec = recommend_gb(mem);
        out.push(Check {
            lvl: if rec > self.s.gpu_gb { Lvl::Bad } else if rec * 2 > self.s.gpu_gb { Lvl::Warn } else { Lvl::Ok },
            label: "GPU memory",
            pill: if rec > self.s.gpu_gb { format!("Over {} GB", self.s.gpu_gb) } else { format!("Fits {} GB", self.s.gpu_gb) },
            detail: format!("≈ {} · suggest {rec} GB GPU", bytes(mem)),
        });
        out.push(Check {
            lvl: if st.degenerate > 0 { Lvl::Warn } else { Lvl::Ok },
            label: "Degenerate faces",
            pill: if st.degenerate > 0 { format!("{} found", fmt(st.degenerate)) } else { "None".into() },
            detail: "Zero-area triangles".into(),
        });
        if st.invalid > 0 {
            out.push(Check { lvl: Lvl::Warn, label: "Invalid faces", pill: format!("{} skipped", fmt(st.invalid)), detail: "Faces pointing to missing vertices".into() });
        }
        out.push(Check {
            lvl: if st.pieces_without_uv > 0 { Lvl::Bad } else { Lvl::Ok },
            label: "UVs",
            pill: (if st.pieces_without_uv > 0 { "Missing" } else { "Present" }).into(),
            detail: (if st.pieces_without_uv > 0 { "Part of the mesh has no UVs" } else { "UVs on the whole mesh" }).into(),
        });
        out
    }

    fn tier_level(&self, tier: usize) -> Option<Lvl> {
        let c = self.checks(tier);
        if c.is_empty() {
            return None;
        }
        Some(if c.iter().any(|c| c.lvl == Lvl::Bad) { Lvl::Bad } else if c.iter().any(|c| c.lvl == Lvl::Warn) { Lvl::Warn } else { Lvl::Ok })
    }
}

impl eframe::App for FormaApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.s);
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        let c = BG.to_normalized_gamma_f32();
        [c[0], c[1], c[2], 1.0]
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let rs = frame.wgpu_render_state().cloned();
        self.poll_job(rs.as_ref());
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let before = self.s.clone();
        let mut want_reframe: Option<bool> = None;

        // dropped files
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if let Some(p) = dropped.into_iter().find(|p| p.extension().map(|e| e.eq_ignore_ascii_case("obj")).unwrap_or(false)) {
            self.start_load(p, self.active);
        }

        // keyboard shortcuts
        let typing = ctx.wants_keyboard_input();
        let now = Instant::now();
        let dt = (now - self.last_t).as_secs_f32().min(0.05);
        self.last_t = now;
        if !typing {
            ctx.input(|i| {
                for (k, (m, _)) in [egui::Key::Num1, egui::Key::Num2, egui::Key::Num3, egui::Key::Num4, egui::Key::Num5, egui::Key::Num6].iter().zip(MODES.iter()) {
                    if i.key_pressed(*k) {
                        self.s.mode = *m;
                    }
                }
                if i.key_pressed(egui::Key::F) {
                    want_reframe = Some(true);
                }
                if i.key_pressed(egui::Key::H) {
                    want_reframe = Some(false);
                }
                let fast = i.modifiers.shift;
                let sp = self.framing.radius * if fast { 2.4 } else { 0.8 } * dt;
                let mut mv = Vec3::ZERO;
                let (fw, rt) = (self.cam.forward(), self.cam.right());
                if i.key_down(egui::Key::W) { mv += fw; }
                if i.key_down(egui::Key::S) { mv -= fw; }
                if i.key_down(egui::Key::D) { mv += rt; }
                if i.key_down(egui::Key::A) { mv -= rt; }
                if i.key_down(egui::Key::E) { mv += Vec3::Y; }
                if i.key_down(egui::Key::Q) { mv -= Vec3::Y; }
                if mv != Vec3::ZERO {
                    self.cam.pos += mv * sp;
                    self.cam.spin = (0.0, 0.0);
                    self.dirty = true;
                }
            });
            if ctx.input(|i| [egui::Key::W, egui::Key::A, egui::Key::S, egui::Key::D, egui::Key::Q, egui::Key::E].iter().any(|k| i.key_down(*k))) {
                ctx.request_repaint();
            }
        }

        // ---------------- top bar
        egui::TopBottomPanel::top("top").frame(egui::Frame::none().fill(BG).inner_margin(egui::Margin::symmetric(16.0, 10.0))).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(l) = &self.logo {
                    ui.add(egui::Image::new(egui::load::SizedTexture::new(l.id(), [30.0, 30.0])));
                }
                ui.add_space(4.0);
                wordmark(ui, 13.0, FG);
                ui.label(RichText::new(crate::VERSION).monospace().size(11.0).color(MUTED));
                ui.add_space(18.0);
                for t in 0..3 {
                    let lvl = self.tier_level(t);
                    let dot = match lvl { Some(Lvl::Ok) => OK, Some(Lvl::Warn) => WARN, Some(Lvl::Bad) => BAD, None => LINE };
                    let sel = self.active == t;
                    let txt = RichText::new(format!("     {}", TIERS[t])).size(13.5).strong().color(if sel { FG } else { MUTED });
                    let btn = egui::Button::new(txt).fill(if sel { RAISED } else { PANEL }).stroke(Stroke::new(1.0_f32, if sel { ACCENT } else { LINE }));
                    let r = ui.add(btn);
                    // recolour the dot
                    let p = Pos2::new(r.rect.left() + 16.0, r.rect.center().y);
                    ui.painter().circle_filled(p, 4.0, dot);
                    if r.clicked() && self.active != t {
                        self.active = t;
                        want_reframe = Some(false);
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let b = egui::Button::new(RichText::new(format!("Open OBJ into {}", TIERS[self.active])).strong().color(Color32::from_rgb(0x0b, 0x14, 0x26))).fill(ACCENT);
                    if ui.add(b).clicked() {
                        self.pick_and_load();
                    }
                });
            });
        });

        // ---------------- left panel
        egui::SidePanel::left("left").resizable(false).exact_width(240.0).frame(egui::Frame::none().fill(PANEL).inner_margin(14.0)).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                heading(ui, "VIEWING MODE");
                for (k, (m, name)) in MODES.iter().enumerate() {
                    let sel = self.s.mode == *m;
                    let r = ui.add_sized([ui.available_width(), 26.0], egui::Button::new(RichText::new(*name).color(FG)).fill(if sel { RAISED } else { PANEL }).stroke(Stroke::new(1.0_f32, if sel { ACCENT } else { LINE })));
                    ui.painter().text(Pos2::new(r.rect.right() - 10.0, r.rect.center().y), egui::Align2::RIGHT_CENTER, format!("{}", k + 1), egui::FontId::monospace(10.0), MUTED);
                    if r.clicked() {
                        self.s.mode = *m;
                    }
                }
                ui.add_space(6.0);
                egui::ComboBox::from_label("Texture size")
                    .selected_text(match self.s.tex_size { TexSize::Auto => "Auto", TexSize::Full => "Full", TexSize::Half => "Half", TexSize::Quarter => "Quarter", TexSize::Off => "Off" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.s.tex_size, TexSize::Auto, "Auto, fit this graphics card");
                        ui.selectable_value(&mut self.s.tex_size, TexSize::Full, "Full");
                        ui.selectable_value(&mut self.s.tex_size, TexSize::Half, "Half");
                        ui.selectable_value(&mut self.s.tex_size, TexSize::Quarter, "Quarter");
                        ui.selectable_value(&mut self.s.tex_size, TexSize::Off, "Off");
                    })
                    .response
                    .on_hover_text("Applies to the next file you open. Quality checks always use the full texture size.");
                ui.checkbox(&mut self.s.grid, "Grid");
                ui.checkbox(&mut self.s.axes, "Axis");
                if self.s.mode == Mode::Checker {
                    ui.add(egui::Slider::new(&mut self.s.checker, 1.0..=32.0).step_by(1.0).text("Checker repeat"));
                }
                ui.add_space(12.0);
                heading(ui, "NAVIGATION");
                ui.columns(4, |c| {
                    for (i, (n, d)) in [("Front", Vec3::Z), ("Side", Vec3::X), ("Back", Vec3::NEG_Z), ("Top", Vec3::new(0.0, 1.0, 0.0001))].iter().enumerate() {
                        if c[i].add_sized([c[i].available_width(), 24.0], egui::Button::new(*n)).clicked() {
                            self.set_view(*d);
                        }
                    }
                });
                ui.columns(2, |c| {
                    if c[0].add_sized([c[0].available_width(), 24.0], egui::Button::new("Centre  F")).clicked() {
                        want_reframe = Some(true);
                    }
                    if c[1].add_sized([c[1].available_width(), 24.0], egui::Button::new("Reset view  H")).clicked() {
                        want_reframe = Some(false);
                    }
                });
                egui::ComboBox::from_label("Orientation")
                    .selected_text(match self.s.orient { Orient::Flat => "Lay flat", Orient::File => "As in file", Orient::ZUp => "Z up" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.s.orient, Orient::Flat, "Lay flat");
                        ui.selectable_value(&mut self.s.orient, Orient::File, "As in file");
                        ui.selectable_value(&mut self.s.orient, Orient::ZUp, "Z up");
                    });
                ui.checkbox(&mut self.s.swap_xy, "Swap X and Y");
                ui.label(RichText::new("Drag to orbit around the model centre, right-drag to pan. W A S D to move, Q and E for down and up, hold Shift to go faster. Scroll moves forward and back, Ctrl + scroll zooms to the cursor.").size(11.5).color(MUTED));
                ui.add_space(12.0);
                heading(ui, "LIGHTING");
                ui.horizontal(|ui| {
                    for (n, p) in [("Studio", [35.0, 40.0, 1.6, 0.6]), ("Top", [0.0, 85.0, 1.8, 0.35]), ("Rim", [190.0, 25.0, 2.4, 0.25])] {
                        if ui.button(n).clicked() {
                            self.s.key_az = p[0];
                            self.s.key_el = p[1];
                            self.s.key_int = p[2];
                            self.s.fill = p[3];
                        }
                    }
                });
                ui.add(egui::Slider::new(&mut self.s.key_az, 0.0..=360.0).text("Key azimuth").suffix("°"));
                ui.add(egui::Slider::new(&mut self.s.key_el, 0.0..=90.0).text("Key elevation").suffix("°"));
                ui.add(egui::Slider::new(&mut self.s.key_int, 0.0..=4.0).text("Key intensity"));
                ui.add(egui::Slider::new(&mut self.s.fill, 0.0..=2.0).text("Fill"));
                egui::ComboBox::from_label("Background").selected_text(BGS[self.s.bg.min(2)].0).show_ui(ui, |ui| {
                    for (i, (n, _)) in BGS.iter().enumerate() {
                        ui.selectable_value(&mut self.s.bg, i, *n);
                    }
                });
            });
        });

        // ---------------- right panel
        egui::SidePanel::right("right").resizable(false).exact_width(310.0).frame(egui::Frame::none().fill(PANEL).inner_margin(14.0)).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                heading(ui, "QUALITY CHECK");
                let checks = self.checks(self.active);
                if checks.is_empty() {
                    ui.label(RichText::new("Open an OBJ to see its checks.").color(MUTED));
                }
                for c in &checks {
                    let col = match c.lvl { Lvl::Ok => OK, Lvl::Warn => WARN, Lvl::Bad => BAD };
                    egui::Frame::none().fill(RAISED).rounding(6.0).inner_margin(egui::Margin { left: 11.0, right: 9.0, top: 7.0, bottom: 7.0 }).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let r = ui.horizontal(|ui| {
                            ui.label(RichText::new(c.label).strong().color(FG));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(&c.pill).size(11.0).strong().color(col));
                            });
                        });
                        ui.label(RichText::new(&c.detail).monospace().size(11.0).color(MUTED));
                        let fr = ui.min_rect();
                        ui.painter().rect_filled(Rect::from_min_size(Pos2::new(fr.left() - 11.0, fr.top() - 7.0), Vec2::new(3.0, fr.height() + 14.0)), 1.5, col);
                        let _ = r;
                    });
                    ui.add_space(4.0);
                }
                ui.add_space(10.0);
                if let Some(v) = self.current() {
                    heading(ui, "MESH");
                    kv(ui, "Triangles", &fmt(v.stats.tris));
                    kv(ui, "Vertices", &fmt(v.stats.verts));
                    kv(ui, "Materials", &fmt(v.materials as u64));
                    let s = v.max - v.min;
                    kv(ui, "Bounds X × Y × Z", &format!("{} × {} × {}", num(s.x), num(s.y), num(s.z)));
                    kv(ui, "GPU geometry", &bytes(v.gpu.geometry_bytes));
                    kv(ui, "File size", &bytes(v.stats.file_bytes));
                    kv(ui, "Load time", &format!("{:.1} s", v.stats.seconds));
                    ui.add_space(10.0);
                    heading(ui, "TEXTURES");
                    if v.textures.is_empty() {
                        ui.label(RichText::new("No textures for this version.").color(MUTED));
                    }
                    // the tiles of one asset share a long prefix: show only the part that tells them apart
                    let names: Vec<&str> = v.textures.iter().map(|t| t.name.as_str()).collect();
                    let mut common = names.first().map(|n| n.len()).unwrap_or(0);
                    for n in &names {
                        common = common.min(names[0].bytes().zip(n.bytes()).take_while(|(a, b)| a == b).count());
                    }
                    let cut = if names.len() > 1 && common > 12 { common.saturating_sub(4) } else { 0 };
                    for t in &v.textures {
                        let short = if cut > 0 && t.name.is_char_boundary(cut) { format!("…{}", &t.name[cut..]) } else { t.name.clone() };
                        let detail = match &t.problem {
                            None if t.ow > t.w.max(t.h) => (format!("{} px, shown {}", t.ow, t.w.max(t.h)), MUTED),
                            None => (format!("{} px · {}", t.w.max(t.h), bytes(t.bytes)), MUTED),
                            Some(p) => (p.clone(), WARN),
                        };
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(detail.0).monospace().size(11.0).color(detail.1));
                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                    ui.add(egui::Label::new(RichText::new(short).size(12.0).color(FG)).truncate()).on_hover_text(&t.name);
                                });
                            });
                        });
                    }
                    ui.add_space(10.0);
                }
                heading(ui, "BUDGETS");
                egui::Grid::new("budgets").num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
                    ui.label("");
                    ui.label(RichText::new("Max triangles").size(11.0).color(MUTED));
                    ui.label(RichText::new("Max texture px").size(11.0).color(MUTED));
                    ui.end_row();
                    for t in 0..3 {
                        ui.label(TIERS[t]);
                        ui.add(egui::DragValue::new(&mut self.s.budgets[t][0]).speed(10_000.0).range(0..=u64::MAX));
                        ui.add(egui::DragValue::new(&mut self.s.budgets[t][1]).speed(64.0).range(0..=65536));
                        ui.end_row();
                    }
                });
                egui::ComboBox::from_label("Target GPU").selected_text(format!("{} GB", self.s.gpu_gb)).show_ui(ui, |ui| {
                    for g in [8u32, 12, 16, 24, 48] {
                        ui.selectable_value(&mut self.s.gpu_gb, g, format!("{g} GB"));
                    }
                });
                ui.add_space(10.0);
                heading(ui, &format!("VERSIONS · {}", TIERS[self.active].to_uppercase()));
                let mut pick: Option<usize> = None;
                let tier = &self.tiers[self.active];
                if tier.versions.is_empty() {
                    ui.label(RichText::new("Nothing loaded yet.").color(MUTED));
                }
                for (i, v) in tier.versions.iter().enumerate() {
                    let sel = i == tier.cur;
                    let prev = if i > 0 { tier.versions.get(i - 1) } else { None };
                    let delta = prev.map(|p| {
                        let d = v.stats.tris as i64 - p.stats.tris as i64;
                        format!(" · {}{} vs v{}", if d >= 0 { "+" } else { "−" }, fmt(d.unsigned_abs()), p.n)
                    });
                    let txt = format!("v{}  {}\n{} tris{}", v.n, v.name, fmt(v.stats.tris), delta.unwrap_or_default());
                    let b = egui::Button::new(RichText::new(txt).size(12.0).color(FG)).fill(RAISED).stroke(Stroke::new(1.0_f32, if sel { ACCENT } else { LINE }));
                    if ui.add_sized([ui.available_width(), 40.0], b).clicked() {
                        pick = Some(i);
                    }
                }
                if let Some(i) = pick {
                    self.tiers[self.active].cur = i;
                    want_reframe = Some(true);
                }
                if ui.button("Open new version").clicked() {
                    self.pick_and_load();
                }
            });
        });

        // ---------------- viewport
        let bgc = BGS[self.s.bg.min(2)].1;
        egui::CentralPanel::default().frame(egui::Frame::none().fill(Color32::from_rgb(bgc[0], bgc[1], bgc[2]))).show(ctx, |ui| {
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
            let ppp = ctx.pixels_per_point();
            let px = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
            if px != self.last_size {
                self.last_size = px;
                self.dirty = true;
            }
            let aspect = rect.width() / rect.height().max(1.0);

            // mouse
            let mods = ctx.input(|i| i.modifiers);
            if resp.dragged_by(egui::PointerButton::Primary) && !mods.shift {
                let d = resp.drag_delta();
                self.cam.orbit(d.x * ROT, d.y * ROT);
                self.cam.spin = (d.x * ROT, d.y * ROT);
                self.drag_t = Instant::now();
                self.dirty = true;
            } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (resp.dragged_by(egui::PointerButton::Primary) && mods.shift) {
                let d = resp.drag_delta();
                let wpp = self.cam.world_per_pixel(rect.height(), self.framing.radius);
                self.cam.pan(d.x, d.y, wpp);
                self.cam.spin = (0.0, 0.0);
                self.dirty = true;
            }
            if resp.drag_stopped() && self.drag_t.elapsed().as_millis() > 80 {
                self.cam.spin = (0.0, 0.0);
            }
            if resp.double_clicked() {
                want_reframe = Some(true);
            }
            if !resp.dragged() && (self.cam.spin.0.abs() > 1e-5 || self.cam.spin.1.abs() > 1e-5) {
                let (sx, sy) = self.cam.spin;
                self.cam.orbit(sx, sy);
                self.cam.spin = (sx * 0.9, sy * 0.9);
                self.dirty = true;
            }
            if resp.hovered() {
                let (scroll, zoom) = ctx.input(|i| (i.raw_scroll_delta, i.zoom_delta()));
                if mods.ctrl || mods.command {
                    let dy = if scroll.y != 0.0 { scroll.y } else { scroll.x };
                    if dy != 0.0 {
                        if let Some(hp) = resp.hover_pos() {
                            let ndc = Vec2::new((hp.x - rect.left()) / rect.width() * 2.0 - 1.0, -((hp.y - rect.top()) / rect.height() * 2.0 - 1.0));
                            self.zoom_at(ndc, aspect, (-dy * 0.004).exp().clamp(0.5, 2.0));
                        }
                    }
                } else if zoom != 1.0 {
                    if let Some(hp) = resp.hover_pos() {
                        let ndc = Vec2::new((hp.x - rect.left()) / rect.width() * 2.0 - 1.0, -((hp.y - rect.top()) / rect.height() * 2.0 - 1.0));
                        self.zoom_at(ndc, aspect, (1.0 / zoom).clamp(0.5, 2.0));
                    }
                } else if scroll != Vec2::ZERO {
                    let m = 0.6 * self.cam.world_per_pixel(rect.height(), self.framing.radius);
                    let mut fwd = self.cam.forward();
                    fwd.y = 0.0;
                    if fwd.length_squared() < 1e-6 {
                        fwd = self.cam.up();
                        fwd.y = 0.0;
                    }
                    let fwd = fwd.normalize_or_zero();
                    let side = fwd.cross(Vec3::Y).normalize_or_zero();
                    if mods.shift {
                        let d = if scroll.y != 0.0 { scroll.y } else { scroll.x };
                        self.cam.pos -= side * d * m;
                    } else {
                        self.cam.pos += fwd * scroll.y * m - side * scroll.x * m;
                    }
                    self.cam.spin = (0.0, 0.0);
                    self.dirty = true;
                }
            }

            // settings that change the scene setup
            if let Some(rs) = rs.as_ref() {
                if before.swap_xy != self.s.swap_xy || before.orient != self.s.orient {
                    want_reframe = Some(false);
                }
                if before.grid != self.s.grid || before.axes != self.s.axes {
                    self.build_lines(rs);
                }
                if let Some(keep) = want_reframe {
                    self.reframe(rs, keep);
                }
            }
            if before != self.s {
                self.dirty = true;
            }
            let key = (self.active, self.tiers[self.active].cur);
            if self.rendered_key != Some(key) {
                self.dirty = true;
            }

            // render the 3D view when something changed
            if self.dirty {
                if let (Some(rs), Some(r)) = (rs.as_ref(), self.renderer.as_mut()) {
                    let fr = self.framing;
                    let dc = self.cam.pos.distance(fr.center);
                    let near = (fr.radius * 2e-4).max((dc * 0.002).min(((dc - fr.radius * 1.5).max(0.0) * 0.5).max(dc * 0.002)));
                    let far = dc + fr.radius * 12.0;
                    let proj = self.cam.proj(aspect, near, far);
                    let view = self.cam.view();
                    let vp = proj * view;
                    let (az, el) = (self.s.key_az.to_radians(), self.s.key_el.to_radians());
                    let kd = Vec3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos());
                    let shape = if self.s.bg == 1 { hex_linear([0xe6, 0xe8, 0xeb], 1.0) } else { hex_linear([0x11, 0x13, 0x15], 1.0) };
                    let mode = MODES.iter().position(|(m, _)| *m == self.s.mode).unwrap_or(1) as f32;
                    let g = Globals {
                        mvp: (vp * self.model).to_cols_array_2d(),
                        vp: vp.to_cols_array_2d(),
                        model: self.model.to_cols_array_2d(),
                        key_dir: [kd.x, kd.y, kd.z, 0.0],
                        key: [self.s.key_int, self.s.fill, 0.0, 0.0],
                        params: [mode, self.s.checker, 0.0, 0.0],
                        shape_col: shape,
                    };
                    let clear = hex_linear(bgc, 1.0);
                    let t = &self.tiers[self.active];
                    let mesh = t.versions.get(t.cur).map(|v| &v.gpu);
                    r.render(&rs.device, &rs.queue, Some(&*rs.renderer), px, &g, clear, mesh, self.s.grid || self.s.axes);
                    self.rendered_key = Some(key);
                }
                self.dirty = false;
            }
            if let Some(id) = self.renderer.as_ref().and_then(|r| r.texture_id) {
                ui.painter().image(id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }

            // axis letters
            if self.s.axes && self.current().is_some() {
                let fr = self.framing;
                let vp = self.cam.proj(aspect, 0.01, 1e6) * self.cam.view();
                let o = Vec3::new(fr.center.x, fr.min.y, fr.center.z);
                let ext = fr.size.x.max(fr.size.z) * 0.65;
                for (txt, p, c) in [("X", o + Vec3::X * ext, BAD), ("Z", o + Vec3::Z * ext, Color32::from_rgb(0x53, 0x9b, 0xf5)), ("Y", o + Vec3::Y * (fr.size.y * 1.2).max(fr.cell), OK)] {
                    let h = vp * p.extend(1.0);
                    if h.w > 0.0 {
                        let sp = Pos2::new(rect.left() + (h.x / h.w * 0.5 + 0.5) * rect.width(), rect.top() + (0.5 - h.y / h.w * 0.5) * rect.height());
                        if rect.contains(sp) {
                            ui.painter().text(sp, egui::Align2::CENTER_CENTER, txt, egui::FontId::proportional(15.0), c);
                        }
                    }
                }
            }

            // overlay buttons
            let tools = Rect::from_min_size(rect.min + Vec2::new(10.0, 10.0), Vec2::new(220.0, 30.0));
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(tools), |ui| {
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(RichText::new("Reset view  H").strong().color(Color32::from_rgb(0xe8, 0xea, 0xed))).fill(Color32::from_rgba_unmultiplied(12, 14, 16, 215)).stroke(Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 40)))).clicked() {
                        if let Some(rs) = rs.as_ref() {
                            self.reframe(rs, false);
                        }
                    }
                    if ui.add(egui::Button::new(RichText::new("Centre  F").strong().color(Color32::from_rgb(0xe8, 0xea, 0xed))).fill(Color32::from_rgba_unmultiplied(12, 14, 16, 215)).stroke(Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 40)))).clicked() {
                        if let Some(rs) = rs.as_ref() {
                            self.reframe(rs, true);
                        }
                    }
                });
            });
            ui.painter().text(
                Pos2::new(rect.right() - 12.0, rect.top() + 18.0),
                egui::Align2::RIGHT_CENTER,
                "Drag orbit · WASD move · Ctrl + scroll zoom · F centre · H reset",
                egui::FontId::proportional(11.0),
                Color32::from_rgb(0xd6, 0xd9, 0xdd),
            );

            // HUD
            if let Some(v) = self.current() {
                let txt = format!("{}   v{} · {}   {} tris   {}{}", TIERS[self.active], v.n, v.name, fmt(v.stats.tris), bytes(v.gpu.geometry_bytes + v.gpu.texture_bytes), if v.complete { String::new() } else { format!("   PARTIAL: {} tris shown", fmt(v.shown_tris)) });
                let p = Pos2::new(rect.left() + 12.0, rect.bottom() - 14.0);
                let galley = ui.painter().layout_no_wrap(txt, egui::FontId::monospace(11.0), Color32::from_rgb(0xe8, 0xea, 0xed));
                let bg = Rect::from_min_size(p - Vec2::new(6.0, galley.size().y * 0.5 + 4.0), galley.size() + Vec2::new(12.0, 8.0));
                ui.painter().rect_filled(bg, 5.0, Color32::from_rgba_unmultiplied(12, 14, 16, 190));
                ui.painter().galley(Pos2::new(p.x, p.y - galley.size().y * 0.5), galley, Color32::WHITE);
            } else if self.job.is_none() {
                let c = rect.center();
                ui.painter().text(c - Vec2::new(0.0, 16.0), egui::Align2::CENTER_CENTER, format!("Open an OBJ into {}", TIERS[self.active]), egui::FontId::proportional(18.0), Color32::from_rgb(0xe8, 0xea, 0xed));
                ui.painter().text(c + Vec2::new(0.0, 12.0), egui::Align2::CENTER_CENTER, "Use the button at the top right, or drop a file here", egui::FontId::proportional(12.5), Color32::from_rgb(0xb4, 0xb9, 0xbf));
            }

            // loading
            if let Some(job) = &self.job {
                let done = job.progress.done.load(Ordering::Relaxed) as f32;
                let total = job.progress.total.load(Ordering::Relaxed).max(1) as f32;
                let stage = job.progress.stage.lock().map(|s| s.clone()).unwrap_or_default();
                let r = Rect::from_center_size(rect.center(), Vec2::new(420.0, 64.0));
                ui.painter().rect_filled(r, 8.0, Color32::from_rgba_unmultiplied(12, 14, 16, 235));
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(r.shrink(12.0)), |ui| {
                    ui.label(RichText::new(format!("{} · {stage}", job.name)).size(12.0).color(Color32::from_rgb(0xe8, 0xea, 0xed)));
                    ui.add(egui::ProgressBar::new((done / total).clamp(0.0, 1.0)).desired_height(6.0).fill(ACCENT));
                });
            }

            // graphics trouble
            if self.gpu_error.load(Ordering::Relaxed) {
                self.toast("The graphics card reported an error, probably out of memory. Close other heavy apps or load a lighter version.", 8.0);
                self.gpu_error.store(false, Ordering::Relaxed);
            }

            // toast
            let live = self.toast.as_ref().map(|(m, t, secs)| (m.clone(), t.elapsed().as_secs_f32() < *secs));
            match live {
                Some((msg, true)) => {
                    let galley = ui.painter().layout(msg, egui::FontId::proportional(12.5), Color32::WHITE, 520.0);
                    let r = Rect::from_center_size(Pos2::new(rect.center().x, rect.top() + 64.0), galley.size() + Vec2::new(24.0, 14.0));
                    ui.painter().rect_filled(r, 6.0, Color32::from_rgba_unmultiplied(12, 14, 16, 230));
                    ui.painter().galley(r.min + Vec2::new(12.0, 7.0), galley, Color32::WHITE);
                    ctx.request_repaint_after(std::time::Duration::from_millis(250));
                }
                Some((_, false)) => self.toast = None,
                None => {}
            }
        });

        if self.cam.spin.0.abs() > 1e-5 || self.cam.spin.1.abs() > 1e-5 || self.dirty {
            ctx.request_repaint();
        }
    }
}

// ---------------------------------------------------------------- helpers

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = Color32::from_rgb(0x15, 0x17, 0x1a);
    v.faint_bg_color = RAISED;
    v.selection.bg_fill = ACCENT.linear_multiply(0.45);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.widgets.inactive.weak_bg_fill = RAISED;
    v.widgets.inactive.bg_fill = RAISED;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, LINE);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x2a, 0x2f, 0x35);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, FG);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, FG);
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.rounding = egui::Rounding::same(6.0);
    }
    v.override_text_color = Some(FG);
    ctx.set_visuals(v);
    ctx.style_mut(|s| {
        s.spacing.item_spacing = Vec2::new(6.0, 6.0);
        s.spacing.button_padding = Vec2::new(10.0, 5.0);
        s.spacing.slider_width = 92.0;
    });
}

fn heading(ui: &mut egui::Ui, t: &str) {
    ui.label(RichText::new(t).size(10.5).strong().color(MUTED));
    ui.add_space(2.0);
}

fn kv(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(k).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(v).monospace().size(11.5).color(FG));
        });
    });
}

/// The FORMA wordmark: thin, widely spaced capitals with an A that has no crossbar.
fn wordmark(ui: &mut egui::Ui, height: f32, col: Color32) {
    let s = height / 108.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(900.0 * s, height), Sense::hover());
    let o = rect.min + Vec2::new(4.0 * s, 4.0 * s);
    let p = |x: f32, y: f32| Pos2::new(o.x + x * s, o.y + y * s);
    let st = Stroke::new((5.0 * s).max(1.0), col);
    let pa = ui.painter();
    let path = |pts: Vec<Pos2>| egui::Shape::line(pts, st);
    pa.add(path(vec![p(10.0, 0.0), p(10.0, 100.0)]));
    pa.add(path(vec![p(7.5, 2.5), p(78.0, 2.5)]));
    pa.add(path(vec![p(7.5, 50.0), p(72.0, 50.0)]));
    pa.add(egui::Shape::closed_line((0..64).map(|i| { let a = i as f32 / 64.0 * std::f32::consts::TAU; p(245.0 + 53.0 * a.cos(), 50.0 + 47.5 * a.sin()) }).collect(), st));
    let mut r = vec![p(390.0, 100.0), p(390.0, 2.5), p(440.0, 2.5)];
    for i in 0..=20 {
        let a = -std::f32::consts::FRAC_PI_2 + i as f32 / 20.0 * std::f32::consts::PI;
        r.push(p(440.0 + 26.25 * a.cos(), 28.75 + 26.25 * a.sin()));
    }
    r.push(p(390.0, 55.0));
    pa.add(path(r));
    pa.add(path(vec![p(434.0, 55.0), p(472.0, 100.0)]));
    pa.add(path(vec![p(575.0, 100.0), p(575.0, 2.0), p(632.0, 64.0), p(689.0, 2.0), p(689.0, 100.0)]));
    pa.add(path(vec![p(762.0, 100.0), p(826.0, 2.0), p(890.0, 100.0)]));
}

fn aabb(m: Mat4, mn: Vec3, mx: Vec3) -> (Vec3, Vec3) {
    let mut a = Vec3::splat(f32::INFINITY);
    let mut b = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..8 {
        let c = Vec3::new(if i & 1 == 0 { mn.x } else { mx.x }, if i & 2 == 0 { mn.y } else { mx.y }, if i & 4 == 0 { mn.z } else { mx.z });
        let w = m.transform_point3(c);
        a = a.min(w);
        b = b.max(w);
    }
    (a, b)
}

fn recommend_gb(bytes: u64) -> u32 {
    let need = (bytes as f64 * 1.5 + 2e9) / 1e9;
    [4u32, 6, 8, 12, 16, 24, 32, 48, 80].into_iter().find(|g| *g as f64 >= need).unwrap_or(96)
}

fn fmt(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{:.0} KB", b / 1e3)
    }
}

fn num(x: f32) -> String {
    let s = format!("{:.4}", x);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() || s == "-" { "0".into() } else { s }
}

#[allow(dead_code)]
fn _quat_unused(_q: Quat) {}
