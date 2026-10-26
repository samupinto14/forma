//! OBJ + MTL loading.
//!
//! The file is memory-mapped and split into byte ranges on line boundaries. Every pass runs over the
//! ranges in parallel:
//!   1. count `v`, `vt`, `vn` lines and collect material names per range,
//!   2. parse vertex data straight into shared arrays (each range owns a known slice),
//!   3. parse faces and build GPU-ready pieces (de-duplicated corners, at most `MAX_VERTS` vertices each).
//! Missing normals are then computed as smooth per-position normals.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const MAX_VERTS: usize = 1 << 20;

#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub stage: Mutex<String>,
}
impl Progress {
    fn stage(&self, s: &str) {
        crate::log::line(&format!("stage: {s}"));
        *self.stage.lock().unwrap() = s.to_string();
    }
}

pub struct Chunk {
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[i16; 4]>,
    pub uv: Option<Vec<[f32; 2]>>,
    pub idx: Vec<u32>,
    pub material: usize,
}

pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    /// RGBA8 levels, full size first.
    pub mips: Vec<Vec<u8>>,
}

pub struct Material {
    pub name: String,
    pub base: [f32; 3],
    pub texture_file: Option<String>,
    pub texture: Option<TextureImage>,
    /// Why the texture is not shown, if it isn't.
    pub texture_problem: Option<String>,
}

#[derive(Clone, Default)]
pub struct MeshStats {
    pub tris: u64,
    pub verts: u64,
    pub gpu_verts: u64,
    pub uvs: u64,
    pub normals: u64,
    pub degenerate: u64,
    pub invalid: u64,
    pub pieces_without_uv: u64,
    pub pieces: u64,
    pub file_bytes: u64,
    pub seconds: f32,
}

#[allow(dead_code)]
pub struct LoadResult {
    pub name: String,
    pub path: PathBuf,
    pub chunks: Vec<Chunk>,
    pub materials: Vec<Material>,
    pub stats: MeshStats,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

// ---------------------------------------------------------------- parsing helpers

#[inline]
fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && (b[*i] == b' ' || b[*i] == b'\t') {
        *i += 1;
    }
}

#[inline]
fn parse_f32(b: &[u8], i: &mut usize) -> f32 {
    skip_ws(b, i);
    let mut neg = false;
    if *i < b.len() && (b[*i] == b'-' || b[*i] == b'+') {
        neg = b[*i] == b'-';
        *i += 1;
    }
    let mut v: f64 = 0.0;
    while *i < b.len() && b[*i].is_ascii_digit() {
        v = v * 10.0 + (b[*i] - b'0') as f64;
        *i += 1;
    }
    if *i < b.len() && b[*i] == b'.' {
        *i += 1;
        let mut f = 0.1f64;
        while *i < b.len() && b[*i].is_ascii_digit() {
            v += (b[*i] - b'0') as f64 * f;
            f *= 0.1;
            *i += 1;
        }
    }
    if *i < b.len() && (b[*i] == b'e' || b[*i] == b'E') {
        *i += 1;
        let mut eneg = false;
        if *i < b.len() && (b[*i] == b'-' || b[*i] == b'+') {
            eneg = b[*i] == b'-';
            *i += 1;
        }
        let mut e: i32 = 0;
        while *i < b.len() && b[*i].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add((b[*i] - b'0') as i32);
            *i += 1;
        }
        v *= 10f64.powi(if eneg { -e } else { e });
    }
    (if neg { -v } else { v }) as f32
}

#[inline]
fn parse_int(b: &[u8], i: &mut usize) -> Option<i64> {
    let start = *i;
    let mut neg = false;
    if *i < b.len() && (b[*i] == b'-' || b[*i] == b'+') {
        neg = b[*i] == b'-';
        *i += 1;
    }
    let ds = *i;
    let mut v: i64 = 0;
    while *i < b.len() && b[*i].is_ascii_digit() {
        v = v * 10 + (b[*i] - b'0') as i64;
        *i += 1;
    }
    if *i == ds {
        *i = start;
        return None;
    }
    Some(if neg { -v } else { v })
}

/// Line kind from its first bytes (after leading whitespace).
#[derive(PartialEq, Clone, Copy)]
enum Kind {
    V,
    Vt,
    Vn,
    F,
    UseMtl,
    MtlLib,
    Other,
}

#[inline]
fn kind(l: &[u8]) -> (Kind, usize) {
    let mut i = 0;
    skip_ws(l, &mut i);
    let r = &l[i..];
    let sp = |c: u8| c == b' ' || c == b'\t';
    if r.len() > 1 && r[0] == b'v' {
        if sp(r[1]) {
            return (Kind::V, i + 1);
        }
        if r.len() > 2 && r[1] == b't' && sp(r[2]) {
            return (Kind::Vt, i + 2);
        }
        if r.len() > 2 && r[1] == b'n' && sp(r[2]) {
            return (Kind::Vn, i + 2);
        }
    } else if r.len() > 1 && r[0] == b'f' && sp(r[1]) {
        return (Kind::F, i + 1);
    } else if r.starts_with(b"usemtl") && r.len() > 6 && sp(r[6]) {
        return (Kind::UseMtl, i + 6);
    } else if r.starts_with(b"mtllib") && r.len() > 6 && sp(r[6]) {
        return (Kind::MtlLib, i + 6);
    }
    (Kind::Other, 0)
}

fn rest_str(l: &[u8], from: usize) -> String {
    String::from_utf8_lossy(&l[from..]).trim().trim_end_matches('\r').trim().to_string()
}

fn lines(b: &[u8]) -> impl Iterator<Item = &[u8]> {
    b.split(|&c| c == b'\n').map(|l| if l.last() == Some(&b'\r') { &l[..l.len() - 1] } else { l })
}

fn split_ranges(data: &[u8], parts: usize) -> Vec<(usize, usize)> {
    let len = data.len();
    let parts = parts.max(1);
    let step = (len / parts).max(1);
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    while start < len {
        let mut end = (start + step).min(len);
        while end < len && data[end - 1] != b'\n' {
            end += 1;
        }
        out.push((start, end));
        start = end;
    }
    if out.is_empty() {
        out.push((0, 0));
    }
    out
}

// ---------------------------------------------------------------- piece builder

struct Builder {
    map: FxHashMap<(u32, u32, u32), u32>,
    pos: Vec<[f32; 3]>,
    nrm: Vec<[i16; 4]>,
    uv: Vec<[f32; 2]>,
    src: Vec<u32>,
    need_n: Vec<bool>,
    any_uv: bool,
    missing_uv: bool,
    any_need_n: bool,
    idx: Vec<u32>,
    material: usize,
}

impl Builder {
    fn new(material: usize) -> Self {
        Builder {
            map: FxHashMap::default(),
            pos: Vec::new(),
            nrm: Vec::new(),
            uv: Vec::new(),
            src: Vec::new(),
            need_n: Vec::new(),
            any_uv: false,
            missing_uv: false,
            any_need_n: false,
            idx: Vec::new(),
            material,
        }
    }
}

pub struct RawPiece {
    chunk: Chunk,
    src: Vec<u32>,
    need_n: Vec<bool>,
    any_need_n: bool,
}

fn finish(b: &mut Builder, out: &mut Vec<RawPiece>) {
    if b.idx.is_empty() {
        return;
    }
    let material = b.material;
    let mut nb = std::mem::replace(b, Builder::new(material));
    // give back the spare capacity vectors grow into, which can be close to half of their size
    nb.pos.shrink_to_fit();
    nb.nrm.shrink_to_fit();
    nb.idx.shrink_to_fit();
    nb.map = FxHashMap::default();
    if nb.any_need_n {
        nb.src.shrink_to_fit();
        nb.need_n.shrink_to_fit();
    } else {
        nb.src = Vec::new();
        nb.need_n = Vec::new();
    }
    let uv = if nb.any_uv {
        nb.uv.shrink_to_fit();
        Some(nb.uv)
    } else {
        None
    };
    out.push(RawPiece {
        chunk: Chunk { pos: nb.pos, nrm: nb.nrm, uv, idx: nb.idx, material },
        src: nb.src,
        need_n: nb.need_n,
        any_need_n: nb.any_need_n,
    });
    // keep the "missing uv" signal for stats
    if nb.missing_uv && !nb.any_uv {
        // whole piece without UVs, counted by caller through `uv == None`
    }
}

#[inline]
fn pack_n(n: [f32; 3]) -> [i16; 4] {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let (x, y, z) = if l > 0.0 { (n[0] / l, n[1] / l, n[2] / l) } else { (0.0, 1.0, 0.0) };
    [(x * 32767.0) as i16, (y * 32767.0) as i16, (z * 32767.0) as i16, 0]
}

// ---------------------------------------------------------------- main entry

pub fn load(path: &Path, progress: &Progress) -> Result<LoadResult, String> {
    let t0 = std::time::Instant::now();
    let file = std::fs::File::open(path).map_err(|e| format!("Could not open the file: {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("Could not read the file: {e}"))?;
    let data: &[u8] = &mmap;
    let total = data.len() as u64;
    progress.total.store(total * 3, Ordering::Relaxed);
    let threads = rayon::current_num_threads();
    let ranges = split_ranges(data, (threads * 8).max((data.len() / (48 << 20)).max(1)));

    // ---- pass 1: counts and material names
    progress.stage("Scanning");
    struct Scan {
        v: usize,
        vt: usize,
        vn: usize,
        mats: Vec<String>,
        libs: Vec<String>,
        last_mat: Option<String>,
        faces: usize,
        corners: usize,
    }
    let scans: Vec<Scan> = ranges
        .par_iter()
        .map(|&(s, e)| {
            let mut sc = Scan { v: 0, vt: 0, vn: 0, mats: Vec::new(), libs: Vec::new(), last_mat: None, faces: 0, corners: 0 };
            for l in lines(&data[s..e]) {
                match kind(l) {
                    (Kind::V, _) => sc.v += 1,
                    (Kind::Vt, _) => sc.vt += 1,
                    (Kind::Vn, _) => sc.vn += 1,
                    (Kind::F, o) => {
                        sc.faces += 1;
                        let mut prev_ws = true;
                        for &c in &l[o..] {
                            let ws = c == b' ' || c == b'\t';
                            if prev_ws && !ws {
                                sc.corners += 1;
                            }
                            prev_ws = ws;
                        }
                    }
                    (Kind::UseMtl, o) => {
                        let n = rest_str(l, o);
                        if !sc.mats.contains(&n) {
                            sc.mats.push(n.clone());
                        }
                        sc.last_mat = Some(n);
                    }
                    (Kind::MtlLib, o) => sc.libs.push(rest_str(l, o)),
                    _ => {}
                }
            }
            progress.done.fetch_add((e - s) as u64, Ordering::Relaxed);
            sc
        })
        .collect();

    let nv: usize = scans.iter().map(|s| s.v).sum();
    let nvt: usize = scans.iter().map(|s| s.vt).sum();
    let nvn: usize = scans.iter().map(|s| s.vn).sum();
    if nv == 0 {
        return Err("No vertices found. Check that this is a mesh OBJ.".into());
    }
    // ---- will it fit in memory? Estimate the peak and compare with what Windows says is free.
    let faces: usize = scans.iter().map(|s| s.faces).sum();
    let corners: usize = scans.iter().map(|s| s.corners).sum();
    let est_tris = corners.saturating_sub(2 * faces) as u64;
    let est_verts = (nv.max(nvt).max(nvn) as f64 * 1.15) as u64;
    let globals = nv as u64 * 12 + nvt as u64 * 8 + nvn as u64 * 12;
    let pieces = est_verts * (12 + 8 + 8 + if nvn == 0 { 5 } else { 0 }) + est_tris * 12;
    let peak = globals + pieces + (64 << 20) * threads as u64;
    let mem = crate::log::memory();
    crate::log::line(&format!(
        "load {}: {} bytes, {nv} v, {nvt} vt, {nvn} vn, {faces} faces, ~{est_tris} tris, estimated peak {}, memory {:?}",
        path.display(), total, crate::log::gb(peak), mem.map(|(a, c, t)| (crate::log::gb(a), crate::log::gb(c), crate::log::gb(t)))
    ));
    if let Some((avail, commit, _total_ram)) = mem {
        if peak > commit {
            return Err(format!(
                "Not enough memory. This file needs about {} and only {} is free on this PC. Close other apps, or open it on a PC with more memory.",
                crate::log::gb(peak), crate::log::gb(commit)
            ));
        }
        if peak > avail {
            progress.stage("Low on memory, this may be slow");
        }
    }

    // material table: 0 is the default material
    let mut mat_names: Vec<String> = vec![String::new()];
    let mut mat_index: FxHashMap<String, usize> = FxHashMap::default();
    for sc in &scans {
        for m in &sc.mats {
            if !mat_index.contains_key(m) {
                mat_index.insert(m.clone(), mat_names.len());
                mat_names.push(m.clone());
            }
        }
    }
    let mut start_mat = Vec::with_capacity(scans.len());
    let mut cur = 0usize;
    for sc in &scans {
        start_mat.push(cur);
        if let Some(m) = &sc.last_mat {
            cur = mat_index[m];
        }
    }
    let mut libs: Vec<String> = Vec::new();
    for sc in &scans {
        for l in &sc.libs {
            if !libs.contains(l) {
                libs.push(l.clone());
            }
        }
    }

    // ---- pass 2: vertex data
    progress.stage("Reading vertices");
    let oom = |what: &str| format!("Not enough memory to read the {what}. Close other apps, or open the file on a PC with more memory.");
    let mut positions: Vec<[f32; 3]> = Vec::new();
    positions.try_reserve_exact(nv).map_err(|_| oom("vertices"))?;
    positions.resize(nv, [0.0; 3]);
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    texcoords.try_reserve_exact(nvt).map_err(|_| oom("UVs"))?;
    texcoords.resize(nvt, [0.0; 2]);
    let mut normals: Vec<[f32; 3]> = Vec::new();
    normals.try_reserve_exact(nvn).map_err(|_| oom("normals"))?;
    normals.resize(nvn, [0.0; 3]);
    {
        let mut ps: &mut [[f32; 3]] = &mut positions;
        let mut ts: &mut [[f32; 2]] = &mut texcoords;
        let mut ns: &mut [[f32; 3]] = &mut normals;
        let mut jobs = Vec::with_capacity(ranges.len());
        for (r, sc) in ranges.iter().zip(scans.iter()) {
            let (a, b) = std::mem::take(&mut ps).split_at_mut(sc.v);
            ps = b;
            let (c, d) = std::mem::take(&mut ts).split_at_mut(sc.vt);
            ts = d;
            let (e, f) = std::mem::take(&mut ns).split_at_mut(sc.vn);
            ns = f;
            jobs.push((*r, a, c, e));
        }
        jobs.into_par_iter().for_each(|((s, e), p, t, n)| {
            let (mut pi, mut ti, mut ni) = (0, 0, 0);
            for l in lines(&data[s..e]) {
                match kind(l) {
                    (Kind::V, o) => {
                        let mut i = o;
                        let x = parse_f32(l, &mut i);
                        let y = parse_f32(l, &mut i);
                        let z = parse_f32(l, &mut i);
                        if pi < p.len() {
                            p[pi] = [x, y, z];
                        }
                        pi += 1;
                    }
                    (Kind::Vt, o) => {
                        let mut i = o;
                        let u = parse_f32(l, &mut i);
                        let v = parse_f32(l, &mut i);
                        if ti < t.len() {
                            t[ti] = [u, v];
                        }
                        ti += 1;
                    }
                    (Kind::Vn, o) => {
                        let mut i = o;
                        let x = parse_f32(l, &mut i);
                        let y = parse_f32(l, &mut i);
                        let z = parse_f32(l, &mut i);
                        if ni < n.len() {
                            n[ni] = [x, y, z];
                        }
                        ni += 1;
                    }
                    _ => {}
                }
            }
            progress.done.fetch_add((e - s) as u64, Ordering::Relaxed);
        });
    }
    let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    let (bmin, bmax) = positions
        .par_chunks(1 << 16)
        .map(|c| {
            let (mut a, mut b) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for p in c {
                for k in 0..3 {
                    a[k] = a[k].min(p[k]);
                    b[k] = b[k].max(p[k]);
                }
            }
            (a, b)
        })
        .reduce(
            || ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
            |(a1, b1), (a2, b2)| {
                let mut a = a1;
                let mut b = b1;
                for k in 0..3 {
                    a[k] = a[k].min(a2[k]);
                    b[k] = b[k].max(b2[k]);
                }
                (a, b)
            },
        );
    for k in 0..3 {
        mn[k] = bmin[k];
        mx[k] = bmax[k];
    }
    let diag2 = (0..3).map(|k| (mx[k] - mn[k]).powi(2)).sum::<f32>().max(1e-30);
    let degen_eps = diag2 * diag2 * 1e-16;

    // ---- pass 3: faces into pieces
    progress.stage("Building mesh");
    let mut v_off = Vec::with_capacity(scans.len());
    let mut t_off = Vec::with_capacity(scans.len());
    let mut n_off = Vec::with_capacity(scans.len());
    let (mut a, mut b, mut c) = (0usize, 0usize, 0usize);
    for sc in &scans {
        v_off.push(a);
        t_off.push(b);
        n_off.push(c);
        a += sc.v;
        b += sc.vt;
        c += sc.vn;
    }
    struct FaceOut {
        pieces: Vec<RawPiece>,
        tris: u64,
        degenerate: u64,
        invalid: u64,
    }
    let outs: Vec<FaceOut> = (0..ranges.len())
        .into_par_iter()
        .map(|r| {
            let (s, e) = ranges[r];
            let (mut cv, mut ct, mut cn) = (v_off[r], t_off[r], n_off[r]);
            let mut out = FaceOut { pieces: Vec::new(), tris: 0, degenerate: 0, invalid: 0 };
            let mut bld = Builder::new(start_mat[r]);
            let mut corners: Vec<(u32, u32, u32)> = Vec::with_capacity(16);
            for l in lines(&data[s..e]) {
                match kind(l) {
                    (Kind::V, _) => cv += 1,
                    (Kind::Vt, _) => ct += 1,
                    (Kind::Vn, _) => cn += 1,
                    (Kind::UseMtl, o) => {
                        let m = mat_index.get(&rest_str(l, o)).copied().unwrap_or(0);
                        if m != bld.material {
                            finish(&mut bld, &mut out.pieces);
                            bld.material = m;
                        }
                    }
                    (Kind::F, o) => {
                        corners.clear();
                        let mut i = o;
                        let mut bad = false;
                        loop {
                            skip_ws(l, &mut i);
                            if i >= l.len() {
                                break;
                            }
                            let Some(vi) = parse_int(l, &mut i) else { break };
                            let mut ti: i64 = 0;
                            let mut ni: i64 = 0;
                            if i < l.len() && l[i] == b'/' {
                                i += 1;
                                if i < l.len() && l[i] != b'/' {
                                    ti = parse_int(l, &mut i).unwrap_or(0);
                                }
                                if i < l.len() && l[i] == b'/' {
                                    i += 1;
                                    ni = parse_int(l, &mut i).unwrap_or(0);
                                }
                            }
                            while i < l.len() && l[i] != b' ' && l[i] != b'\t' {
                                i += 1;
                            }
                            let res = |x: i64, count: usize, total: usize| -> Option<u32> {
                                let z = if x > 0 { x - 1 } else if x < 0 { count as i64 + x } else { return Some(u32::MAX) };
                                if z >= 0 && (z as usize) < total { Some(z as u32) } else { None }
                            };
                            let v = match res(vi, cv, nv) {
                                Some(v) if v != u32::MAX => v,
                                _ => {
                                    bad = true;
                                    u32::MAX
                                }
                            };
                            let t = res(ti, ct, nvt).unwrap_or(u32::MAX);
                            let n = res(ni, cn, nvn).unwrap_or(u32::MAX);
                            corners.push((v, t, n));
                        }
                        if corners.len() < 3 {
                            continue;
                        }
                        if bad {
                            out.invalid += 1;
                            continue;
                        }
                        if bld.pos.len() + corners.len() > MAX_VERTS {
                            finish(&mut bld, &mut out.pieces);
                        }
                        let mut ids = [0u32; 64];
                        let mut big: Vec<u32> = Vec::new();
                        for (k, &(v, t, n)) in corners.iter().enumerate() {
                            let key = (v, t, n);
                            let id = if let Some(&id) = bld.map.get(&key) {
                                id
                            } else {
                                let id = bld.pos.len() as u32;
                                bld.pos.push(positions[v as usize]);
                                if t != u32::MAX {
                                    bld.uv.push(texcoords[t as usize]);
                                    bld.any_uv = true;
                                } else {
                                    bld.uv.push([0.0, 0.0]);
                                    bld.missing_uv = true;
                                }
                                if n != u32::MAX {
                                    bld.nrm.push(pack_n(normals[n as usize]));
                                    bld.need_n.push(false);
                                } else {
                                    bld.nrm.push([0, 32767, 0, 0]);
                                    bld.need_n.push(true);
                                    bld.any_need_n = true;
                                }
                                bld.src.push(v);
                                bld.map.insert(key, id);
                                id
                            };
                            if k < 64 {
                                ids[k] = id;
                            } else {
                                if big.is_empty() {
                                    big.extend_from_slice(&ids);
                                }
                                big.push(id);
                            }
                        }
                        let get = |k: usize| if k < 64 && big.is_empty() { ids[k] } else { big[k] };
                        for k in 1..corners.len() - 1 {
                            let (i0, i1, i2) = (get(0), get(k), get(k + 1));
                            let p0 = bld.pos[i0 as usize];
                            let p1 = bld.pos[i1 as usize];
                            let p2 = bld.pos[i2 as usize];
                            let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                            let w = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                            let cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
                            if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] < degen_eps {
                                out.degenerate += 1;
                            }
                            bld.idx.extend_from_slice(&[i0, i1, i2]);
                            out.tris += 1;
                        }
                    }
                    _ => {}
                }
            }
            finish(&mut bld, &mut out.pieces);
            progress.done.fetch_add((e - s) as u64, Ordering::Relaxed);
            out
        })
        .collect();
    drop(texcoords);

    crate::log::line(&format!("faces built in {:.1} s", t0.elapsed().as_secs_f32()));
    let mut stats = MeshStats {
        verts: nv as u64,
        uvs: nvt as u64,
        normals: nvn as u64,
        file_bytes: total,
        ..Default::default()
    };
    let mut raw: Vec<RawPiece> = Vec::new();
    for o in outs {
        stats.tris += o.tris;
        stats.degenerate += o.degenerate;
        stats.invalid += o.invalid;
        raw.extend(o.pieces);
    }
    if stats.tris == 0 {
        return Err("No faces found. Check that this is a mesh OBJ.".into());
    }

    // ---- smooth normals where the file has none
    if raw.iter().any(|p| p.any_need_n) {
        progress.stage("Computing normals");
        let mut acc = vec![[0f32; 3]; nv];
        for p in &raw {
            let c = &p.chunk;
            for t in c.idx.chunks_exact(3) {
                let (a, b, d) = (c.pos[t[0] as usize], c.pos[t[1] as usize], c.pos[t[2] as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
                let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
                for &k in t {
                    let s = p.src[k as usize] as usize;
                    acc[s][0] += n[0];
                    acc[s][1] += n[1];
                    acc[s][2] += n[2];
                }
            }
        }
        raw.par_iter_mut().for_each(|p| {
            if !p.any_need_n {
                return;
            }
            for (i, need) in p.need_n.iter().enumerate() {
                if *need {
                    p.chunk.nrm[i] = pack_n(acc[p.src[i] as usize]);
                }
            }
        });
    }
    drop(positions);
    drop(normals);

    let chunks: Vec<Chunk> = raw.into_iter().map(|p| p.chunk).collect();
    stats.pieces = chunks.len() as u64;
    stats.pieces_without_uv = chunks.iter().filter(|c| c.uv.is_none()).count() as u64;
    stats.gpu_verts = chunks.iter().map(|c| c.pos.len() as u64).sum();

    // ---- materials and textures
    progress.stage("Loading textures");
    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut materials: Vec<Material> = mat_names
        .iter()
        .map(|n| Material { name: n.clone(), base: [0.8, 0.8, 0.8], texture_file: None, texture: None, texture_problem: None })
        .collect();
    let mut mtl_dir = dir.clone();
    for lib in &libs {
        let p = dir.join(lib);
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        if let Some(pd) = p.parent() {
            mtl_dir = pd.to_path_buf();
        }
        let mut cur: Option<usize> = None;
        for line in text.lines() {
            let t = line.trim();
            let mut parts = t.splitn(2, char::is_whitespace);
            let key = parts.next().unwrap_or("");
            let rest = parts.next().unwrap_or("").trim();
            match key {
                "newmtl" => cur = mat_index.get(rest).copied(),
                "Kd" => {
                    if let Some(m) = cur {
                        let v: Vec<f32> = rest.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                        if v.len() >= 3 {
                            materials[m].base = [v[0], v[1], v[2]];
                        }
                    }
                }
                k if k.eq_ignore_ascii_case("map_Kd") => {
                    if let Some(m) = cur {
                        // options like "-s 1 1 1" may come first; the file name is what is left at the end
                        let toks: Vec<&str> = rest.split_whitespace().collect();
                        let mut name = toks.last().copied().unwrap_or("").to_string();
                        if let Some(first_plain) = toks.iter().position(|x| !x.starts_with('-')) {
                            let joined = toks[first_plain..].join(" ");
                            if mtl_dir.join(&joined).exists() {
                                name = joined;
                            }
                        }
                        if !name.is_empty() {
                            materials[m].texture_file = Some(name.replace('\\', "/"));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let lookup = |name: &str| -> Option<PathBuf> {
        let direct = mtl_dir.join(name);
        if direct.exists() {
            return Some(direct);
        }
        let base = Path::new(name).file_name()?;
        let flat = mtl_dir.join(base);
        if flat.exists() {
            return Some(flat);
        }
        let flat2 = dir.join(base);
        if flat2.exists() {
            return Some(flat2);
        }
        None
    };
    materials.par_iter_mut().for_each(|m| {
        let Some(name) = m.texture_file.clone() else { return };
        let Some(p) = lookup(&name) else {
            m.texture_problem = Some("not found".into());
            return;
        };
        match image::open(&p) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                m.texture = Some(TextureImage { width: w, height: h, mips: build_mips(w, h, rgba.into_raw()) });
            }
            Err(_) => m.texture_problem = Some("format not supported".into()),
        }
    });

    stats.seconds = t0.elapsed().as_secs_f32();
    progress.done.store(total * 3, Ordering::Relaxed);
    Ok(LoadResult {
        name: path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        path: path.to_path_buf(),
        chunks,
        materials,
        stats,
        min: mn,
        max: mx,
    })
}

/// Box-filter mip chain on the CPU (sRGB treated as linear, which is fine for review).
fn build_mips(w: u32, h: u32, base: Vec<u8>) -> Vec<Vec<u8>> {
    let mut mips = vec![base];
    let (mut cw, mut ch) = (w, h);
    while cw > 1 || ch > 1 {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let prev = mips.last().unwrap();
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        next.par_chunks_mut((nw * 4) as usize).enumerate().for_each(|(y, row)| {
            for x in 0..nw as usize {
                for c in 0..4 {
                    let mut s = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx).min(cw as usize - 1);
                        let sy = (y * 2 + dy).min(ch as usize - 1);
                        s += prev[(sy * cw as usize + sx) * 4 + c] as u32;
                    }
                    row[x * 4 + c] = (s / 4) as u8;
                }
            }
        });
        mips.push(next);
        cw = nw;
        ch = nh;
    }
    mips
}
