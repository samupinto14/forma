// Forma mesh + line shaders. Output goes to an sRGB target, so everything here is in linear light.

struct Globals {
    mvp: mat4x4<f32>,      // projection * view * model, for meshes
    vp: mat4x4<f32>,       // projection * view, for grid and axis lines
    model: mat4x4<f32>,    // model rotation, for normals
    key_dir: vec4<f32>,    // xyz direction towards the key light
    key: vec4<f32>,        // x = key intensity, y = fill intensity
    params: vec4<f32>,     // x = mode, y = checker repeat
    shape_col: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

struct Mat {
    base: vec4<f32>,
    flags: vec4<f32>,      // x = has texture
};
@group(1) @binding(0) var<uniform> m: Mat;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) n: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) has_uv: f32,
};

@vertex
fn vs_uv(@location(0) p: vec3<f32>, @location(1) n: vec4<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var o: VOut;
    o.pos = g.mvp * vec4<f32>(p, 1.0);
    o.n = (g.model * vec4<f32>(n.xyz, 0.0)).xyz;
    o.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    o.has_uv = 1.0;
    return o;
}

@vertex
fn vs_nouv(@location(0) p: vec3<f32>, @location(1) n: vec4<f32>) -> VOut {
    var o: VOut;
    o.pos = g.mvp * vec4<f32>(p, 1.0);
    o.n = (g.model * vec4<f32>(n.xyz, 0.0)).xyz;
    o.uv = vec2<f32>(0.0, 0.0);
    o.has_uv = 0.0;
    return o;
}

fn lit(n: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let ground = vec3<f32>(0.08, 0.09, 0.10);
    let hemi = mix(ground, vec3<f32>(1.0, 1.0, 1.0), n.y * 0.5 + 0.5) * g.key.y;
    let k = max(dot(n, normalize(g.key_dir.xyz)), 0.0) * g.key.x;
    // keep the brightest lit point at the texture's own brightness, so lighting reveals form without blowing out
    let norm = max(g.key.x + g.key.y, 1.0);
    return albedo * (hemi + vec3<f32>(k, k, k)) / norm * 1.25;
}

fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h, h, h) + k) * 6.0 - vec3<f32>(3.0, 3.0, 3.0));
    let c = v * mix(vec3<f32>(1.0, 1.0, 1.0), clamp(p - vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0), vec3<f32>(1.0)), s);
    return pow(c, vec3<f32>(2.2, 2.2, 2.2));
}

@fragment
fn fs_main(v: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
    // sample first: texture sampling must stay in uniform control flow
    let t = textureSample(tex, samp, v.uv);
    var n = normalize(v.n);
    if (!ff) { n = -n; }
    let mode = i32(g.params.x + 0.5);
    var albedo = m.base.rgb;
    if (m.flags.x > 0.5 && v.has_uv > 0.5) { albedo = albedo * t.rgb; }
    var col: vec3<f32>;
    if (mode == 0) {
        col = g.shape_col.rgb;
    } else if (mode == 1) {
        col = lit(n, vec3<f32>(0.46, 0.46, 0.46));
    } else if (mode == 2) {
        col = albedo;
    } else if (mode == 3) {
        col = lit(n, albedo);
    } else if (mode == 4) {
        let rep = g.params.y;
        let c = v.uv * rep;
        let cell = floor(c);
        let odd = (i32(cell.x) + i32(cell.y)) & 1;
        let h = fract(cell.x / max(rep, 1.0)) * 0.85;
        var base = hsv(h, 0.45, 0.28);
        if (odd == 1) { base = hsv(h, 0.45, 0.78); }
        if (v.has_uv < 0.5) { base = vec3<f32>(0.3, 0.3, 0.3); }
        col = lit(n, base);
    } else {
        col = pow(n * 0.5 + vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(2.2, 2.2, 2.2));
    }
    return vec4<f32>(col, 1.0);
}

struct LOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) c: vec4<f32>,
};

@vertex
fn vs_line(@location(0) p: vec3<f32>, @location(1) c: vec4<f32>) -> LOut {
    var o: LOut;
    o.pos = g.vp * vec4<f32>(p, 1.0);
    o.c = c;
    return o;
}

@fragment
fn fs_line(l: LOut) -> @location(0) vec4<f32> {
    return l.c;
}
