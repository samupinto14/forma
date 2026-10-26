//! Camera that orbits around a fixed pivot (the model centre) without re-aiming, so the view never jumps.

use glam::{Mat4, Quat, Vec3};

pub struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
    pub fov_y: f32,
    pub pivot: Vec3,
    pub spin: (f32, f32),
}

impl Default for Camera {
    fn default() -> Self {
        Camera { pos: Vec3::new(0.0, 0.0, 5.0), rot: Quat::IDENTITY, fov_y: 35f32.to_radians(), pivot: Vec3::ZERO, spin: (0.0, 0.0) }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        self.rot * Vec3::NEG_Z
    }
    pub fn right(&self) -> Vec3 {
        self.rot * Vec3::X
    }
    pub fn up(&self) -> Vec3 {
        self.rot * Vec3::Y
    }
    pub fn view(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rot, self.pos).inverse()
    }
    pub fn proj(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect.max(0.01), near, far)
    }
    pub fn look_at(&mut self, eye: Vec3, target: Vec3) {
        self.pos = eye;
        let dir = (target - eye).normalize_or_zero();
        // looking straight down: pick a stable "up" so the view isn't undefined
        let up = if dir.y.abs() > 0.999 { Vec3::NEG_Z } else { Vec3::Y };
        let view = Mat4::look_at_rh(eye, target, up);
        self.rot = Quat::from_mat4(&view.inverse()).normalize();
        self.spin = (0.0, 0.0);
    }
    /// dx, dy in radians.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let mut qp = Quat::from_axis_angle(self.right(), -dy);
        if (qp * self.up()).y < 0.01 {
            qp = Quat::IDENTITY; // stop at straight up / straight down instead of flipping over
        }
        let q = Quat::from_axis_angle(Vec3::Y, -dx) * qp;
        self.pos = self.pivot + q * (self.pos - self.pivot);
        self.rot = (q * self.rot).normalize();
    }
    /// Screen-space pan, `wpp` = world units per pixel at the pivot's distance.
    pub fn pan(&mut self, dx: f32, dy: f32, wpp: f32) {
        self.pos += self.right() * (-dx * wpp) + self.up() * (dy * wpp);
    }
    pub fn world_per_pixel(&self, viewport_h: f32, radius: f32) -> f32 {
        let d = self.pos.distance(self.pivot).max(radius * 0.01);
        2.0 * d * (self.fov_y * 0.5).tan() / viewport_h.max(1.0)
    }
    /// Ray through a viewport position in normalised device coordinates.
    pub fn ray(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> (Vec3, Vec3) {
        let t = (self.fov_y * 0.5).tan();
        let dir = (self.forward() + self.right() * (ndc_x * t * aspect) + self.up() * (ndc_y * t)).normalize();
        (self.pos, dir)
    }
}
