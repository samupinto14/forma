#![windows_subsystem = "windows"]

mod app;
mod camera;
mod loader;
mod log;
mod render;
mod selftest;

use eframe::{egui, egui_wgpu, wgpu};
use std::sync::Arc;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> eframe::Result<()> {
    log::install_panic_hook();
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 && args[1] == "--selftest" {
        if let Err(e) = selftest::run(&args[2], &args[3]) {
            let _ = std::fs::write(format!("{}_error.txt", args[3]), e);
            std::process::exit(1);
        }
        std::process::exit(0);
    }
    let icon = image::load_from_memory(include_bytes!("../assets/logo.png")).ok().map(|img| {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        egui::IconData { rgba: rgba.into_raw(), width: w, height: h }
    });
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Forma")
        .with_inner_size([1440.0, 900.0])
        .with_min_inner_size([960.0, 600.0])
        .with_drag_and_drop(true);
    if let Some(icon) = icon {
        viewport = viewport.with_icon(Arc::new(icon));
    }
    let options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            power_preference: wgpu::PowerPreference::HighPerformance,
            device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
                label: Some("forma device"),
                required_features: wgpu::Features::empty(),
                // ask for everything the graphics card offers: big buffers and big textures
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            }),
            ..Default::default()
        },
        persist_window: true,
        ..Default::default()
    };
    eframe::run_native("Forma", options, Box::new(|cc| Ok(Box::new(app::FormaApp::new(cc)))))
}
