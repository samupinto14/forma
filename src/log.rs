//! A small log file so problems can be traced: %LOCALAPPDATA%\Forma\forma-log.txt on Windows.

use std::io::Write;
use std::path::PathBuf;

pub fn path() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("Forma").join("forma-log.txt")
}

pub fn line(msg: &str) {
    let p = path();
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    // keep the file from growing forever
    if std::fs::metadata(&p).map(|m| m.len() > 2_000_000).unwrap_or(false) {
        let _ = std::fs::remove_file(&p);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = writeln!(f, "[{t}] {msg}");
    }
}

pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        line(&format!("PANIC: {info}\n{bt}"));
    }));
}

/// Free physical memory and remaining commit (RAM + page file) in bytes, where the OS can tell us.
pub fn memory() -> Option<(u64, u64, u64)> {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 {
            return Some((m.ullAvailPhys, m.ullAvailPageFile, m.ullTotalPhys));
        }
    }
    None
}

pub fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}
