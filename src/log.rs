//! A small log file so problems can be traced: %LOCALAPPDATA%\Forma\forma-log.txt on Windows,
//! ~/Library/Logs/Forma/forma-log.txt on a Mac.

use std::io::Write;
use std::path::PathBuf;

pub fn path() -> PathBuf {
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library").join("Logs")).unwrap_or_else(std::env::temp_dir);
    #[cfg(not(target_os = "macos"))]
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

/// Free physical memory, remaining commit (RAM + page file) and total RAM in bytes, where the OS can tell us.
pub fn memory() -> Option<(u64, u64, u64)> {
    #[cfg(target_os = "macos")]
    unsafe {
        // macOS compresses memory and grows swap as needed, so there is no hard commit limit like on
        // Windows. Report total RAM, treat most of it as usable and allow twice that before refusing.
        let mut total: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        if libc::sysctlbyname(name.as_ptr() as *const libc::c_char, &mut total as *mut u64 as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) == 0 && total > 0 {
            return Some((total / 10 * 7, total * 2, total));
        }
    }
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
