//! Files on network drives are copied to a local cache before loading.
//!
//! The loader memory-maps files for speed. On a network share, a brief connection drop while Windows
//! pages in part of a mapped file cannot be handled and ends the app ("I/O operation failed"). Copying
//! with ordinary reads lets us retry through drop-outs instead; the local copy is then safe to map, and
//! opening the same file again later skips the network entirely.

use crate::loader::Progress;
use std::hash::{Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

pub fn cache_dir() -> PathBuf {
    crate::log::path().parent().map(|p| p.join("cache")).unwrap_or_else(|| std::env::temp_dir().join("forma-cache"))
}

/// True for UNC paths (\\server\share) and drive letters mapped to a network share.
pub fn is_remote(path: &Path) -> bool {
    if std::env::var_os("FORMA_FORCE_CACHE").is_some() {
        return true;
    }
    let s = path.to_string_lossy();
    if s.starts_with("\\\\") || s.starts_with("//") {
        return !s.starts_with("\\\\?\\") || s.starts_with("\\\\?\\UNC\\");
    }
    #[cfg(windows)]
    {
        let b = s.as_bytes();
        if b.len() >= 2 && b[1] == b':' {
            use windows_sys::Win32::Storage::FileSystem::GetDriveTypeW;
            let root: Vec<u16> = format!("{}:\\", b[0] as char).encode_utf16().chain(std::iter::once(0)).collect();
            // DRIVE_REMOTE = 4
            return unsafe { GetDriveTypeW(root.as_ptr()) } == 4;
        }
    }
    false
}

fn free_space(dir: &Path) -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let w: Vec<u16> = dir.to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
        let mut free = 0u64;
        if unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) } != 0 {
            return Some(free);
        }
    }
    let _ = dir;
    None
}

/// Returns the path to load from: the file itself when it is local, otherwise an up-to-date local copy.
pub fn local_source(path: &Path, progress: &Progress) -> Result<PathBuf, String> {
    if !is_remote(path) {
        return Ok(path.to_path_buf());
    }
    let meta = std::fs::metadata(path).map_err(|e| format!("Could not reach the file on the network: {e}"))?;
    let size = meta.len();
    let mut h = rustc_hash::FxHasher::default();
    path.to_string_lossy().to_lowercase().hash(&mut h);
    size.hash(&mut h);
    meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).hash(&mut h);
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Could not create the local cache folder: {e}"))?;
    let target = dir.join(format!("{:016x}.obj", h.finish()));
    if std::fs::metadata(&target).map(|m| m.len() == size).unwrap_or(false) {
        crate::log::line(&format!("network file: using local copy {}", target.display()));
        return Ok(target);
    }

    // make room: drop older cached copies (oldest first) until this one fits with some margin
    let mut old: Vec<(std::time::SystemTime, PathBuf, u64)> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| {
                    let m = e.metadata().ok()?;
                    Some((m.modified().ok()?, e.path(), m.len()))
                })
                .collect()
        })
        .unwrap_or_default();
    old.sort();
    let need = size + (2 << 30);
    for (_, p, _) in &old {
        if free_space(&dir).map(|f| f >= need).unwrap_or(true) {
            break;
        }
        let _ = std::fs::remove_file(p);
    }
    if let Some(f) = free_space(&dir) {
        if f < size + (512 << 20) {
            return Err(format!(
                "The file is on a network drive and needs to be copied to this PC first, but there is only {} free on the local disk and the file is {}.",
                crate::log::gb(f),
                crate::log::gb(size)
            ));
        }
    }

    // copy with retries
    *progress.stage.lock().unwrap() = "Copying from the network".into();
    crate::log::line(&format!("network file: copying {} bytes to {}", size, target.display()));
    progress.total.store(size * 4, Ordering::Relaxed);
    let part = target.with_extension("part");
    let mut out = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&part).map_err(|e| format!("Could not write the local copy: {e}"))?);
    let mut src = std::fs::File::open(path).map_err(|e| format!("Could not open the file: {e}"))?;
    let mut buf = vec![0u8; 8 << 20];
    let mut off = 0u64;
    let mut failures = 0u32;
    while off < size {
        match src.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                out.write_all(&buf[..n]).map_err(|e| format!("Could not write the local copy: {e}"))?;
                off += n as u64;
                failures = 0;
                progress.done.store(off, Ordering::Relaxed);
            }
            Err(e) => {
                failures += 1;
                crate::log::line(&format!("network read failed at {off} (attempt {failures}): {e}"));
                if failures > 10 {
                    drop(out);
                    let _ = std::fs::remove_file(&part);
                    return Err(format!("The network connection kept failing while reading the file ({e}). Check the connection and try again."));
                }
                *progress.stage.lock().unwrap() = format!("Network hiccup, retrying ({failures})");
                std::thread::sleep(Duration::from_millis(500 * failures as u64));
                // reopen and continue where we stopped
                if let Ok(mut f) = std::fs::File::open(path) {
                    if f.seek(SeekFrom::Start(off)).is_ok() {
                        src = f;
                        *progress.stage.lock().unwrap() = "Copying from the network".into();
                    }
                }
            }
        }
    }
    out.flush().map_err(|e| format!("Could not write the local copy: {e}"))?;
    drop(out);
    if off != size {
        let _ = std::fs::remove_file(&part);
        return Err("The file changed size while it was being read. Try again.".into());
    }
    let _ = std::fs::remove_file(&target);
    std::fs::rename(&part, &target).map_err(|e| format!("Could not finish the local copy: {e}"))?;
    crate::log::line("network file: copy complete");
    Ok(target)
}
