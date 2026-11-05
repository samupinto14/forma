//! Files on network drives are copied to a local cache before loading.
//!
//! The loader memory-maps files for speed. On a network share, a brief connection drop while Windows
//! pages in part of a mapped file cannot be handled and ends the app ("I/O operation failed"). Copying
//! with ordinary reads lets us retry through drop-outs instead; the local copy is then safe to map, and
//! opening the same file again later skips the network entirely.

use crate::loader::Progress;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

pub fn cache_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Some(h) = std::env::var_os("HOME") {
        return PathBuf::from(h).join("Library").join("Caches").join("Forma");
    }
    crate::log::path().parent().map(|p| p.join("cache")).unwrap_or_else(|| std::env::temp_dir().join("forma-cache"))
}

#[cfg(target_os = "macos")]
fn statfs(path: &Path) -> Option<libc::statfs> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statfs(c.as_ptr(), &mut st) } == 0).then_some(st)
}

/// True for UNC paths (\\server\share), drive letters mapped to a network share, and network volumes on a Mac
/// (SMB, AFP, NFS, WebDAV).
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
    #[cfg(target_os = "macos")]
    if let Some(st) = statfs(path) {
        return st.f_flags & (libc::MNT_LOCAL as u32) == 0;
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
    #[cfg(target_os = "macos")]
    if let Some(st) = statfs(dir) {
        return Some(st.f_bavail as u64 * st.f_bsize as u64);
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
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Could not create the local cache folder: {e}"))?;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let target = dir.join(format!("forma-{}-{}.obj", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    if let Some(f) = free_space(&dir) {
        if f < size + (512 << 20) {
            return Err(format!(
                "The file is on a network drive and needs to be copied to this computer first, but there is only {} free on the local disk and the file is {}.",
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
    let mut part_guard = TempCopy(Some(part.clone())); // removes the half-written copy on any failure below
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
        return Err("The file changed size while it was being read. Try again.".into());
    }
    std::fs::rename(&part, &target).map_err(|e| format!("Could not finish the local copy: {e}"))?;
    part_guard.0 = None;
    crate::log::line("network file: copy complete");
    Ok(target)
}

/// Deletes the temporary local copy when dropped (does nothing for files that were loaded in place).
pub struct TempCopy(Option<PathBuf>);

impl TempCopy {
    pub fn new(original: &Path, source: &Path) -> Self {
        TempCopy((original != source).then(|| source.to_path_buf()))
    }
}

impl Drop for TempCopy {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            match std::fs::remove_file(&p) {
                Ok(()) => crate::log::line("network file: temporary copy deleted"),
                Err(e) => crate::log::line(&format!("network file: could not delete {}: {e}", p.display())),
            }
            let _ = std::fs::remove_dir(cache_dir()); // only succeeds when empty
        }
    }
}

/// Removes copies left behind if Forma was closed or crashed while loading. Also clears copies kept by
/// versions before 5.1.0-alpha.2, which reused them between loads.
pub fn clear_leftovers() {
    let dir = cache_dir();
    let Ok(rd) = std::fs::read_dir(&dir) else { return };
    let mut freed = 0u64;
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
        // a copy still being written by another open Forma window is left alone
        let busy = ext == "part" && e.metadata().and_then(|m| m.modified()).map(|t| t.elapsed().map(|d| d.as_secs() < 120).unwrap_or(true)).unwrap_or(false);
        if (ext == "obj" || ext == "part") && !busy {
            let len = e.metadata().map(|m| m.len()).unwrap_or(0);
            if std::fs::remove_file(&p).is_ok() {
                freed += len;
            }
        }
    }
    let _ = std::fs::remove_dir(&dir);
    if freed > 0 {
        crate::log::line(&format!("cleared {} of leftover network copies", crate::log::gb(freed)));
    }
}

/// Number of files currently in the cache folder (used by the self-test).
pub fn leftover_count() -> usize {
    std::fs::read_dir(cache_dir()).map(|rd| rd.count()).unwrap_or(0)
}
