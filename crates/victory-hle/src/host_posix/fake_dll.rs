//! Module handles for the system DLLs we serve (`kernel32.dll`, the
//! `api-ms-win-core-*` sets, `ucrtbase.dll`, ...), which are no files and
//! no images here. A title's start-up code asks for them by name
//! (`GetModuleHandleW(L"kernel32.dll")`) and looks functions up with
//! `GetProcAddress`.
//!
//! A handle is a stable 64 KiB region per normalised DLL name holding a
//! minimal read-only PE header (DOS stub pointer, `PE\0\0`, an optional
//! header with no sections and empty data directories), so title code that
//! reads the headers sees a valid, export-less 64-bit image instead of a
//! crash. `GetProcAddress` resolves through the import binder's resolver and
//! hands out the same bridge stub the binder would, cached per (DLL, name).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{LazyLock, Mutex};

use victory_loader::native::bridge;
use victory_loader::{ImportResolver, Resolution};
use victory_pe::ImportSymbol;
use victory_platform::mem::{self, Protect};

use super::{GRANULARITY, lock, set_error};
use crate::{HleResolver, HostPolicy};

const ERROR_PROC_NOT_FOUND: u32 = 127;

/// (lower-case DLL file name, base).
static MODULES: LazyLock<Mutex<Vec<(String, usize)>>> = LazyLock::new(Mutex::default);
static PROCS: LazyLock<Mutex<HashMap<(usize, String), usize>>> = LazyLock::new(Mutex::default);

/// `kernel32` for `C:\Windows\System32\Kernel32.DLL`, `kernel32.dll` for `kernel32`.
fn normalise(path: &str) -> String {
    let file = path.rsplit(['\\', '/']).next().unwrap_or(path);
    let file = file.to_ascii_lowercase();
    if file.contains('.') {
        file
    } else {
        format!("{file}.dll")
    }
}

/// The handle of the system DLL `path` names, made on first use; `None` if
/// the host policy does not pass that DLL (it would not resolve anything).
pub(crate) fn handle(path: &str) -> Option<*mut c_void> {
    let name = normalise(path);
    if !HostPolicy::default().allows(&name) {
        return None;
    }
    let mut modules = lock(&MODULES);
    if let Some((_, base)) = modules.iter().find(|(n, _)| *n == name) {
        return Some(*base as *mut c_void);
    }
    let base = map_header().ok()?;
    modules.push((name, base));
    Some(base as *mut c_void)
}

/// The DLL name behind a handle from [`handle`].
fn name_of(module: *mut c_void) -> Option<String> {
    lock(&MODULES)
        .iter()
        .find(|(_, b)| *b == module as usize)
        .map(|(n, _)| n.clone())
}

pub(crate) fn is_fake(module: *mut c_void) -> bool {
    name_of(module).is_some()
}

fn map_header() -> std::io::Result<usize> {
    let base = super::reserve_granular(None, GRANULARITY)?;
    let init = (|| -> std::io::Result<()> {
        // SAFETY: the reservation just made; written, then made read-only.
        unsafe {
            mem::commit(base, GRANULARITY, Protect::READ_WRITE)?;
            let put = |at: usize, bytes: &[u8]| {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(at), bytes.len());
            };
            put(0, b"MZ");
            put(0x3C, &0x80u32.to_le_bytes()); // e_lfanew
            put(0x80, b"PE\0\0");
            put(0x84, &0x8664u16.to_le_bytes()); // machine
            put(0x94, &0xF0u16.to_le_bytes()); // optional header size
            put(0x96, &0x2022u16.to_le_bytes()); // executable, large address aware, DLL
            let opt = 0x98;
            put(opt, &0x20Bu16.to_le_bytes()); // PE32+
            put(opt + 24, &(base as u64).to_le_bytes()); // image base
            put(opt + 32, &0x1000u32.to_le_bytes()); // section alignment
            put(opt + 36, &0x200u32.to_le_bytes()); // file alignment
            put(opt + 40, &10u16.to_le_bytes()); // OS version
            put(opt + 48, &10u16.to_le_bytes()); // subsystem version
            put(opt + 56, &(GRANULARITY as u32).to_le_bytes()); // size of image
            put(opt + 60, &0x200u32.to_le_bytes()); // size of headers
            put(opt + 68, &3u16.to_le_bytes()); // console subsystem
            put(opt + 108, &16u32.to_le_bytes()); // data directories, all empty
            mem::protect(base, GRANULARITY, Protect::READ)?;
        }
        Ok(())
    })();
    if let Err(err) = init {
        // SAFETY: `base` was reserved above and is not exposed on failure.
        let _ = unsafe { mem::release(base, GRANULARITY) };
        return Err(err);
    }
    Ok(base as usize)
}

/// A callable entry for `name` (or `#ordinal`) of a fake module: null with
/// `ERROR_PROC_NOT_FOUND` if neither our runtime nor `host_posix` has it.
pub(crate) fn proc_address(module: *mut c_void, symbol: ImportSymbol) -> Option<*mut c_void> {
    let dll = name_of(module)?;
    let key = (module as usize, symbol.to_string());
    if let Some(&stub) = lock(&PROCS).get(&key) {
        return Some(stub as *mut c_void);
    }
    let found = match HleResolver::default().resolve(&dll, &symbol) {
        Resolution::Found { address, .. } => bridge::stub_for(address).ok(),
        Resolution::Missing => None,
    };
    let Some(stub) = found else {
        tracing::warn!(dll, symbol = %key.1, "GetProcAddress: not implemented");
        set_error(ERROR_PROC_NOT_FOUND);
        return Some(std::ptr::null_mut());
    };
    lock(&PROCS).insert(key, stub);
    Some(stub as *mut c_void)
}

/// A callable win64 entry for one of our own functions (behind a console
/// pseudo module): title code must reach it through the GS-swapping bridge.
pub(crate) fn bridged(address: usize) -> Option<usize> {
    let key = (0, format!("{address:#x}"));
    if let Some(&stub) = lock(&PROCS).get(&key) {
        return Some(stub);
    }
    let stub = bridge::stub_for(address).ok()?;
    lock(&PROCS).insert(key, stub);
    Some(stub)
}

/// `C:\Windows\System32\<name>` as UTF-16, no NUL.
pub(crate) fn path(module: *mut c_void) -> Option<Vec<u16>> {
    let name = name_of(module)?;
    Some(format!(r"C:\Windows\System32\{name}").encode_utf16().collect())
}
