//! Safe wrapper over the C ABI.
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

use crate::ffi;

/// A failed C ABI call: its status code and a description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub status: i32,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "aegis status {}: {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn status_string(status: i32) -> String {
    // SAFETY: aegis_status_string returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::aegis_status_string(status)) }
        .to_string_lossy()
        .into_owned()
}

fn check(status: i32, ctx: Option<&Context>) -> Result<()> {
    if status == ffi::AEGIS_OK as i32 {
        return Ok(());
    }
    let mut message = status_string(status);
    if let Some(ctx) = ctx {
        let detail = ctx.last_error();
        if !detail.is_empty() {
            message = format!("{message}: {detail}");
        }
    }
    Err(Error { status, message })
}

/// Copies a string out of a `(buf, len, needed)` C API using its size query.
fn read_string(mut call: impl FnMut(*mut c_char, usize, *mut usize) -> i32) -> Result<String> {
    let mut needed = 0usize;
    check(call(ptr::null_mut(), 0, &mut needed), None)?;
    let mut buf = vec![0u8; needed];
    check(call(buf.as_mut_ptr().cast(), buf.len(), &mut needed), None)?;
    let s = CStr::from_bytes_until_nul(&buf).map_err(|_| Error {
        status: ffi::AEGIS_ERR_INVALID_ARGUMENT,
        message: "missing NUL terminator".into(),
    })?;
    Ok(s.to_string_lossy().into_owned())
}

/// C ABI version, `major * 1000 + minor`.
pub fn abi_version() -> i32 {
    // SAFETY: no arguments, no side effects.
    unsafe { ffi::aegis_abi_version() }
}

/// The device manifest JSON (measured bandwidth included) for `device`.
pub fn device_manifest(device: i32) -> Result<String> {
    // SAFETY: buf/len/needed follow the size-query protocol in aegis.h.
    read_string(|buf, len, needed| unsafe { ffi::aegis_device_manifest(device, buf, len, needed) })
}

/// One device plus one stream. Usable from one thread at a time: `Send`, not `Sync`.
pub struct Context {
    raw: NonNull<ffi::aegis_context>,
}

// SAFETY: aegis.h allows a context to move between threads as long as only
// one uses it at a time, which `!Sync` enforces.
unsafe impl Send for Context {}

impl Context {
    pub fn new(device: i32) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer.
        check(unsafe { ffi::aegis_create(&mut raw, device) }, None)?;
        Ok(Self { raw: NonNull::new(raw).expect("aegis_create returned OK with NULL") })
    }

    fn as_ptr(&self) -> *mut ffi::aegis_context {
        self.raw.as_ptr()
    }

    /// The last CUDA error text this context saw ("" if none).
    pub fn last_error(&self) -> String {
        // SAFETY: valid context; returns a string owned by the context.
        unsafe { CStr::from_ptr(ffi::aegis_last_error(self.as_ptr())) }
            .to_string_lossy()
            .into_owned()
    }

    /// Wait for all work queued on this context's stream.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: valid context.
        check(unsafe { ffi::aegis_synchronize(self.as_ptr()) }, Some(self))
    }

    /// `c[m,n] = a[m,k] * b[k,n]`, row-major FP16 (as `u16` bit patterns),
    /// FP32 accumulate. Asynchronous; `c.to_host()` or `synchronize()` waits.
    pub fn gemm_f16(
        &self,
        a: &DeviceBuffer<'_>,
        b: &DeviceBuffer<'_>,
        c: &mut DeviceBuffer<'_>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let invalid = |what: &str| Error {
            status: ffi::AEGIS_ERR_INVALID_ARGUMENT,
            message: what.to_string(),
        };
        let dim = |x: usize| i32::try_from(x).ok().filter(|&v| v > 0);
        let (Some(mi), Some(ni), Some(ki)) = (dim(m), dim(n), dim(k)) else {
            return Err(invalid("dimensions must be in 1..=i32::MAX"));
        };
        if a.len < m * k || b.len < k * n || c.len < m * n {
            return Err(invalid("buffer smaller than its matrix"));
        }
        for buf in [a.ctx, b.ctx, c.ctx] {
            if !ptr::eq(buf, self) {
                return Err(invalid("buffer belongs to another context"));
            }
        }
        // SAFETY: valid context; device pointers sized-checked above.
        check(
            unsafe { ffi::aegis_gemm_f16(self.as_ptr(), a.ptr, b.ptr, c.ptr, mi, ni, ki) },
            Some(self),
        )
    }

    /// Which implementation `gemm_f16` would run for this shape.
    pub fn gemm_choice(&self, m: i32, n: i32, k: i32) -> Result<String> {
        // SAFETY: valid context; size-query protocol.
        read_string(|buf, len, needed| unsafe {
            ffi::aegis_gemm_choice(self.as_ptr(), m, n, k, buf, len, needed)
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: we own the context; buffers borrow it, so none outlive it.
        unsafe { ffi::aegis_destroy(self.as_ptr()) };
    }
}

/// Device memory holding `len` FP16 values, freed on drop. Borrows its context.
pub struct DeviceBuffer<'ctx> {
    ctx: &'ctx Context,
    ptr: *mut c_void,
    len: usize,
    _not_sync: PhantomData<*mut u16>,
}

impl<'ctx> DeviceBuffer<'ctx> {
    pub fn zeroed(ctx: &'ctx Context, len: usize) -> Result<Self> {
        let buf = Self::alloc(ctx, len)?;
        buf.write(&vec![0u16; len])?;
        Ok(buf)
    }

    pub fn from_host(ctx: &'ctx Context, data: &[u16]) -> Result<Self> {
        let buf = Self::alloc(ctx, data.len())?;
        buf.write(data)?;
        Ok(buf)
    }

    fn alloc(ctx: &'ctx Context, len: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        // SAFETY: valid context and out-pointer. A zero-size request is rejected by the C side.
        check(unsafe { ffi::aegis_device_alloc(ctx.as_ptr(), len * 2, &mut ptr) }, Some(ctx))?;
        Ok(Self { ctx, ptr, len, _not_sync: PhantomData })
    }

    fn write(&self, data: &[u16]) -> Result<()> {
        // SAFETY: `ptr` holds `len` u16 and `data.len() == len` (callers).
        check(
            unsafe {
                ffi::aegis_copy_to_device(self.ctx.as_ptr(), self.ptr, data.as_ptr().cast(), data.len() * 2)
            },
            Some(self.ctx),
        )
    }

    /// Copy back to the host (waits for queued work on the context's stream).
    pub fn to_host(&self) -> Result<Vec<u16>> {
        let mut out = vec![0u16; self.len];
        // SAFETY: `out` holds `len` u16, as does the device allocation.
        check(
            unsafe {
                ffi::aegis_copy_to_host(self.ctx.as_ptr(), out.as_mut_ptr().cast(), self.ptr, self.len * 2)
            },
            Some(self.ctx),
        )?;
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: we own the allocation; its context is still alive (borrow).
        unsafe { ffi::aegis_device_free(self.ctx.as_ptr(), self.ptr) };
    }
}
