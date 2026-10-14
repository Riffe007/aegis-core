//! AEGIS Core from Rust.
//!
//! * [`ffi`]: raw bindgen output for `include/aegis/aegis.h`.
//! * [`safe`]: RAII context and device buffers, `Result`-returning calls.
//!
//! ```no_run
//! use aegis_sys::safe::{Context, DeviceBuffer};
//! let ctx = Context::new(0)?;
//! let a = DeviceBuffer::from_host(&ctx, &vec![0u16; 64 * 64])?;
//! let b = DeviceBuffer::from_host(&ctx, &vec![0u16; 64 * 64])?;
//! let mut c = DeviceBuffer::zeroed(&ctx, 64 * 64)?;
//! ctx.gemm_f16(&a, &b, &mut c, 64, 64, 64)?;
//! let out = c.to_host()?;
//! # Ok::<(), aegis_sys::safe::Error>(())
//! ```

#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub mod safe;
