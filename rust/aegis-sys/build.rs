//! Generates bindings from include/aegis/aegis.h and links libaegis_capi.so.
//!
//! The shared library comes from the CMake build. Its directory is taken from
//! AEGIS_LIB_DIR, defaulting to <repo>/build. It is added to the rpath of this
//! crate's tests and examples, so they run without LD_LIBRARY_PATH.
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.join("../..").canonicalize().expect("repo root");
    let header = repo.join("include/aegis/aegis.h");
    let lib_dir = env::var("AEGIS_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo.join("build"));

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=AEGIS_LIB_DIR");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=aegis_capi");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    // Dependents that need the rpath can read DEP_AEGIS_CAPI_LIB_DIR.
    println!("cargo:lib_dir={}", lib_dir.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .allowlist_function("aegis_.*")
        .allowlist_type("aegis_.*")
        .allowlist_var("AEGIS_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed on aegis.h");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(out).expect("write bindings");
}
