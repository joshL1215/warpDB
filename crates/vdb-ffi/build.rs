use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../cpp/vector_engine_ffi.h");

    let bindings = bindgen::Builder::default()
        .header("../../cpp/vector_engine_ffi.h")
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        .clang_arg("-I../../cpp")
        .allowlist_type("NativeVectorEngine")
        .allowlist_type("NativeSearchResults")
        .allowlist_function("native_.*")
        .layout_tests(false)
        .generate()
        .expect("failed to generate FFI bindings with bindgen");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR was not set"));

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write generated bindings");
}
