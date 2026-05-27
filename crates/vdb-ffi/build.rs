use std::path::PathBuf;

fn main() {
    // linking the compiled cpp files
    let cpp_dir = PathBuf::from("../../cpp");
    let mut native_build = cc::Build::new();
    native_build
        .cpp(true)
        .std("c++17")
        .include(&cpp_dir)
        .file(cpp_dir.join("vector_engine.cpp"))
        .file(cpp_dir.join("vector_engine_ffi.cpp"));

    native_build.compile("vector_engine_native");

    // generates rust declarations based of cpp ffi
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
