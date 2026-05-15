fn main() {
    cc::Build::new()
        .cpp(true)
        .file("native/src/vector_engine_ffi.cpp")
        .include("native/include")
        .std("c++17")
        .compile("vector_engine_ffi");
}
