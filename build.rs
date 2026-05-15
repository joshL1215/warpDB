fn main() {
    cc::Build::new()
        .cpp(true)
        .file("cpp/src/vector_engine_ffi.cpp")
        .include("cpp/include")
        .std("c++17")
        .compile("vector_engine_ffi");
}
