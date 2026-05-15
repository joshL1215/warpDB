#include "vector_engine_ffi.h"

struct NativeVectorEngine {};

extern "C" NativeVectorEngine* native_vector_engine_new() {
    return new NativeVectorEngine();
}

extern "C" void native_vector_engine_free(NativeVectorEngine* engine) {
    delete engine;
}
