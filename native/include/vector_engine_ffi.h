#ifndef VECTOR_ENGINE_FFI_H
#define VECTOR_ENGINE_FFI_H

struct NativeVectorEngine;

extern "C" {

NativeVectorEngine* native_vector_engine_new();
void native_vector_engine_free(NativeVectorEngine* engine);

}

#endif
