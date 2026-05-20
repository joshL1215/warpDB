#ifndef VECTOR_ENGINE_FFI_H
#define VECTOR_ENGINE_FFI_H

#include <stddef.h>

struct NativeVectorEngine;
struct NativeSearchResults;

extern "C" {

    // subject to change
    NativeVectorEngine* native_vector_engine_new();
    void native_vector_engine_free(NativeVectorEngine* engine);

    // insert / delete / search
    void native_vector_engine_insert( NativeVectorEngine* engine, const char* id, const float* vector, size_t len);
    bool native_vector_engine_delete( NativeVectorEngine* engine, const char* id);
    NativeSearchResults* native_vector_engine_search( const NativeVectorEngine* engine, const float* query,size_t len, size_t k);

    // working with pointers to send info back to rust
    size_t native_search_results_len(const NativeSearchResults* results);
    const char* native_search_results_id_at(const NativeSearchResults* results, size_t index);
    float native_search_results_score_at(const NativeSearchResults* results, size_t index);
    void native_search_results_free(NativeSearchResults* results);

}

#endif
