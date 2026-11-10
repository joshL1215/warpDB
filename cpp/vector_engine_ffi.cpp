#include "vector_engine_ffi.h"
#include "vector_engine.h"

#include <string>
#include <vector>

struct NativeVectorEngine {
    VectorEngine engine;
};

struct NativeSearchResults {
    std::vector<SearchResult> results;
};

struct NativeBatchSearchResults {
    std::vector<std::vector<SearchResult>> results;
};

extern "C" NativeVectorEngine* native_vector_engine_new() {
    return new NativeVectorEngine();
}

extern "C" void native_vector_engine_free(NativeVectorEngine* engine) {
    delete engine;
}

extern "C" size_t native_search_results_len(const NativeSearchResults* results) {
    return results->results.size();
}

extern "C" const char* native_search_results_id_at(const NativeSearchResults* results, size_t index) {
    return results->results[index].id.c_str();
}

extern "C" float native_search_results_score_at(const NativeSearchResults* results, size_t index) {
    return results->results[index].score;
}

extern "C" void native_search_results_free(NativeSearchResults* results) {
    delete results;
}

extern "C" void native_vector_engine_insert(NativeVectorEngine* engine, const char* id, const float* vector, size_t len) {
    engine->engine.insert(std::string(id), vector, len);
}

extern "C" bool native_vector_engine_delete(NativeVectorEngine* engine, const char* id) {
    return engine->engine.erase(std::string(id));
}

extern "C" NativeSearchResults* native_vector_engine_search(NativeVectorEngine* engine, const float* query, size_t len, size_t k) {
    auto* results = new NativeSearchResults();
    results->results = engine->engine.search(query, len, k);
    return results;
}

extern "C" NativeBatchSearchResults* native_vector_engine_search_batch(NativeVectorEngine* engine, const float* queries, size_t n_queries, size_t len, size_t k) {
    auto* results = new NativeBatchSearchResults();
    results->results = engine->engine.search_batch(queries, n_queries, len, k);
    return results;
}

extern "C" size_t native_batch_search_results_num_queries(const NativeBatchSearchResults* results) {
    return results->results.size();
}

extern "C" size_t native_batch_search_results_len(const NativeBatchSearchResults* results, size_t query_index) {
    return results->results[query_index].size();
}

extern "C" const char* native_batch_search_results_id_at(const NativeBatchSearchResults* results, size_t query_index, size_t index) {
    return results->results[query_index][index].id.c_str();
}

extern "C" float native_batch_search_results_score_at(const NativeBatchSearchResults* results, size_t query_index, size_t index) {
    return results->results[query_index][index].score;
}

extern "C" void native_batch_search_results_free(NativeBatchSearchResults* results) {
    delete results;
}
