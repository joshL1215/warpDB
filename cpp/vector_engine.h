#ifndef VECTOR_ENGINE_H
#define VECTOR_ENGINE_H

#include <cuda_runtime.h>
#include <kernels.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

struct SearchResult {
    std::string id;
    float score;
};

class VectorEngine {
private:
    std::vector<float> vectors;
    std::size_t dimension;

    std::vector<std::string> ids;
    std::unordered_map<std::string, std::size_t> id_map;

    std::vector<std::uint8_t> tombstones;
    int delete_count = 0;
    int compaction_limit;

    float *d_vectors = nullptr;
    std::uint8_t *d_tombstones = nullptr;
    std::size_t gpu_vec_cap;
    bool gpu_dirty = false;

    void sync_to_gpu();

public:
    VectorEngine();
    ~VectorEngine();

    void insert(
        std::string id,
        const float *vector,
        std::size_t len
    );

    bool erase(std::string id);

    // max results per query; must match K_MAX in kernels/cosine_search.cu
    static constexpr std::size_t MAX_K = 32;

    std::vector<SearchResult> search(
        const float *query,
        std::size_t len,
        std::size_t k
    );

    // queries is row-major [n_queries x len]; returns one result list per query,
    // in the same order as the input (empty lists if the batch could not run)
    std::vector<std::vector<SearchResult>> search_batch(
        const float *queries,
        std::size_t n_queries,
        std::size_t len,
        std::size_t k
    );
};

#endif
