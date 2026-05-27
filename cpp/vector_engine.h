#ifndef VECTOR_ENGINE_H
#define VECTOR_ENGINE_H

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

public:
    VectorEngine();
    ~VectorEngine();

    void insert(
        std::string id,
        const float *vector,
        std::size_t len
    );

    bool erase(std::string id);

    std::vector<SearchResult> search(
        const float *query,
        std::size_t len,
        std::size_t k
    );
};

#endif
