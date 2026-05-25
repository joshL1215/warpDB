#ifndef VECTOR_ENGINE_H
#define VECTOR_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

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
    int delete_count;
    int compaction_limit;

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
