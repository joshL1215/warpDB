#include "vector_engine.h"

namespace {
    std::vector<float> l2normed_cpu(const float *vector, std::size_t len) {
        float partial = 0.0f;
        for (std::size_t i = 0; i < len; i++) {
            float val = vector[i];
            partial += val * val;
        }

        float norm = std::sqrt(partial);
        std::vector<float> res(len);

        if (norm == 0.0f) {
            return res;
        }

        for (std::size_t i = 0; i < len; i++) {
            res[i] = vector[i] / norm;
        }
        return res;
    }
}

VectorEngine::VectorEngine()
    : dimension{768}, compaction_limit{25}, gpu_vec_cap{5000} {
}
// Dimensions is 768 only, can change or generalize later
// but for simplicity we only allow 768 dim vectors at all
// TODO: generalize ^

VectorEngine::~VectorEngine() {}

void VectorEngine::insert(
    std::string id,
    const float *vector,
    std::size_t len
) {

    if (len != dimension) return;

    std::vector<float> normalized_vec = l2normed_cpu(vector, len);
    id_map.insert({id, ids.size()});
    ids.push_back(id);
    tombstones.push_back(0);
    vectors.insert(
        vectors.end(),
        normalized_vec.begin(),
        normalized_vec.end()
    );
}

bool VectorEngine::erase(std::string id) {

    // amortized compaction at checkpoints
    if (delete_count >= compaction_limit) {
        for (std::size_t i = 0; i < ids.size(); i++) {
            if (tombstones[i] == 1) {
                tombstones.erase(tombstones.begin() + i);
                ids.erase(ids.begin() + i);
            }
        }
        delete_count = 0;
    }

    if (id_map.contains(id)) {
        int idx = id_map[id];
        tombstones[idx] = 1;
        id_map.erase(ids[idx]);
        delete_count++;
        return true;
    }

    return false;
}

std::vector<SearchResult> VectorEngine::search(
    const float *query,
    std::size_t len,
    std::size_t k
) {
    return {};
}
