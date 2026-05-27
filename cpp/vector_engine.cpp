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
    : dimension{768}, compaction_limit{25}, gpu_vec_cap{10240} {
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
    gpu_dirty = true;
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
        gpu_dirty = true;
    }

    if (id_map.find(id) != id_map.end()) {
        int idx = id_map[id];
        tombstones[idx] = 1;
        id_map.erase(ids[idx]);
        delete_count++;
        gpu_dirty = true;
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

void VectorEngine::sync_to_gpu() {
    if (!gpu_dirty) return;

    std::size_t cpu_vec_count = ids.size();

    if (cpu_vec_count == 0) {
        gpu_dirty = false;
        return;
    }

    // if no more room in gpu memory, allocate more
    if (cpu_vec_count > gpu_vec_cap) {
        if (d_vectors) cudaFree(d_vectors);
        if (d_tombstones) cudaFree(d_tombstones);

        gpu_vec_cap = std::max(cpu_vec_count, gpu_vec_cap * 2);

        cudaMalloc((void**) &d_vectors, gpu_vec_cap * dimension * sizeof(float));
        cudaMalloc((void**) &d_tombstones, gpu_vec_cap * sizeof(std::uint8_t));
    }

    cudaMemcpy(
        d_vectors,
        vectors.data(),
        cpu_vec_count * dimension * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_tombstones,
        tombstones.data(),
        cpu_vec_count * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice
    );

    gpu_dirty = false;
}

int main() {
    return 0;
}
