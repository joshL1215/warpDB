#include "vector_engine.h"

#include <algorithm>

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
    std::vector<std::vector<SearchResult>> batch = search_batch(query, 1, len, k);
    if (batch.empty()) return {};
    return std::move(batch[0]);
}

std::vector<std::vector<SearchResult>> VectorEngine::search_batch(
    const float *queries,
    std::size_t n_queries,
    std::size_t len,
    std::size_t k
) {
    sync_to_gpu();

    // always one (possibly empty) result list per query so callers can index by query
    std::vector<std::vector<SearchResult>> batch_results(n_queries);

    if (queries == nullptr || n_queries == 0 || ids.empty() || k == 0 || len != dimension) {
        return batch_results;
    }

    // the kernel keeps its top-k list in a single warp, so it can return at most MAX_K
    // per query. Clamp here so the host and kernel agree on the output row stride.
    k = std::min(k, MAX_K);

    std::size_t vec_count = ids.size();

    // normalize every query on CPU and pack them row-major: [n_queries x dimension]
    std::vector<float> normed_queries(n_queries * dimension);
    for (std::size_t q = 0; q < n_queries; q++) {
        std::vector<float> normed = l2normed_cpu(queries + q * dimension, dimension);
        std::copy(normed.begin(), normed.end(), normed_queries.begin() + q * dimension);
    }

    float *d_queries = nullptr;
    float *d_top_k_values = nullptr;
    std::uint64_t *d_top_k_indices = nullptr;

    auto free_device = [&]() {
        if (d_queries) cudaFree(d_queries);
        if (d_top_k_values) cudaFree(d_top_k_values);
        if (d_top_k_indices) cudaFree(d_top_k_indices);
    };

    if (cudaMalloc((void**) &d_queries, n_queries * dimension * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &d_top_k_values, n_queries * k * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &d_top_k_indices, n_queries * k * sizeof(std::uint64_t)) != cudaSuccess) {
        free_device();
        return batch_results;
    }

    cudaMemcpy(
        d_queries,
        normed_queries.data(),
        n_queries * dimension * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // one block per query, all queries scan the same device-resident vectors
    launch_cosine_search(
        d_queries,
        d_vectors,
        d_tombstones,
        d_top_k_values,
        d_top_k_indices,
        static_cast<int>(n_queries),
        static_cast<int>(vec_count),
        static_cast<int>(dimension),
        static_cast<int>(k)
    );

    std::vector<float> h_top_k_values(n_queries * k);
    std::vector<std::uint64_t> h_top_k_indices(n_queries * k);

    cudaError_t values_err = cudaMemcpy(
        h_top_k_values.data(),
        d_top_k_values,
        n_queries * k * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaError_t indices_err = cudaMemcpy(
        h_top_k_indices.data(),
        d_top_k_indices,
        n_queries * k * sizeof(std::uint64_t),
        cudaMemcpyDeviceToHost
    );

    free_device();

    if (values_err != cudaSuccess || indices_err != cudaSuccess) {
        return batch_results;
    }

    for (std::size_t q = 0; q < n_queries; q++) {
        std::vector<SearchResult> &results = batch_results[q];
        results.reserve(k);

        for (std::size_t i = 0; i < k; i++) {
            std::uint64_t vec_idx = h_top_k_indices[q * k + i];
            float score = h_top_k_values[q * k + i];

            // empty slots: fewer live vectors than k leaves the kernel's sentinel index / -inf score
            if (vec_idx >= vec_count || tombstones[vec_idx] || std::isinf(score)) continue;

            results.push_back({ids[vec_idx], score});
        }
    }

    return batch_results;
}

void VectorEngine::sync_to_gpu() {
    if (!gpu_dirty) return;

    std::size_t cpu_vec_count = ids.size();

    if (cpu_vec_count == 0) {
        gpu_dirty = false;
        return;
    }

    // if no more room in gpu memory, allocate more
    if (d_vectors == nullptr ||
        d_tombstones == nullptr ||
        cpu_vec_count > gpu_vec_cap) {

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
