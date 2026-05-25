#include "vector_engine.h"

VectorEngine::VectorEngine()
    : vectors{}, dimension{768}, ids{}, id_map{},
    tombstones{}, delete_count{}, compaction_limit{25} {
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
    id_map.insert({id, ids.size()});
    ids.push_back(id);
    tombstones.push_back(0);
    vectors.insert(
        vectors.end(),
        vector,
        vector + len
    );
}

bool VectorEngine::erase(std::string id) {
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
    // TODO
    return {};
}
