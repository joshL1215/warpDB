#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define private public
#include "../vector_engine.h"
#undef private

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ \
                      << ": " << #cond << "\n"; \
            std::exit(1); \
        } \
    } while (0)

#define CHECK_CLOSE(left, right) \
    CHECK(std::fabs((left) - (right)) < 1e-6f)

static std::vector<float> make_vector(float value) {
    return std::vector<float>(768, value);
}

static void test_search_result_holds_id_and_score() {
    SearchResult result{"doc-1", 0.75f};

    CHECK(result.id == "doc-1");
    CHECK(std::fabs(result.score - 0.75f) < 1e-6f);
}

static void test_new_engine_searches_empty() {
    VectorEngine engine;
    std::vector<float> query = make_vector(1.0f);

    std::vector<SearchResult> results = engine.search(query.data(), query.size(), 10);

    CHECK(results.empty());
}

static void test_delete_missing_id_returns_false() {
    VectorEngine engine;

    CHECK(!engine.erase("missing"));
}

static void test_insert_wrong_dimension_is_ignored() {
    VectorEngine engine;
    std::vector<float> vector(3, 1.0f);

    engine.insert("bad-dimension", vector.data(), vector.size());

    CHECK(engine.vectors.empty());
    CHECK(!engine.erase("bad-dimension"));
}

static void test_insert_stores_l2_normalized_vector() {
    VectorEngine engine;
    std::vector<float> vector = make_vector(0.0f);
    vector[0] = 3.0f;
    vector[1] = 4.0f;

    engine.insert("doc-1", vector.data(), vector.size());

    CHECK(engine.vectors.size() == 768);
    CHECK_CLOSE(engine.vectors[0], 0.6f);
    CHECK_CLOSE(engine.vectors[1], 0.8f);
    CHECK_CLOSE(engine.vectors[2], 0.0f);

    float norm = 0.0f;
    for (float value : engine.vectors) {
        norm += value * value;
    }
    CHECK_CLOSE(std::sqrt(norm), 1.0f);
}

static void test_insert_then_delete_existing_id() {
    VectorEngine engine;
    std::vector<float> vector = make_vector(1.0f);

    engine.insert("doc-1", vector.data(), vector.size());

    CHECK(engine.erase("doc-1"));
    CHECK(!engine.erase("doc-1"));
}

int main() {
    test_search_result_holds_id_and_score();
    test_new_engine_searches_empty();
    test_delete_missing_id_returns_false();
    test_insert_wrong_dimension_is_ignored();
    test_insert_stores_l2_normalized_vector();
    test_insert_then_delete_existing_id();

    std::cout << "All vector_engine tests passed\n";
    return 0;
}
