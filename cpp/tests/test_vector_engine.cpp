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

static void test_search_returns_nearest_synthetic_vector() {
    VectorEngine engine;

    std::vector<float> doc_x = make_vector(0.0f);
    std::vector<float> doc_y = make_vector(0.0f);
    std::vector<float> query = make_vector(0.0f);

    doc_x[0] = 1.0f;
    doc_y[1] = 1.0f;
    query[0] = 1.0f;

    engine.insert("doc-x", doc_x.data(), doc_x.size());
    engine.insert("doc-y", doc_y.data(), doc_y.size());

    std::vector<SearchResult> results = engine.search(query.data(), query.size(), 1);

    CHECK(results.size() == 1);
    CHECK(results[0].id == "doc-x");
    CHECK_CLOSE(results[0].score, 1.0f);
}

static void test_search_batch_returns_per_query_results_in_order() {
    VectorEngine engine;

    std::vector<float> doc_x = make_vector(0.0f);
    std::vector<float> doc_y = make_vector(0.0f);
    std::vector<float> doc_z = make_vector(0.0f);
    doc_x[0] = 1.0f;
    doc_y[1] = 1.0f;
    doc_z[2] = 1.0f;

    engine.insert("doc-x", doc_x.data(), doc_x.size());
    engine.insert("doc-y", doc_y.data(), doc_y.size());
    engine.insert("doc-z", doc_z.data(), doc_z.size());

    // three queries packed row-major, each pointing at a different doc
    std::vector<float> queries(3 * 768, 0.0f);
    queries[0 * 768 + 2] = 1.0f; // -> doc-z
    queries[1 * 768 + 0] = 1.0f; // -> doc-x
    queries[2 * 768 + 1] = 1.0f; // -> doc-y

    auto batch = engine.search_batch(queries.data(), 3, 768, 1);

    CHECK(batch.size() == 3);
    CHECK(batch[0].size() == 1 && batch[0][0].id == "doc-z");
    CHECK(batch[1].size() == 1 && batch[1][0].id == "doc-x");
    CHECK(batch[2].size() == 1 && batch[2][0].id == "doc-y");
    CHECK_CLOSE(batch[1][0].score, 1.0f);
}

static void test_search_batch_skips_empty_slots_when_k_exceeds_live_vectors() {
    VectorEngine engine;

    std::vector<float> doc_x = make_vector(0.0f);
    std::vector<float> doc_y = make_vector(0.0f);
    doc_x[0] = 1.0f;
    doc_y[1] = 1.0f;
    engine.insert("doc-x", doc_x.data(), doc_x.size());
    engine.insert("doc-y", doc_y.data(), doc_y.size());
    CHECK(engine.erase("doc-y"));

    std::vector<float> queries(2 * 768, 0.0f);
    queries[0] = 1.0f;
    queries[768 + 1] = 1.0f;

    auto batch = engine.search_batch(queries.data(), 2, 768, 10);

    CHECK(batch.size() == 2);
    CHECK(batch[0].size() == 1 && batch[0][0].id == "doc-x");
    CHECK(batch[1].size() == 1 && batch[1][0].id == "doc-x"); // doc-y is tombstoned
}

// more queries than one kernel block handles (8), and more stored vectors than fit in a uint8 index
static void test_search_batch_large_batch_across_blocks() {
    VectorEngine engine;
    const std::size_t n_docs = 300;
    const std::size_t n_queries = 20;

    for (std::size_t i = 0; i < n_docs; i++) {
        std::vector<float> doc = make_vector(0.0f);
        doc[i % 768] = 1.0f;
        engine.insert("doc-" + std::to_string(i), doc.data(), doc.size());
    }

    // query q targets doc 280 + q, so every match index is > 255
    std::vector<float> queries(n_queries * 768, 0.0f);
    for (std::size_t q = 0; q < n_queries; q++) {
        queries[q * 768 + (280 + q)] = 1.0f;
    }

    auto batch = engine.search_batch(queries.data(), n_queries, 768, 1);

    CHECK(batch.size() == n_queries);
    for (std::size_t q = 0; q < n_queries; q++) {
        CHECK(batch[q].size() == 1);
        CHECK(batch[q][0].id == "doc-" + std::to_string(280 + q));
        CHECK_CLOSE(batch[q][0].score, 1.0f);
    }
}

static void test_search_batch_wrong_dimension_returns_empty_lists() {
    VectorEngine engine;
    std::vector<float> doc = make_vector(1.0f);
    engine.insert("doc", doc.data(), doc.size());

    std::vector<float> queries(2 * 3, 1.0f);
    auto batch = engine.search_batch(queries.data(), 2, 3, 1);

    CHECK(batch.size() == 2);
    CHECK(batch[0].empty() && batch[1].empty());
}

int main() {
    test_search_result_holds_id_and_score();
    test_new_engine_searches_empty();
    test_delete_missing_id_returns_false();
    test_insert_wrong_dimension_is_ignored();
    test_insert_stores_l2_normalized_vector();
    test_insert_then_delete_existing_id();
    test_search_returns_nearest_synthetic_vector();
    test_search_batch_returns_per_query_results_in_order();
    test_search_batch_skips_empty_slots_when_k_exceeds_live_vectors();
    test_search_batch_large_batch_across_blocks();
    test_search_batch_wrong_dimension_returns_empty_lists();

    std::cout << "All vector_engine tests passed\n";
    return 0;
}
