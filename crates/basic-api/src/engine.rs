use std::collections::HashMap;

use crate::models::SearchResult;
use vdb_ffi::FfiVectorEngine;

pub trait VectorEngine {
    fn insert(&mut self, id: String, vector: Vec<f32>);
    fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult>;
    fn delete(&mut self, id: &str);
    fn dimension(&self) -> Option<usize>;
}

pub struct FfiEngineAdapter {
    engine: FfiVectorEngine,
    stored_dimensions: HashMap<String, usize>, // maps vector ID : dimension
    dimension: Option<usize>,
}

impl FfiEngineAdapter {
    pub fn new() -> Self {
        Self {
            engine: FfiVectorEngine::new(),
            stored_dimensions: HashMap::new(),
            dimension: None,
        }
    }
}

impl VectorEngine for FfiEngineAdapter {
    fn insert(&mut self, id: String, vector: Vec<f32>) {
        let vector_dimension = vector.len();

        self.engine.insert(&id, &vector);
        self.stored_dimensions.insert(id, vector_dimension);
        self.dimension = Some(vector_dimension);
    }

    fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult> {
        let results = self.engine.search(&query, k);

        (0..results.len())
            .map(|index| SearchResult {
                id: results.id_at(index),
                score: results.score_at(index),
            })
            .collect()
    }

    fn delete(&mut self, id: &str) {
        if self.engine.delete(id) {
            self.stored_dimensions.remove(id);

            if self.stored_dimensions.is_empty() {
                self.dimension = None;
            }
        }
    }

    fn dimension(&self) -> Option<usize> {
        self.dimension
    }
}


// Old hardcoded implementation minus the FFI
// ------------------------------------------------
// #[derive(Default)]
// pub struct FlatIndex {
//     vectors: Vec<(String, Vec<f32>)>,
// }

// impl FlatIndex {
//     pub fn new() -> Self {
//         Self::default()
//     }
// }
// impl VectorEngine for FlatIndex {
//     fn insert(&mut self, id: String, vector: Vec<f32>) {
//         self.delete(&id);
//         self.vectors.push((id, vector));
//     }

//     fn search(&self, query: Vec<f32>, k: usize) -> Vec<SearchResult> {
//         let mut results: Vec<SearchResult> = self
//             .vectors
//             .iter()
//             .map(|(id, vector)| SearchResult {
//                 id: id.clone(),
//                 score: cosine_similarity(&query, vector),
//             })
//             .collect();

//         results.sort_by(|left, right| right.score.total_cmp(&left.score));
//         results.truncate(k);
//         results
//     }

//     fn delete(&mut self, id: &str) {
//         self.vectors.retain(|(stored_id, _)| stored_id != id);
//     }

//     fn dimension(&self) -> Option<usize> {
//         self.vectors.first().map(|(_, vector)| vector.len())
//     }
// }

// fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
//     let mut dot = 0.0;
//     let mut left_norm = 0.0;
//     let mut right_norm = 0.0;

//     for (left_value, right_value) in left.iter().zip(right.iter()) {
//         dot += left_value * right_value;
//         left_norm += left_value * left_value;
//         right_norm += right_value * right_value;
//     }

//     if left_norm == 0.0 || right_norm == 0.0 {
//         return 0.0;
//     }

//     dot / (left_norm.sqrt() * right_norm.sqrt())
// }