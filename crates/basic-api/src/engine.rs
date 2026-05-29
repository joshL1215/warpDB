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
    stored_dimensions: HashMap<String, usize>,
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

        (0..results.len()) // creating range from 0 to len - 1
            .map(|index| SearchResult { // for every number, build SearchResult
                id: results.id_at(index),
                score: results.score_at(index),
            })
            .collect()  // return vector of SearchResults
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
