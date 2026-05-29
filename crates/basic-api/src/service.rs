use crate::{
    api::response::{bad_request, ApiError},
    models::{MessageResponse, SearchResponse},
    state::AppState,
};

#[derive(Clone)]
pub struct VectorService {
    state: AppState,
}

impl VectorService {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    pub fn insert(&self, id: String, vector: Vec<f32>) -> Result<MessageResponse, ApiError> {
        let mut engine = self.state.engine.lock().expect("engine lock failed");

        if let Some(expected_dimension) = engine.dimension() {
            if vector.len() != expected_dimension {
                return Err(bad_request("vector dimension does not match existing vectors"));
            }
        }

        engine.insert(id, vector);

        Ok(MessageResponse {
            message: "vector inserted".to_string(),
        })
    }

    pub fn search(&self, query: Vec<f32>, k: usize) -> Result<SearchResponse, ApiError> {
        let engine = self.state.engine.lock().expect("engine lock failed");

        if let Some(expected_dimension) = engine.dimension() {
            if query.len() != expected_dimension {
                return Err(bad_request("query dimension does not match stored vectors"));
            }
        }

        let results = engine.search(query, k);
        Ok(SearchResponse { results })
    }

    pub fn delete(&self, id: &str) -> Result<MessageResponse, ApiError> {
        let mut engine = self.state.engine.lock().expect("engine lock failed");
        engine.delete(id);

        Ok(MessageResponse {
            message: "vector deleted".to_string(),
        })
    }
}
