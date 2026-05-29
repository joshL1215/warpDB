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
        self.validate_id(&id)?;
        self.validate_vector(&vector)?;

        let mut engine = self.state.engine.lock().expect("engine lock failed");

        self.validate_dimension(engine.dimension(), vector.len(), "vector dimension does not match existing vectors")?;

        engine.insert(id, vector);

        Ok(MessageResponse {
            message: "vector inserted".to_string(),
        })
    }

    pub fn search(&self, query: Vec<f32>, k: usize) -> Result<SearchResponse, ApiError> {
        self.validate_k(k)?;
        self.validate_vector(&query)?;

        let engine = self.state.engine.lock().expect("engine lock failed");

        self.validate_dimension(engine.dimension(), query.len(), "query dimension does not match stored vectors")?;

        let results = engine.search(query, k);
        Ok(SearchResponse { results })
    }

    pub fn delete(&self, id: &str) -> Result<MessageResponse, ApiError> {
        self.validate_id(id)?;

        let mut engine = self.state.engine.lock().expect("engine lock failed");
        engine.delete(id);

        Ok(MessageResponse {
            message: "vector deleted".to_string(),
        })
    }

    fn validate_id(&self, id: &str) -> Result<(), ApiError> {
        if id.trim().is_empty() {
            return Err(bad_request("id must not be empty"));
        }

        Ok(())
    }

    fn validate_k(&self, k: usize) -> Result<(), ApiError> {
        if k == 0 {
            return Err(bad_request("k must be greater than 0"));
        }

        Ok(())
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), ApiError> {
        if vector.is_empty() {
            return Err(bad_request("vector must not be empty"));
        }

        if vector.iter().any(|value| !value.is_finite()) {
            return Err(bad_request("vector must contain only finite floats"));
        }

        Ok(())
    } 

    fn validate_dimension(&self, expected_dimension: Option<usize>, actual_dimension: usize, message: &str) -> Result<(), ApiError> {
        // Option<type> in Rust always returns None or Some(value), so we strip it
        if let Some(expected_dimension) = expected_dimension {
            if actual_dimension != expected_dimension {
                return Err(bad_request(message));
            }
        }

        Ok(())
    }
}
