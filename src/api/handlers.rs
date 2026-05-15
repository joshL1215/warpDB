use axum::{extract::State, Json};

use crate::{
    api::response::{bad_request, ApiError, ApiResult},
    engine::VectorEngine,
    models::{
        DeleteRequest, InsertRequest, MessageResponse, SearchRequest, SearchResponse,
    },
    state::AppState,
};

pub async fn health() -> &'static str {
    "ok"
}

pub async fn insert(
    State(state): State<AppState>,
    Json(request): Json<InsertRequest>,
) -> ApiResult<MessageResponse> {
    if request.id.trim().is_empty() {
        return Err(bad_request("id must not be empty"));
    }

    validate_vector(&request.vector)?;

    let mut engine = state.engine.lock().expect("engine mutex poisoned");

    if let Some(expected_dimension) = engine.dimension() {
        if request.vector.len() != expected_dimension {
            return Err(bad_request("vector dimension does not match existing vectors"));
        }
    }

    engine.insert(request.id, request.vector);

    Ok(Json(MessageResponse {
        message: "vector inserted".to_string(),
    }))
}

pub async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> ApiResult<SearchResponse> {
    if request.k == 0 {
        return Err(bad_request("k must be greater than 0"));
    }

    validate_vector(&request.query)?;

    let engine = state.engine.lock().expect("engine mutex poisoned");

    if let Some(expected_dimension) = engine.dimension() {
        if request.query.len() != expected_dimension {
            return Err(bad_request("query dimension does not match stored vectors"));
        }
    }

    let results = engine.search(request.query, request.k);
    Ok(Json(SearchResponse { results }))
}

pub async fn delete_vector(
    State(state): State<AppState>,
    Json(request): Json<DeleteRequest>,
) -> ApiResult<MessageResponse> {
    if request.id.trim().is_empty() {
        return Err(bad_request("id must not be empty"));
    }

    let mut engine = state.engine.lock().expect("engine mutex poisoned");
    engine.delete(&request.id);

    Ok(Json(MessageResponse {
        message: "vector deleted".to_string(),
    }))
}

fn validate_vector(vector: &[f32]) -> Result<(), ApiError> {
    if vector.is_empty() {
        return Err(bad_request("vector must not be empty"));
    }

    if vector.iter().any(|value| !value.is_finite()) {
        return Err(bad_request("vector must contain only finite floats"));
    }

    Ok(())
}
