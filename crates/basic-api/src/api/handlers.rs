use axum::{extract::State, Json};

use crate::{
    api::response::{bad_request, ApiError, ApiResult},
    models::{DeleteRequest, InsertRequest, MessageResponse, SearchRequest, SearchResponse},
    service::VectorService,
    state::AppState
};

pub async fn health() -> &'static str {
    "ok"
}

pub async fn insert(State(state): State<AppState>, Json(request): Json<InsertRequest>) -> ApiResult<MessageResponse> {
    if request.id.trim().is_empty() {
        return Err(bad_request("id must not be empty"));
    }

    validate_vector(&request.vector)?;
    let service = VectorService::new(state);
    Ok(Json(service.insert(request.id, request.vector)?))
}

pub async fn search(State(state): State<AppState>, Json(request): Json<SearchRequest>) -> ApiResult<SearchResponse> {
    if request.k == 0 {
        return Err(bad_request("k must be greater than 0"));
    }

    validate_vector(&request.query)?;
    let service = VectorService::new(state);
    Ok(Json(service.search(request.query, request.k)?))
}

pub async fn delete_vector(State(state): State<AppState>, Json(request): Json<DeleteRequest>) -> ApiResult<MessageResponse> {
    if request.id.trim().is_empty() {
        return Err(bad_request("id must not be empty"));
    }
    let service = VectorService::new(state);
    Ok(Json(service.delete(&request.id)?))
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
