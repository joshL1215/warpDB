use axum::{extract::State, Json};

use crate::{
    api::response::ApiResult,
    models::{DeleteRequest, InsertRequest, MessageResponse, SearchRequest, SearchResponse},
    service::VectorService,
};

pub async fn health() -> &'static str {
    "ok"
}

pub async fn insert(State(service): State<VectorService>, Json(request): Json<InsertRequest>) -> ApiResult<MessageResponse> {
    Ok(Json(service.insert(request.id, request.vector)?))
}

pub async fn search(State(service): State<VectorService>, Json(request): Json<SearchRequest>) -> ApiResult<SearchResponse> {
    Ok(Json(service.search(request.query, request.k)?))
}

pub async fn delete_vector(State(service): State<VectorService>, Json(request): Json<DeleteRequest>) -> ApiResult<MessageResponse> {
    Ok(Json(service.delete(&request.id)?))
}
