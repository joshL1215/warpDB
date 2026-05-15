use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: String,
    pub vector: Vec<f32>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    pub k: usize,
}

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub id: String,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Serialize)]
pub struct MessageResponse {
    pub message: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
