mod api;
mod engine;
mod models;
mod service;
mod state;

use axum::{routing::{get, post}, Router};
use tokio::net::TcpListener;

use crate::{api::handlers, state::AppState};

#[tokio::main]
async fn main() {
    let state = AppState::new();

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/insert", post(handlers::insert))
        .route("/search", post(handlers::search))
        .route("/delete", post(handlers::delete_vector))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("failed to bind server");

    println!("listening on http://127.0.0.1:3000");

    axum::serve(listener, app)
        .await
        .expect("server exited unexpectedly");
}
