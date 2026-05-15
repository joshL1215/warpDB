use axum::{routing::get, Router};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let app = Router::new().route("/health", get(health));
    let listener = TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("failed to bind server");

    println!("listening on http://127.0.0.1:3000");

    axum::serve(listener, app)
        .await
        .expect("server exited unexpectedly");
}

async fn health() -> &'static str {
    "ok"
}
