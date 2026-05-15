use std::sync::{Arc, Mutex};

use crate::engine::FlatIndex;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<FlatIndex>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(FlatIndex::new())),
        }
    }
}
