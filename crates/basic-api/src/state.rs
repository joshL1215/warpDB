use std::sync::{Arc, Mutex};

use crate::engine::{FlatIndex, VectorEngine};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Box<dyn VectorEngine + Send>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(Box::new(FlatIndex::new()))),
        }
    }
}
