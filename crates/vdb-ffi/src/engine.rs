use crate::bindings::{
    native_batch_search_results_free,
    native_batch_search_results_id_at,
    native_batch_search_results_len,
    native_batch_search_results_num_queries,
    native_batch_search_results_score_at,
    native_vector_engine_search_batch,
    NativeBatchSearchResults,
    native_search_results_free,
    native_search_results_id_at,
    native_search_results_len,
    native_search_results_score_at,
    native_vector_engine_delete,
    native_vector_engine_free,
    native_vector_engine_insert,
    native_vector_engine_new,
    native_vector_engine_search,
    NativeSearchResults,
    NativeVectorEngine,
};
use std::ffi::{CStr, CString};

pub struct FfiVectorEngine {
    handle: *mut NativeVectorEngine,
}

pub struct FfiSearchResults {
    handle: *mut NativeSearchResults,
}

pub struct FfiBatchSearchResults {
    handle: *mut NativeBatchSearchResults,
}

// Send is built into Rust as a trait, allows FfiVectorEngine to be moved between threads safely.
unsafe impl Send for FfiVectorEngine {}

impl FfiVectorEngine {
    pub fn new() -> Self {
        let handle = unsafe { native_vector_engine_new() };

        assert!(
            !handle.is_null(),
            "native_vector_engine_new went wrong somewhere"
        );

        Self { 
            handle 
        }
    }

    pub fn insert(&mut self, id: &str, vector: &[f32]) {
        let id = CString::new(id).expect("vector IDs must not contain null bytes");

        unsafe {
            native_vector_engine_insert( self.handle, id.as_ptr(), vector.as_ptr(), vector.len(),);
        }
    }

    pub fn delete(&mut self, id: &str) -> bool {
        let id = CString::new(id).expect("vector IDs must not contain null bytes");
        unsafe { native_vector_engine_delete(self.handle, id.as_ptr()) }
    }

    pub fn search(&self, query: &[f32], k: usize) -> FfiSearchResults {
        let handle = unsafe {
            native_vector_engine_search(self.handle, query.as_ptr(), query.len(), k) };

            assert!(
                !handle.is_null(),
                "native_vector_engine_search returned a null results handle"
        );

        FfiSearchResults { handle }
    }

    /// `queries` is a flat, row-major buffer of `n_queries` vectors, each `dimension` long.
    pub fn search_batch(&self, queries: &[f32], n_queries: usize, dimension: usize, k: usize) -> FfiBatchSearchResults {
        assert_eq!(
            queries.len(),
            n_queries * dimension,
            "batched query buffer length must equal n_queries * dimension"
        );

        let handle = unsafe {
            native_vector_engine_search_batch(self.handle, queries.as_ptr(), n_queries, dimension, k)
        };

        assert!(
            !handle.is_null(),
            "native_vector_engine_search_batch returned a null results handle"
        );

        FfiBatchSearchResults { handle }
    }
}

impl FfiBatchSearchResults {
    pub fn num_queries(&self) -> usize {
        unsafe { native_batch_search_results_num_queries(self.handle) }
    }

    pub fn len(&self, query_index: usize) -> usize {
        assert!(query_index < self.num_queries(), "query index out of bounds");
        unsafe { native_batch_search_results_len(self.handle, query_index) }
    }

    pub fn id_at(&self, query_index: usize, index: usize) -> String {
        assert!(index < self.len(query_index), "result index out of bounds");
        let id_ptr = unsafe { native_batch_search_results_id_at(self.handle, query_index, index) };

        unsafe { CStr::from_ptr(id_ptr) }.to_str().expect("native search result ID must be valid").to_string()
    }

    pub fn score_at(&self, query_index: usize, index: usize) -> f32 {
        assert!(index < self.len(query_index), "result index out of bounds");
        unsafe { native_batch_search_results_score_at(self.handle, query_index, index) }
    }
}

impl Drop for FfiBatchSearchResults {
    fn drop(&mut self) {
        unsafe {
            native_batch_search_results_free(self.handle);
        }
    }
}

impl FfiSearchResults {
    pub fn len(&self) -> usize {
        unsafe { native_search_results_len(self.handle) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn id_at(&self, index: usize) -> String {
        let id_ptr = unsafe { native_search_results_id_at(self.handle, index) };

        unsafe { CStr::from_ptr(id_ptr) }.to_str().expect("native search result ID must be valid").to_string()
        // first converting pointer to rust CStr, then rust string
    }

    pub fn score_at(&self, index: usize) -> f32 {
        unsafe { native_search_results_score_at(self.handle, index) }
    }
}

impl Drop for FfiVectorEngine {
    fn drop(&mut self) {
        unsafe {
            native_vector_engine_free(self.handle);
        }
    }
}

impl Drop for FfiSearchResults {
    fn drop(&mut self) {
        unsafe {
            native_search_results_free(self.handle);
        }
    }
}
