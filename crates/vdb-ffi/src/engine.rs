use crate::bindings::{
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

// The wrapper owns the native handle and is always accessed behind higher-level
// synchronization in the API layer.
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
