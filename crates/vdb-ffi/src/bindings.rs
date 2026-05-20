#[repr(C)]
pub struct NativeVectorEngine {
    _private: [u8; 0], // rust short hand for declaring "opaque" struct
}

#[repr(C)]
pub struct NativeSearchResults {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn native_vector_engine_new() -> *mut NativeVectorEngine;
    pub fn native_vector_engine_free(engine: *mut NativeVectorEngine);

    pub fn native_vector_engine_insert( engine: *mut NativeVectorEngine, id: *const std::os::raw::c_char, vector: *const f32, len: usize, );

    pub fn native_vector_engine_delete( engine: *mut NativeVectorEngine, id: *const std::os::raw::c_char, ) -> bool;
    pub fn native_vector_engine_search( engine: *const NativeVectorEngine, query: *const f32, len: usize, k: usize, ) -> *mut NativeSearchResults;
    pub fn native_search_results_len(results: *const NativeSearchResults) -> usize;
    
    pub fn native_search_results_id_at( results: *const NativeSearchResults, index: usize, ) -> *const std::os::raw::c_char;
    pub fn native_search_results_score_at( results: *const NativeSearchResults, index: usize, ) -> f32;
    pub fn native_search_results_free(results: *mut NativeSearchResults);
}
