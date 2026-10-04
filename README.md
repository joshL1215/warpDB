# warpDB

A vector store with a Rust API, C++ storage engine, and CUDA cosine-similarity search. Its search kernel scans stored vectors and maintains a WarpSelect top-k list within a GPU warp.


## Repository map

| Path | Responsibility |
| --- | --- |
| [`crates/basic-api`](crates/basic-api/src/main.rs) | Axum/Tokio server: routes, JSON models, validation, shared state, and engine adapter. |
| [`crates/vdb-ffi`](crates/vdb-ffi/src/engine.rs) | Rust wrappers for FFI using generated `bindgen` declarations and C++ compilation. |
| [`cpp`](cpp/vector_engine.cpp) | Normalized vector storage, ID lookup, tombstones, GPU synchronization, and search logic. |
| [`kernels`](kernels/cosine_search.cu) | Cosine/top-k search, standalone L2 normalization and dot products, kernel tests, and benchmark source. |
| [`cpp/tests`](cpp/tests/Makefile) | Native engine tests and a compilation script. |

## Architecture

```mermaid
flowchart TD
    Client[HTTP client] --> Router[Axum router]
    Router --> Health[GET /health: plain text ok]
    Router --> Handlers[JSON handlers: insert, search, delete]
    Handlers --> Service[VectorService: validation and orchestration]
    Service --> State[AppState: Arc + Mutex + boxed VectorEngine]
    State --> Adapter[FfiEngineAdapter: Rust dimension tracking]
    Adapter --> FFI[vdb-ffi: Rust handle wrappers]
    FFI --> ABI[C-linkage native functions]
    ABI --> Native[C++ VectorEngine]
    Native --> CPU[CPU vectors, IDs, ID map, tombstones]
    Native --> GPU[CUDA device buffers]
    GPU --> Kernel[Fused cosine similarity and top-k kernel]
```

Startup creates one engine and binds to **`127.0.0.1:3000`**.

Insert, search, and delete each hold the same `std::sync::Mutex` while invoking the engine.

### Validation and errors


```mermaid
flowchart TD
    Request[POST request] --> Extract{JSON extraction succeeds?}
    Extract -->|No| Rejection[Axum rejection]
    Extract -->|Yes| Handler[Route handler]
    Handler --> Validate{ID, vector, or k valid?}
    Validate -->|No| Bad[400 JSON error]
    Validate -->|Yes| Lock[Lock shared engine]
    Lock --> Operation{Operation}
    Operation -->|Insert or search| Dimension{Matches tracked dimension, if set?}
    Dimension -->|No| Bad
    Dimension -->|Yes| Call[Call adapter and FFI]
    Operation -->|Delete| Call
    Call --> Response[200 JSON response]
```



## Storage and request flows

### Insert and delete

```mermaid
flowchart TD
    Insert[Validated insert under mutex] --> Size{Native length is 768?}
    Size -->|No| Ignore[Native insert silently returns]
    Size -->|Yes| Normalize[CPU L2 normalization]
    Normalize --> Append[Append vector, ID, tombstone; insert ID mapping]
    Append --> Dirty[Mark GPU buffers dirty]
    Ignore --> Track[Adapter records ID length and dimension]
    Dirty --> Track
    Track --> InsertOK[200: vector inserted]

    Delete[Validated delete under mutex] --> Compact{25 deletions already accumulated?}
    Compact -->|Yes| Attempt[Attempt compaction; reset count; mark dirty]
    Compact -->|No| Lookup{ID exists in native map?}
    Attempt --> Lookup
    Lookup -->|Yes| Tombstone[Mark row deleted; remove mapping; increment count; mark dirty]
    Tombstone --> Metadata[Remove Rust ID metadata; reset dimension if empty]
    Lookup -->|No| DeleteOK[200: vector deleted]
    Metadata --> DeleteOK
```

Duplicate insertion appends another row while preserving the original ID mapping, so deleting that ID tombstones only the mapped row. 


## Development layout

`basic-api` depends on `vdb-ffi` by path. The FFI build script uses `cc` to compile C++ and `bindgen` to generate Rust declarations from `vector_engine_ffi.h`.

The API entry point is `crates/basic-api/src/main.rs`. Running the complete stack requires an NVIDIA CUDA-capable GPU, CUDA toolkit/runtime, a compatible C++17 compiler, Rust/Cargo, and Clang/libclang.