# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

trtx-rs provides safe Rust bindings to NVIDIA TensorRT-RTX for high-performance deep learning inference. The project is **experimental** and uses a workspace with two crates:

- **trtx-sys**: Raw FFI bindings (unsafe, internal use only)
- **trtx**: Safe Rust wrapper (user-facing API)

## Build Commands

### Development (Mock Mode - No GPU Required)

Mock mode allows development without TensorRT-RTX installed:

```bash
# Build
make build                    # Debug build with mock feature
make build-release            # Release build with mock feature

# Test
make test                     # Run all tests with mock
make test-mock               # Verbose test output

# Single test
cargo test --features mock test_name

# Run example
cargo run --features mock --example rustnn_executor
```

### Production (Real TensorRT-RTX)

```bash
# Set environment
export TENSORRT_RTX_DIR=/path/to/tensorrt-rtx
export CUDA_ROOT=/usr/local/cuda

# Build without mock
cargo build --release
cargo test
```

### Code Quality

```bash
make fmt                     # Format code
make fmt-check              # Check formatting
make clippy                 # Run lints (-D warnings)
make clippy-fix             # Auto-fix warnings
make check-all              # Run fmt-check + clippy + test
```

### Publishing

```bash
make package                # Package both crates
make publish-dry            # Dry-run publish
make publish                # Publish to crates.io
```

## Architecture

### Hybrid FFI Design (autocxx + Minimal C Wrapper)

```
┌─────────────────────────┐
│  Rust Safe API (trtx)   │  <- RAII, Result<T, Error>, lifetimes
├─────────────────────────┤
│  Raw FFI (trtx-sys)     │  <- autocxx-generated + logger bridge
├─────────────────────────┤
│  Logger Bridge (C++)    │  <- logger_bridge.hpp/cpp (callbacks only)
├─────────────────────────┤
│  TensorRT-RTX C++ API   │  <- NVIDIA library (direct via autocxx)
└─────────────────────────┘
```

**Why hybrid approach?**
- **autocxx** provides direct C++ bindings for most TensorRT classes
- **Minimal C wrapper** handles Logger callbacks (virtual methods)
- autocxx automatically handles C++ exceptions and memory management
- Reduces manual FFI code maintenance
- Better type safety with C++ templates and classes

**Key components:**
- `trtx-sys/src/lib.rs`: autocxx include_cpp! macro for TensorRT bindings
- `trtx-sys/logger_bridge.cpp`: Minimal C wrapper for Logger callbacks
- `trtx-sys/build.rs`: Builds both autocxx bindings and logger bridge

See `docs/FFI_GUIDE.md` for detailed FFI development workflow.

### Two-Phase Workflow

**Build Phase (AOT):**
```
Logger → Builder → NetworkDefinition → BuilderConfig → SerializedEngine
```

**Inference Phase (Runtime):**
```
Runtime → Deserialize Engine → ExecutionContext → Bind Tensors → Execute
```

### Core Modules

- `trtx/src/logger.rs`: Logger abstraction with custom handlers
- `trtx/src/builder.rs`: Builder, BuilderConfig, NetworkDefinition
- `trtx/src/runtime.rs`: Runtime, CudaEngine, ExecutionContext
- `trtx/src/onnx_parser.rs`: ONNX model parsing
- `trtx/src/cuda.rs`: CUDA memory management wrappers
- `trtx/src/executor.rs`: High-level executor API (rustnn-compatible)
- `trtx/src/error.rs`: Error types

## FFI Bindings

### When Modifying FFI

1. **Update autocxx bindings** in `trtx-sys/src/lib.rs` (add new `generate!()` calls)
2. **Rebuild** to regenerate bindings: `cargo clean -p trtx-sys && cargo build`
3. **Update mock** in `trtx-sys/build.rs` (`generate_mock_bindings`) and `trtx-sys/mock.c` if needed
4. **Add safe wrapper** in appropriate `trtx/src/*.rs` file

### Adding New TensorRT Classes

To expose a new TensorRT class via autocxx:

```rust
// In trtx-sys/src/lib.rs, inside include_cpp! block:
generate!("nvinfer1::NewClassName")
generate!("nvinfer1::newFactoryFunction")
```

### Logger Bridge (Special Case)

The Logger uses a minimal C wrapper for callbacks:
- `trtx-sys/logger_bridge.hpp`: Header with C interface
- `trtx-sys/logger_bridge.cpp`: Implementation bridging Rust callbacks to C++
- Rust callbacks defined in `trtx/src/logger.rs`

### Error Handling

- **autocxx mode**: C++ exceptions are caught by autocxx, check return values and null pointers
- **Mock mode**: Uses old error pattern with error_msg buffers for compatibility

## Mock Mode

Mock mode is **critical** for development. It provides stub implementations allowing:
- Development on machines without TensorRT-RTX (e.g., macOS)
- CI/CD on any platform
- API validation without GPU

**Always test with mock:** When adding new FFI functions, update both real bindings AND mock implementations.

### Mock Implementation Files

- `trtx-sys/build.rs`: `generate_mock_bindings()` function defines Rust FFI stubs
- `trtx-sys/mock.c`: C implementations that return `TRTX_SUCCESS` with dummy data

## Important Notes

### Build System

- `trtx-sys/build.rs` uses autocxx-build to generate C++ bindings from `src/lib.rs`
- Also compiles `logger_bridge.cpp` using cc crate
- Generated files: `target/debug/build/trtx-sys-*/out/` (autocxx-generated)
- Manual edits to generated files are **overwritten** on rebuild
- Changes must go in source files (`src/lib.rs`, `logger_bridge.cpp`, `build.rs`)

### Memory Management

- **RAII everywhere**: Use `Drop` trait for automatic cleanup
- **Opaque pointers**: C++ objects exposed as opaque types in Rust
- **CUDA memory**: Wrapped in `DeviceBuffer` with automatic deallocation
- **Error buffers**: Allocated by Rust, passed to C, freed by Rust

### Safety

- Most operations are safe due to RAII
- CUDA operations require `unsafe`:
  - `ExecutionContext::set_tensor_address()` - must point to valid CUDA memory
  - `ExecutionContext::enqueue_v3()` - requires valid CUDA stream
  - CUDA memory operations - explicit device buffer management

### Workspace Dependencies

- Workspace version: 0.2.0
- `trtx-sys` has no dependencies (raw FFI only)
- `trtx` depends on `trtx-sys` and `thiserror`
- Both crates use workspace-level version, edition, license, authors

## Testing Strategy

- Unit tests with mock mode for API validation
- Integration tests for full workflows
- GPU tests on self-hosted runner (Windows + T4 GPU)
- Examples demonstrate real usage patterns

## Environment Variables

- `TENSORRT_RTX_DIR`: TensorRT-RTX installation path (default: `/usr/local/tensorrt-rtx`)
- `CUDA_ROOT`: CUDA installation path (default: `/usr/local/cuda`)
- `LIBCLANG_PATH`: Path to libclang for bindgen (if needed)

## Resources

- Design documentation: `docs/DESIGN.md`
- FFI guide: `docs/FFI_GUIDE.md`
- Integration notes: `docs/RUSTNN_INTEGRATION.md`
- Windows GPU setup: `docs/WINDOWS_GPU_SETUP.md`
- Pre-commit hooks: `.githooks/` (run `cp .githooks/pre-commit .git/hooks/pre-commit`)
