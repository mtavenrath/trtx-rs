//! Safe Rust bindings to NVIDIA TensorRT-RTX
//!
//! ⚠️ **EXPERIMENTAL - NOT FOR PRODUCTION USE**
//!
//! This crate is in early experimental development. The API is unstable and will change.
//! This is NOT production-ready software. Use at your own risk.
//!
//! This crate provides safe, ergonomic Rust bindings to the TensorRT-RTX library
//! for high-performance deep learning inference on NVIDIA GPUs.
//!
//! # Overview
//!
//! TensorRT-RTX enables efficient inference by:
//! - Optimizing neural network graphs
//! - Fusing layers and operations
//! - Selecting optimal kernels for your hardware
//! - Supporting dynamic shapes and batching
//!
//! # Workflow
//!
//! Using TensorRT-RTX typically follows two phases:
//!
//! ## Build Phase (Ahead-of-Time)
//!
//! 1. Create a [`Logger`] to capture TensorRT messages
//! 2. Create a [`Builder`] to construct an optimized engine
//! 3. Define your network using [`NetworkDefinition`]
//! 4. Configure optimization with [`BuilderConfig`]
//! 5. Build and serialize the engine to disk
//!
//! ## Inference Phase (Runtime)
//!
//! 1. Create a [`Runtime`] with a logger
//! 2. Deserialize the engine using [`Runtime::deserialize_cuda_engine`]
//! 3. Create an [`ExecutionContext`] from the engine
//! 4. Bind input/output tensors
//! 5. Execute inference with [`ExecutionContext::enqueue_v3`]
//!
//! # Example
//!
//! ```rust,no_run
//! use trtx::{Logger, Builder, Runtime};
//! use trtx::builder::{BuilderConfig, MemoryPoolType, network_flags};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create logger
//! let logger = Logger::stderr()?;
//!
//! // Build phase
//! let builder = Builder::new(&logger)?;
//! let mut network = builder.create_network(network_flags::EXPLICIT_BATCH)?;
//! let mut config = builder.create_config()?;
//!
//! // Configure memory
//! config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?;
//!
//! // Build and serialize
//! let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
//! std::fs::write("model.engine", &engine_data)?;
//!
//! // Inference phase
//! let runtime = Runtime::new(&logger)?;
//! let engine = runtime.deserialize_cuda_engine(&engine_data)?;
//! let context = engine.create_execution_context()?;
//!
//! // List I/O tensors
//! let num_tensors = engine.get_nb_io_tensors()?;
//! for i in 0..num_tensors {
//!     let name = engine.get_tensor_name(i)?;
//!     println!("Tensor {}: {}", i, name);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Safety
//!
//! This crate provides safe abstractions over the underlying C++ API. However,
//! some operations (like setting tensor addresses and enqueueing inference)
//! require careful management of CUDA memory and are marked as `unsafe`.
//!
//! # Prerequisites
//!
//! - NVIDIA TensorRT-RTX library installed
//! - CUDA Runtime
//! - Compatible NVIDIA GPU
//!
//! Set the `TENSORRT_RTX_DIR` environment variable to the installation path
//! if TensorRT-RTX is not in a standard location.

// Allow unnecessary casts - they're needed for real mode (u32) but not mock mode (i32)
#![cfg_attr(feature = "mock", allow(clippy::unnecessary_cast))]

// Mock mode - pure Rust implementations that return errors
#[cfg(feature = "mock")]
mod mock;

// Real TensorRT mode - FFI bindings (default)
#[cfg(not(feature = "mock"))]
pub mod builder;
#[cfg(not(feature = "mock"))]
pub mod cuda;
pub mod error;
#[cfg(not(feature = "mock"))]
pub mod executor;
#[cfg(not(feature = "mock"))]
pub mod logger;
#[cfg(not(feature = "mock"))]
pub mod network;
#[cfg(not(feature = "mock"))]
pub mod onnx_parser;
#[cfg(not(feature = "mock"))]
pub mod runtime;

// Re-export commonly used types

// Real TensorRT mode - re-export from real modules (default)
#[cfg(not(feature = "mock"))]
pub use builder::{Builder, BuilderConfig};
#[cfg(not(feature = "mock"))]
pub use network::{NetworkDefinition, Tensor};
#[cfg(not(feature = "mock"))]
pub use cuda::{synchronize, DeviceBuffer};
#[cfg(not(feature = "mock"))]
pub use executor::{run_onnx_with_tensorrt, run_onnx_zeroed, TensorInput, TensorOutput};
#[cfg(not(feature = "mock"))]
pub use logger::{LogHandler, Logger, Severity, StderrLogger};
#[cfg(not(feature = "mock"))]
pub use onnx_parser::OnnxParser;
#[cfg(not(feature = "mock"))]
pub use runtime::{CudaEngine, ExecutionContext, Runtime};

// Mock mode - re-export from mock module
#[cfg(feature = "mock")]
pub use mock::{
    Builder, BuilderConfig, CudaEngine, DeviceBuffer, ExecutionContext, LogHandler, Logger,
    MemoryPoolType, NetworkDefinition, OnnxParser, Runtime, Severity, StderrLogger, Tensor,
    TensorInput, TensorOutput, get_default_stream, network_flags, run_onnx_with_tensorrt,
    run_onnx_zeroed, synchronize,
};

// For mock mode, also need builder submodule for network_flags
#[cfg(feature = "mock")]
pub mod builder {
    pub use crate::mock::{network_flags, BuilderConfig, MemoryPoolType};
}

// Error types are always available (not feature-dependent)
pub use error::{Error, Result};
