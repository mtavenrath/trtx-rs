//! Mock implementations for trtx without TensorRT-RTX
//!
//! These implementations allow the crate to build and type-check without TensorRT,
//! but return errors at runtime. The Logger entry point returns a clear error message,
//! and all other functions are unreachable since they require a valid Logger to be created first.

use crate::error::{Error, Result};
use std::marker::PhantomData;

// ============================================================================
// Logger - Entry Point (Returns Error)
// ============================================================================

/// Logger trait for handling log messages
pub trait LogHandler: Send + Sync {
    fn log(&self, severity: Severity, message: &str);
}

/// Severity levels for log messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Severity {
    InternalError = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Verbose = 4,
}

/// Default stderr logger
#[derive(Debug)]
pub struct StderrLogger;

impl LogHandler for StderrLogger {
    fn log(&self, severity: Severity, message: &str) {
        eprintln!("[TensorRT {:?}] {}", severity, message);
    }
}

/// Logger (mock - always returns error)
pub struct Logger {
    _private: (),
}

impl Logger {
    /// Create a new logger with a custom handler
    /// 
    /// # Mock Mode
    /// 
    /// This function returns an error in mock mode indicating that TensorRT-RTX is not available.
    pub fn new<H: LogHandler + 'static>(_handler: H) -> Result<Self> {
        Err(Error::Runtime(
            "TensorRT-RTX not available. This binary was built with the 'mock' feature.\n\
             \n\
             To use real TensorRT:\n\
             1. Install TensorRT-RTX and set TENSORRT_RTX_DIR environment variable\n\
             2. Rebuild without mock: cargo build --no-default-features\n\
             \n\
             Or use from workspace root: make build-real"
                .to_string(),
        ))
    }

    /// Create a logger that prints to stderr
    ///
    /// # Mock Mode
    ///
    /// This function returns an error in mock mode indicating that TensorRT-RTX is not available.
    pub fn stderr() -> Result<Self> {
        Self::new(StderrLogger)
    }
}

// Logger must be Send and Sync
unsafe impl Send for Logger {}
unsafe impl Sync for Logger {}

// ============================================================================
// Builder - Unreachable (Requires Logger)
// ============================================================================

/// Network definition builder flags
pub mod network_flags {
    /// Explicit batch sizes
    pub const EXPLICIT_BATCH: u32 = 1 << 0;
}

/// Memory pool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemoryPoolType {
    /// Workspace memory
    Workspace = 0,
    /// DLA managed SRAM
    DlaManagedSram = 1,
    /// DLA local DRAM
    DlaLocalDram = 2,
    /// DLA global DRAM
    DlaGlobalDram = 3,
}

/// Builder configuration
pub struct BuilderConfig {
    _private: (),
}

impl BuilderConfig {
    /// Set memory pool limit
    pub fn set_memory_pool_limit(&mut self, _pool: MemoryPoolType, _size: usize) -> Result<()> {
        unreachable!("BuilderConfig cannot be created in mock mode - Logger initialization failed")
    }
}

/// Builder for creating optimized TensorRT engines
pub struct Builder<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Builder<'a> {
    /// Create a new builder
    pub fn new(_logger: &'a Logger) -> Result<Self> {
        unreachable!("Builder::new cannot be called - Logger initialization failed in mock mode")
    }

    /// Create a network definition
    pub fn create_network(&self, _flags: u32) -> Result<NetworkDefinition> {
        unreachable!("Builder::create_network cannot be called in mock mode")
    }

    /// Create a builder configuration
    pub fn create_config(&self) -> Result<BuilderConfig> {
        unreachable!("Builder::create_config cannot be called in mock mode")
    }

    /// Build a serialized network (engine)
    pub fn build_serialized_network(
        &self,
        _network: &mut NetworkDefinition,
        _config: &mut BuilderConfig,
    ) -> Result<Vec<u8>> {
        unreachable!("Builder::build_serialized_network cannot be called in mock mode")
    }
}

unsafe impl Send for Builder<'_> {}

// ============================================================================
// Runtime - Unreachable (Requires Logger)
// ============================================================================

/// A CUDA engine containing optimized inference code
pub struct CudaEngine {
    _private: (),
}

impl CudaEngine {
    /// Get the number of I/O tensors
    pub fn get_nb_io_tensors(&self) -> Result<i32> {
        unreachable!("CudaEngine cannot be created in mock mode")
    }

    /// Get the name of a tensor by index
    pub fn get_tensor_name(&self, _index: i32) -> Result<String> {
        unreachable!("CudaEngine cannot be created in mock mode")
    }

    /// Create an execution context for inference
    pub fn create_execution_context(&self) -> Result<ExecutionContext<'_>> {
        unreachable!("CudaEngine cannot be created in mock mode")
    }
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

/// Execution context for running inference
pub struct ExecutionContext<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> ExecutionContext<'a> {
    /// Set the address of a tensor for input or output
    pub unsafe fn set_tensor_address(
        &mut self,
        _name: &str,
        _data: *mut std::ffi::c_void,
    ) -> Result<()> {
        unreachable!("ExecutionContext cannot be created in mock mode")
    }

    /// Enqueue inference work on a CUDA stream
    pub unsafe fn enqueue_v3(&mut self, _cuda_stream: *mut std::ffi::c_void) -> Result<()> {
        unreachable!("ExecutionContext cannot be created in mock mode")
    }
}

unsafe impl Send for ExecutionContext<'_> {}

/// Runtime for deserializing engines
pub struct Runtime<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Runtime<'a> {
    /// Create a new runtime
    pub fn new(_logger: &'a Logger) -> Result<Self> {
        unreachable!("Runtime::new cannot be called - Logger initialization failed in mock mode")
    }

    /// Deserialize a CUDA engine from serialized data
    pub fn deserialize_cuda_engine(&self, _data: &[u8]) -> Result<CudaEngine> {
        unreachable!("Runtime::deserialize_cuda_engine cannot be called in mock mode")
    }
}

unsafe impl Send for Runtime<'_> {}

// ============================================================================
// Network - Unreachable (Requires Builder)
// ============================================================================

/// Tensor handle
pub struct Tensor {
    _private: (),
}

impl Tensor {
    /// Get the tensor name
    pub fn name(&self) -> Result<String> {
        unreachable!("Tensor cannot be created in mock mode")
    }

    /// Set the tensor name
    pub fn set_name(&mut self, _name: &str) -> Result<()> {
        unreachable!("Tensor cannot be created in mock mode")
    }

    /// Get tensor dimensions
    pub fn dimensions(&self) -> Result<Vec<i32>> {
        unreachable!("Tensor cannot be created in mock mode")
    }

    /// Get tensor data type
    pub fn get_type(&self) -> Result<i32> {
        unreachable!("Tensor cannot be created in mock mode")
    }
}

/// Network definition for building TensorRT engines
pub struct NetworkDefinition {
    _private: (),
}

impl NetworkDefinition {
    /// Add an input tensor to the network
    pub fn add_input(&mut self, _name: &str, _data_type: i32, _dims: &[i32]) -> Result<Tensor> {
        unreachable!("NetworkDefinition cannot be created in mock mode")
    }

    /// Mark a tensor as a network output
    pub fn mark_output(&mut self, _tensor: &Tensor) -> Result<()> {
        unreachable!("NetworkDefinition cannot be created in mock mode")
    }

    /// Get the number of inputs
    pub fn get_nb_inputs(&self) -> i32 {
        unreachable!("NetworkDefinition cannot be created in mock mode")
    }

    /// Get the number of outputs
    pub fn get_nb_outputs(&self) -> i32 {
        unreachable!("NetworkDefinition cannot be created in mock mode")
    }

    /// Get an input tensor by index
    pub fn get_input(&self, _index: i32) -> Result<Tensor> {
        unreachable!("NetworkDefinition cannot be created in mock mode")
    }

    /// Get an output tensor by index
    pub fn get_output(&self, _index: i32) -> Result<Tensor> {
        unreachable!("NetworkDefinition cannot be created in mock mode")
    }
}

// ============================================================================
// CUDA - Unreachable
// ============================================================================

/// RAII wrapper for CUDA device memory
pub struct DeviceBuffer {
    _private: (),
}

impl DeviceBuffer {
    /// Allocate CUDA device memory
    pub fn new(_size: usize) -> Result<Self> {
        unreachable!("DeviceBuffer cannot be created in mock mode - CUDA not available")
    }

    /// Get the raw device pointer
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        unreachable!("DeviceBuffer cannot be created in mock mode")
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        unreachable!("DeviceBuffer cannot be created in mock mode")
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, _data: &[u8]) -> Result<()> {
        unreachable!("DeviceBuffer cannot be created in mock mode")
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, _data: &mut [u8]) -> Result<()> {
        unreachable!("DeviceBuffer cannot be created in mock mode")
    }
}

unsafe impl Send for DeviceBuffer {}

/// Synchronize CUDA device
pub fn synchronize() -> Result<()> {
    unreachable!("CUDA synchronize cannot be called in mock mode - CUDA not available")
}

/// Get the default CUDA stream
pub fn get_default_stream() -> *mut std::ffi::c_void {
    unreachable!("CUDA get_default_stream cannot be called in mock mode - CUDA not available")
}

// ============================================================================
// ONNX Parser - Unreachable (Requires Logger)
// ============================================================================

/// ONNX model parser
pub struct OnnxParser {
    _private: (),
}

impl OnnxParser {
    /// Create a new ONNX parser for the given network
    pub fn new(_network: &mut NetworkDefinition, _logger: &Logger) -> Result<Self> {
        unreachable!("OnnxParser::new cannot be called - Logger initialization failed in mock mode")
    }

    /// Parse an ONNX model from bytes
    pub fn parse(&self, _model_bytes: &[u8]) -> Result<()> {
        unreachable!("OnnxParser::parse cannot be called in mock mode")
    }
}

unsafe impl Send for OnnxParser {}

// ============================================================================
// Executor - High-level API (Also Unreachable)
// ============================================================================

/// Input descriptor for TensorRT execution
#[derive(Debug, Clone)]
pub struct TensorInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Output descriptor from TensorRT execution
#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Execute an ONNX model with TensorRT using provided inputs
///
/// # Mock Mode
///
/// This function returns an error in mock mode since TensorRT is not available.
pub fn run_onnx_with_tensorrt(
    _onnx_model_bytes: &[u8],
    _inputs: &[TensorInput],
) -> Result<Vec<TensorOutput>> {
    Err(Error::Runtime(
        "TensorRT-RTX not available in mock mode. Rebuild without mock: cargo build --no-default-features"
            .to_string(),
    ))
}

/// Execute ONNX with zero-filled inputs
///
/// # Mock Mode
///
/// This function returns an error in mock mode since TensorRT is not available.
pub fn run_onnx_zeroed(
    _onnx_model_bytes: &[u8],
    _input_descriptors: &[(String, Vec<usize>)],
) -> Result<Vec<TensorOutput>> {
    Err(Error::Runtime(
        "TensorRT-RTX not available in mock mode. Rebuild without mock: cargo build --no-default-features"
            .to_string(),
    ))
}
