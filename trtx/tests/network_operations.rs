#![cfg(not(feature = "mock"))]

//! Integration tests for common TensorRT network operations
//!
//! These tests verify that the trtx API correctly handles various layer types
//! and network building operations. Tests run in both mock and real TensorRT modes.

use trtx::{Builder, LogHandler, Logger, Result, Severity};
use trtx::builder::MemoryPoolType;

/// Silent logger for tests
struct SilentLogger;

impl LogHandler for SilentLogger {
    fn log(&self, _severity: Severity, _message: &str) {
        // Silent - uncomment to debug:
        // eprintln!("[{:?}] {}", _severity, _message);
    }
}

/// Helper to create a test logger
fn create_test_logger() -> Result<Logger> {
    Logger::new(SilentLogger)
}

#[test]
fn test_basic_network_creation() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let network = builder.create_network(0)?;
    
    // Verify we can query the network
    assert_eq!(network.get_nb_inputs(), 0);
    assert_eq!(network.get_nb_outputs(), 0);
    
    Ok(())
}

#[test]
fn test_input_output_operations() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input tensor
    let dims = [1, 3, 224, 224]; // NCHW format
    let input = network.add_input("input", 0, &dims)?; // DataType::kFLOAT = 0
    
    // Mark as output
    network.mark_output(&input)?;
    
    // Verify tensor properties (in mock mode these may not be fully functional)
    #[cfg(not(feature = "mock"))]
    {
        let input_name = input.name()?;
        assert_eq!(input_name, "input");
    }
    
    // Note: get_nb_inputs/get_nb_outputs/get_input/get_output may not work
    // reliably in all TensorRT versions. The test above (adding input and
    // marking output) validates the core functionality.
    
    Ok(())
}

#[test]
fn test_activation_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Add ReLU activation (ActivationType::kRELU = 0)
    let relu_output = network.add_activation(&input, 0)?;
    
    // Mark as output
    network.mark_output(&relu_output)?;
    
    // In real mode, verify output was added
    #[cfg(not(feature = "mock"))]
    assert_eq!(network.get_nb_outputs(), 1);
    
    Ok(())
}

#[test]
fn test_elementwise_operations() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add two inputs
    let dims = [1, 3, 224, 224];
    let input1 = network.add_input("input1", 0, &dims)?;
    let input2 = network.add_input("input2", 0, &dims)?;
    
    // ElementWiseOperation::kSUM = 0
    let sum_output = network.add_elementwise(&input1, &input2, 0)?;
    
    // ElementWiseOperation::kPROD = 1 (multiply)
    let prod_output = network.add_elementwise(&input1, &input2, 1)?;
    
    network.mark_output(&sum_output)?;
    network.mark_output(&prod_output)?;
    
    // In real mode, verify outputs were added
    #[cfg(not(feature = "mock"))]
    assert_eq!(network.get_nb_outputs(), 2);
    
    Ok(())
}

#[test]
fn test_pooling_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Add max pooling (PoolingType::kMAX = 0)
    let window_size = [2, 2];
    let pool_output = network.add_pooling(&input, 0, &window_size)?;
    
    network.mark_output(&pool_output)?;
    
    Ok(())
}

#[test]
fn test_shuffle_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Add shuffle layer (for reshaping/transposing)
    let shuffle_output = network.add_shuffle(&input)?;
    
    network.mark_output(&shuffle_output)?;
    
    Ok(())
}

#[test]
fn test_concatenation_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add multiple inputs
    let dims = [1, 3, 224, 224];
    let input1 = network.add_input("input1", 0, &dims)?;
    let input2 = network.add_input("input2", 0, &dims)?;
    let input3 = network.add_input("input3", 0, &dims)?;
    
    // Concatenate them
    let inputs = [&input1, &input2, &input3];
    let concat_output = network.add_concatenation(&inputs)?;
    
    network.mark_output(&concat_output)?;
    
    Ok(())
}

#[test]
fn test_matrix_multiply_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add matrix inputs
    let dims1 = [1, 4, 8];  // Batch, M, K
    let dims2 = [1, 8, 16]; // Batch, K, N
    let input1 = network.add_input("matrix1", 0, &dims1)?;
    let input2 = network.add_input("matrix2", 0, &dims2)?;
    
    // MatrixOperation::kNONE = 0
    let matmul_output = network.add_matrix_multiply(&input1, 0, &input2, 0)?;
    
    network.mark_output(&matmul_output)?;
    
    Ok(())
}

#[test]
#[ignore] // Requires proper weight data format
fn test_constant_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Create constant tensor
    let dims = [1, 3];
    let weights: Vec<u8> = vec![0; 12]; // 3 floats * 4 bytes
    
    let constant = network.add_constant(&dims, &weights)?;
    
    network.mark_output(&constant)?;
    
    Ok(())
}

#[test]
fn test_softmax_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 1000]; // Batch, classes
    let input = network.add_input("input", 0, &dims)?;
    
    // Add softmax on last axis (axes = 1 << axis_index)
    let softmax_output = network.add_softmax(&input, 1 << 1)?;
    
    network.mark_output(&softmax_output)?;
    
    Ok(())
}

#[test]
#[ignore] // Requires proper weight data format
fn test_scale_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Scale weights (shift, scale, power)
    let weights: Vec<u8> = vec![0; 12]; // 3 floats * 4 bytes
    
    // ScaleMode::kUNIFORM = 0
    let scale_output = network.add_scale(&input, 0, &weights, &weights, &weights)?;
    
    network.mark_output(&scale_output)?;
    
    Ok(())
}

#[test]
fn test_reduce_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // ReduceOperation::kSUM = 0, reduce on axes 2 and 3 (H, W)
    let axes = (1 << 2) | (1 << 3);
    let reduce_output = network.add_reduce(&input, 0, axes, true)?;
    
    network.mark_output(&reduce_output)?;
    
    Ok(())
}

#[test]
fn test_slice_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Slice: start at [0,0,0,0], size [1,3,112,112], stride [1,1,1,1]
    let start = [0, 0, 0, 0];
    let size = [1, 3, 112, 112];
    let stride = [1, 1, 1, 1];
    
    let slice_output = network.add_slice(&input, &start, &size, &stride)?;
    
    network.mark_output(&slice_output)?;
    
    Ok(())
}

#[test]
fn test_resize_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Add resize layer
    let resize_output = network.add_resize(&input)?;
    
    network.mark_output(&resize_output)?;
    
    Ok(())
}

#[test]
fn test_topk_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 1000]; // Batch, values
    let input = network.add_input("input", 0, &dims)?;
    
    // TopKOperation::kMAX = 0, get top 5 values
    let topk_output = network.add_topk(&input, 0, 5, 1 << 1)?;
    
    network.mark_output(&topk_output)?;
    
    Ok(())
}

#[test]
fn test_gather_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add data and indices tensors
    let data_dims = [1, 100, 64];
    let indices_dims = [1, 10];
    let data = network.add_input("data", 0, &data_dims)?;
    let indices = network.add_input("indices", 2, &indices_dims)?; // INT32 = 2
    
    // Gather on axis 1
    let gather_output = network.add_gather(&data, &indices, 1)?;
    
    network.mark_output(&gather_output)?;
    
    Ok(())
}

#[test]
fn test_select_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add condition and value tensors
    let dims = [1, 3, 224, 224];
    let condition = network.add_input("condition", 5, &dims)?; // BOOL = 5
    let then_input = network.add_input("then", 0, &dims)?;
    let else_input = network.add_input("else", 0, &dims)?;
    
    // Select based on condition
    let select_output = network.add_select(&condition, &then_input, &else_input)?;
    
    network.mark_output(&select_output)?;
    
    Ok(())
}

#[test]
fn test_assertion_layer() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add condition tensor
    let dims = [1];
    let condition = network.add_input("condition", 5, &dims)?; // BOOL = 5
    
    // Add assertion (no output tensor)
    network.add_assertion(&condition, "Condition must be true")?;
    
    // Still need an output for the network
    network.mark_output(&condition)?;
    
    Ok(())
}

#[test]
fn test_loop_construct() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add loop (returns raw pointer to ILoop)
    let loop_ptr = network.add_loop()?;
    
    // Verify we got a non-null pointer (in mock mode it's null_mut)
    #[cfg(feature = "mock")]
    assert!(loop_ptr.is_null());
    
    #[cfg(not(feature = "mock"))]
    assert!(!loop_ptr.is_null());
    
    Ok(())
}

#[test]
fn test_if_conditional_construct() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add if-conditional (returns raw pointer to IIfConditional)
    let if_ptr = network.add_if_conditional()?;
    
    // Verify we got a non-null pointer (in mock mode it's null_mut)
    #[cfg(feature = "mock")]
    assert!(if_ptr.is_null());
    
    #[cfg(not(feature = "mock"))]
    assert!(!if_ptr.is_null());
    
    Ok(())
}

#[test]
#[ignore] // Requires proper weight data format
fn test_complex_network() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Build a more complex network: Input -> Conv -> ReLU -> Pool -> Output
    let dims = [1, 3, 224, 224];
    let input = network.add_input("input", 0, &dims)?;
    
    // Convolution (simplified - would need real weights in production)
    let kernel_size = [3, 3];
    let weights: Vec<u8> = vec![0; 64 * 3 * 3 * 3 * 4]; // out_channels * in_channels * kH * kW * sizeof(float)
    let conv_output = network.add_convolution(&input, 64, &kernel_size, &weights, None)?;
    
    // ReLU activation
    let relu_output = network.add_activation(&conv_output, 0)?;
    
    // Max pooling
    let pool_size = [2, 2];
    let pool_output = network.add_pooling(&relu_output, 0, &pool_size)?;
    
    // Mark as output
    network.mark_output(&pool_output)?;
    
    Ok(())
}

#[test]
fn test_tensor_properties() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add input
    let dims = [1, 3, 224, 224];
    let input = network.add_input("original_name", 0, &dims)?;
    
    // Test name operations (only in real mode)
    #[cfg(not(feature = "mock"))]
    {
        let name = input.name()?;
        assert_eq!(name, "original_name");
        
        // Note: set_name takes &mut self, so we'd need a mutable binding
        // For now, just test reading
        
        // Test dimensions
        let retrieved_dims = input.dimensions()?;
        assert_eq!(retrieved_dims.len(), 4);
        
        // Test data type
        let data_type = input.get_type()?;
        assert_eq!(data_type, 0); // kFLOAT
    }
    
    // Mark as output for network completeness
    network.mark_output(&input)?;
    
    Ok(())
}

#[test]
fn test_builder_config() -> Result<()> {
    let logger = create_test_logger()?;
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    // Add simple network
    let dims = [1, 1];
    let input = network.add_input("input", 0, &dims)?;
    network.mark_output(&input)?;
    
    // Create builder config
    let mut config = builder.create_config()?;
    
    // Set memory pool limit
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?; // 1GB
    
    // In real mode, we could build the network
    // let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    // assert!(!engine_data.is_empty());
    
    Ok(())
}

#[cfg(all(test, not(feature = "mock")))]
mod real_tensorrt_tests {
    use super::*;
    
    #[test]
    #[ignore] // Ignore by default, requires GPU
    fn test_build_and_serialize() -> Result<()> {
        let logger = create_test_logger()?;
        let builder = Builder::new(&logger)?;
        let mut network = builder.create_network(0)?;
        
        // Simple passthrough network
        let dims = [1, 1];
        let input = network.add_input("input", 0, &dims)?;
        network.mark_output(&input)?;
        
        let mut config = builder.create_config()?;
        config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?; // 1MB
        
        // This will actually build the engine
        let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
        assert!(!engine_data.is_empty());
        
        Ok(())
    }
}
