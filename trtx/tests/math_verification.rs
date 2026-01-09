//! Math operation verification tests
//!
//! These tests build networks, run inference, and verify outputs match expected values.
//! They test the actual computation, not just API calls.

use trtx::{Builder, LogHandler, Logger, Result, Runtime, Severity};
use trtx::builder::MemoryPoolType;
use trtx::cuda::{synchronize, DeviceBuffer};

/// Silent logger for tests
struct SilentLogger;

impl LogHandler for SilentLogger {
    fn log(&self, _severity: Severity, _message: &str) {
        // Uncomment to debug:
        // eprintln!("[{:?}] {}", _severity, _message);
    }
}

fn create_test_logger() -> Result<Logger> {
    Logger::new(SilentLogger)
}

/// Helper to get tensor names from engine
fn get_tensor_names(engine: &trtx::CudaEngine, expected_count: usize) -> Result<Vec<String>> {
    let nb_tensors = engine.get_nb_io_tensors()?;
    let mut names = Vec::new();
    
    for i in 0..nb_tensors.min(expected_count as i32) {
        if let Ok(name) = engine.get_tensor_name(i) {
            names.push(name);
        }
    }
    
    Ok(names)
}

/// Helper to convert f32 slice to bytes
fn f32_to_bytes(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// Helper to convert bytes to f32 slice
fn bytes_to_f32(data: &[u8]) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const f32,
            data.len() / std::mem::size_of::<f32>(),
        )
    }
}

#[test]
fn test_activation_relu_computation() -> Result<()> {
    let logger = create_test_logger()?;
    
    // 1. Build network: Input -> ReLU -> Output
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    let dims = [1, 4]; // Batch=1, 4 values
    let input = network.add_input("input", 0, &dims)?;
    let relu = network.add_activation(&input, 0)?; // ReLU
    network.mark_output(&relu)?;
    
    // 2. Build engine
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?;
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    
    // 3. Create runtime and deserialize
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;
    let mut context = engine.create_execution_context()?;
    
    // 4. Prepare test data: mix of positive, negative, and zero
    let input_data: Vec<f32> = vec![2.0, -3.0, 0.0, 5.5];
    let expected_output: Vec<f32> = vec![2.0, 0.0, 0.0, 5.5]; // max(0, x)
    
    // 5. Allocate device memory
    let size_bytes = input_data.len() * std::mem::size_of::<f32>();
    let mut input_device = DeviceBuffer::new(size_bytes)?;
    let output_device = DeviceBuffer::new(size_bytes)?;
    
    // 6. Copy input to device
    input_device.copy_from_host(f32_to_bytes(&input_data))?;
    
    // 7. Get actual tensor names from engine
    let nb_tensors = engine.get_nb_io_tensors()?;
    println!("  Engine has {} I/O tensors", nb_tensors);
    
    let mut input_name = String::from("input");
    let mut output_name = String::from("output");
    
    for i in 0..nb_tensors {
        if let Ok(name) = engine.get_tensor_name(i) {
            println!("    Tensor {}: {}", i, name);
            if i == 0 {
                input_name = name;
            } else if i == 1 {
                output_name = name;
            }
        }
    }
    
    // 8. Set tensor addresses
    unsafe {
        context.set_tensor_address(&input_name, input_device.as_ptr())?;
        context.set_tensor_address(&output_name, output_device.as_ptr())?;
    }
    
    // 8. Run inference
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    
    // 9. Copy output back
    let mut output_host = vec![0u8; size_bytes];
    output_device.copy_to_host(&mut output_host)?;
    let output_values = bytes_to_f32(&output_host);
    
    // 10. Verify results
    println!("ReLU Test:");
    println!("  Input:    {:?}", input_data);
    println!("  Expected: {:?}", expected_output);
    println!("  Actual:   {:?}", output_values);
    
    for (i, (&actual, &expected)) in output_values.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "ReLU output mismatch at index {}: expected {}, got {}", i, expected, actual);
    }
    
    println!("  ✓ ReLU computation verified!");
    
    Ok(())
}

#[test]
fn test_elementwise_add_computation() -> Result<()> {
    let logger = create_test_logger()?;
    
    // 1. Build network: Input1 + Input2 -> Output
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    let dims = [1, 4];
    let input1 = network.add_input("input1", 0, &dims)?;
    let input2 = network.add_input("input2", 0, &dims)?;
    let sum = network.add_elementwise(&input1, &input2, 0)?; // ADD = 0
    network.mark_output(&sum)?;
    
    // 2. Build engine
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?;
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    
    // 3. Create runtime
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;
    let mut context = engine.create_execution_context()?;
    
    // 4. Test data
    let input1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input2_data: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let expected_output: Vec<f32> = vec![1.5, 3.5, 5.5, 7.5];
    
    // 5. Allocate device memory
    let size_bytes = input1_data.len() * std::mem::size_of::<f32>();
    let mut input1_device = DeviceBuffer::new(size_bytes)?;
    let mut input2_device = DeviceBuffer::new(size_bytes)?;
    let output_device = DeviceBuffer::new(size_bytes)?;
    
    // 6. Copy inputs
    input1_device.copy_from_host(f32_to_bytes(&input1_data))?;
    input2_device.copy_from_host(f32_to_bytes(&input2_data))?;
    
    // 7. Get tensor names
    let tensor_names = get_tensor_names(&engine, 3)?;
    let input1_name = tensor_names.get(0).map(|s| s.as_str()).unwrap_or("input1");
    let input2_name = tensor_names.get(1).map(|s| s.as_str()).unwrap_or("input2");
    let output_name = tensor_names.get(2).map(|s| s.as_str()).unwrap_or("output");
    
    // 8. Bind tensors
    unsafe {
        context.set_tensor_address(input1_name, input1_device.as_ptr())?;
        context.set_tensor_address(input2_name, input2_device.as_ptr())?;
        context.set_tensor_address(output_name, output_device.as_ptr())?;
    }
    
    // 9. Run inference
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    
    // 10. Get results
    let mut output_host = vec![0u8; size_bytes];
    output_device.copy_to_host(&mut output_host)?;
    let output_values = bytes_to_f32(&output_host);
    
    // 11. Verify
    println!("ElementWise ADD Test:");
    println!("  Input1:   {:?}", input1_data);
    println!("  Input2:   {:?}", input2_data);
    println!("  Expected: {:?}", expected_output);
    println!("  Actual:   {:?}", output_values);
    
    for (i, (&actual, &expected)) in output_values.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "ADD output mismatch at index {}: expected {}, got {}", i, expected, actual);
    }
    
    println!("  ✓ ElementWise ADD verified!");
    
    Ok(())
}

#[test]
fn test_elementwise_multiply_computation() -> Result<()> {
    let logger = create_test_logger()?;
    
    // 1. Build network: Input1 * Input2 -> Output
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    let dims = [1, 4];
    let input1 = network.add_input("input1", 0, &dims)?;
    let input2 = network.add_input("input2", 0, &dims)?;
    let product = network.add_elementwise(&input1, &input2, 1)?; // PROD = 1
    network.mark_output(&product)?;
    
    // 2. Build engine
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?;
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    
    // 3. Create runtime
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;
    let mut context = engine.create_execution_context()?;
    
    // 4. Test data
    let input1_data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
    let input2_data: Vec<f32> = vec![0.5, 2.0, 1.5, 0.0];
    let expected_output: Vec<f32> = vec![1.0, 6.0, 6.0, 0.0];
    
    // 5. Allocate device memory
    let size_bytes = input1_data.len() * std::mem::size_of::<f32>();
    let mut input1_device = DeviceBuffer::new(size_bytes)?;
    let mut input2_device = DeviceBuffer::new(size_bytes)?;
    let output_device = DeviceBuffer::new(size_bytes)?;
    
    // 6. Copy inputs
    input1_device.copy_from_host(f32_to_bytes(&input1_data))?;
    input2_device.copy_from_host(f32_to_bytes(&input2_data))?;
    
    // 7. Get tensor names
    let tensor_names = get_tensor_names(&engine, 3)?;
    let input1_name = tensor_names.get(0).map(|s| s.as_str()).unwrap_or("input1");
    let input2_name = tensor_names.get(1).map(|s| s.as_str()).unwrap_or("input2");
    let output_name = tensor_names.get(2).map(|s| s.as_str()).unwrap_or("output");
    
    // 8. Bind tensors
    unsafe {
        context.set_tensor_address(input1_name, input1_device.as_ptr())?;
        context.set_tensor_address(input2_name, input2_device.as_ptr())?;
        context.set_tensor_address(output_name, output_device.as_ptr())?;
    }
    
    // 9. Run inference
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    
    // 10. Get results
    let mut output_host = vec![0u8; size_bytes];
    output_device.copy_to_host(&mut output_host)?;
    let output_values = bytes_to_f32(&output_host);
    
    // 11. Verify
    println!("ElementWise MULTIPLY Test:");
    println!("  Input1:   {:?}", input1_data);
    println!("  Input2:   {:?}", input2_data);
    println!("  Expected: {:?}", expected_output);
    println!("  Actual:   {:?}", output_values);
    
    for (i, (&actual, &expected)) in output_values.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "MULTIPLY output mismatch at index {}: expected {}, got {}", i, expected, actual);
    }
    
    println!("  ✓ ElementWise MULTIPLY verified!");
    
    Ok(())
}

#[test]
fn test_activation_sigmoid_computation() -> Result<()> {
    let logger = create_test_logger()?;
    
    // 1. Build network: Input -> Sigmoid -> Output
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    let dims = [1, 4];
    let input = network.add_input("input", 0, &dims)?;
    let sigmoid = network.add_activation(&input, 1)?; // SIGMOID = 1
    network.mark_output(&sigmoid)?;
    
    // 2. Build engine
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?;
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    
    // 3. Create runtime
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;
    let mut context = engine.create_execution_context()?;
    
    // 4. Test data
    let input_data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0];
    // sigmoid(x) = 1 / (1 + exp(-x))
    let expected_output: Vec<f32> = vec![
        0.5,        // sigmoid(0) = 0.5
        0.7310586,  // sigmoid(1) ≈ 0.731
        0.2689414,  // sigmoid(-1) ≈ 0.269
        0.8807971,  // sigmoid(2) ≈ 0.881
    ];
    
    // 5. Allocate device memory
    let size_bytes = input_data.len() * std::mem::size_of::<f32>();
    let mut input_device = DeviceBuffer::new(size_bytes)?;
    let output_device = DeviceBuffer::new(size_bytes)?;
    
    // 6. Copy input
    input_device.copy_from_host(f32_to_bytes(&input_data))?;
    
    // 7. Get tensor names
    let tensor_names = get_tensor_names(&engine, 2)?;
    let input_name = tensor_names.get(0).map(|s| s.as_str()).unwrap_or("input");
    let output_name = tensor_names.get(1).map(|s| s.as_str()).unwrap_or("output");
    
    // 8. Bind tensors
    unsafe {
        context.set_tensor_address(input_name, input_device.as_ptr())?;
        context.set_tensor_address(output_name, output_device.as_ptr())?;
    }
    
    // 9. Run inference
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    
    // 10. Get results
    let mut output_host = vec![0u8; size_bytes];
    output_device.copy_to_host(&mut output_host)?;
    let output_values = bytes_to_f32(&output_host);
    
    // 11. Verify
    println!("Sigmoid Test:");
    println!("  Input:    {:?}", input_data);
    println!("  Expected: {:?}", expected_output);
    println!("  Actual:   {:?}", output_values);
    
    for (i, (&actual, &expected)) in output_values.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "Sigmoid output mismatch at index {}: expected {}, got {}", i, expected, actual);
    }
    
    println!("  ✓ Sigmoid computation verified!");
    
    Ok(())
}

#[test]
fn test_activation_tanh_computation() -> Result<()> {
    let logger = create_test_logger()?;
    
    // 1. Build network: Input -> Tanh -> Output
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    let dims = [1, 4];
    let input = network.add_input("input", 0, &dims)?;
    let tanh = network.add_activation(&input, 2)?; // TANH = 2
    network.mark_output(&tanh)?;
    
    // 2. Build engine
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?;
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    
    // 3. Create runtime
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;
    let mut context = engine.create_execution_context()?;
    
    // 4. Test data
    let input_data: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    let expected_output: Vec<f32> = vec![
        0.0,        // tanh(0) = 0
        0.7615942,  // tanh(1) ≈ 0.762
        -0.7615942, // tanh(-1) ≈ -0.762
        0.46211717, // tanh(0.5) ≈ 0.462
    ];
    
    // 5. Allocate device memory
    let size_bytes = input_data.len() * std::mem::size_of::<f32>();
    let mut input_device = DeviceBuffer::new(size_bytes)?;
    let output_device = DeviceBuffer::new(size_bytes)?;
    
    // 6. Copy input
    input_device.copy_from_host(f32_to_bytes(&input_data))?;
    
    // 7. Get tensor names
    let tensor_names = get_tensor_names(&engine, 2)?;
    let input_name = tensor_names.get(0).map(|s| s.as_str()).unwrap_or("input");
    let output_name = tensor_names.get(1).map(|s| s.as_str()).unwrap_or("output");
    
    // 8. Bind tensors
    unsafe {
        context.set_tensor_address(input_name, input_device.as_ptr())?;
        context.set_tensor_address(output_name, output_device.as_ptr())?;
    }
    
    // 8. Run inference
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    
    // 9. Get results
    let mut output_host = vec![0u8; size_bytes];
    output_device.copy_to_host(&mut output_host)?;
    let output_values = bytes_to_f32(&output_host);
    
    // 10. Verify
    println!("Tanh Test:");
    println!("  Input:    {:?}", input_data);
    println!("  Expected: {:?}", expected_output);
    println!("  Actual:   {:?}", output_values);
    
    for (i, (&actual, &expected)) in output_values.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "Tanh output mismatch at index {}: expected {}, got {}", i, expected, actual);
    }
    
    println!("  ✓ Tanh computation verified!");
    
    Ok(())
}

#[test]
fn test_chained_operations() -> Result<()> {
    let logger = create_test_logger()?;
    
    // 1. Build network: (Input1 + Input2) -> ReLU -> Output
    let builder = Builder::new(&logger)?;
    let mut network = builder.create_network(0)?;
    
    let dims = [1, 4];
    let input1 = network.add_input("input1", 0, &dims)?;
    let input2 = network.add_input("input2", 0, &dims)?;
    let sum = network.add_elementwise(&input1, &input2, 0)?; // ADD
    let relu = network.add_activation(&sum, 0)?; // ReLU
    network.mark_output(&relu)?;
    
    // 2. Build engine
    let mut config = builder.create_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 20)?;
    let engine_data = builder.build_serialized_network(&mut network, &mut config)?;
    
    // 3. Create runtime
    let runtime = Runtime::new(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine_data)?;
    let mut context = engine.create_execution_context()?;
    
    // 4. Test data
    let input1_data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
    let input2_data: Vec<f32> = vec![0.5, 3.0, -1.0, 2.0];
    // sum = [1.5, 1.0, 2.0, -2.0]
    // relu = [1.5, 1.0, 2.0, 0.0]
    let expected_output: Vec<f32> = vec![1.5, 1.0, 2.0, 0.0];
    
    // 5. Allocate device memory
    let size_bytes = input1_data.len() * std::mem::size_of::<f32>();
    let mut input1_device = DeviceBuffer::new(size_bytes)?;
    let mut input2_device = DeviceBuffer::new(size_bytes)?;
    let output_device = DeviceBuffer::new(size_bytes)?;
    
    // 6. Copy inputs
    input1_device.copy_from_host(f32_to_bytes(&input1_data))?;
    input2_device.copy_from_host(f32_to_bytes(&input2_data))?;
    
    // 7. Get tensor names
    let tensor_names = get_tensor_names(&engine, 3)?;
    let input1_name = tensor_names.get(0).map(|s| s.as_str()).unwrap_or("input1");
    let input2_name = tensor_names.get(1).map(|s| s.as_str()).unwrap_or("input2");
    let output_name = tensor_names.get(2).map(|s| s.as_str()).unwrap_or("output");
    
    // 8. Bind tensors
    unsafe {
        context.set_tensor_address(input1_name, input1_device.as_ptr())?;
        context.set_tensor_address(input2_name, input2_device.as_ptr())?;
        context.set_tensor_address(output_name, output_device.as_ptr())?;
    }
    
    // 8. Run inference
    let stream = trtx::cuda::get_default_stream();
    unsafe {
        context.enqueue_v3(stream)?;
    }
    synchronize()?;
    
    // 9. Get results
    let mut output_host = vec![0u8; size_bytes];
    output_device.copy_to_host(&mut output_host)?;
    let output_values = bytes_to_f32(&output_host);
    
    // 10. Verify
    println!("Chained Operations (ADD + ReLU) Test:");
    println!("  Input1:   {:?}", input1_data);
    println!("  Input2:   {:?}", input2_data);
    println!("  Expected: {:?}", expected_output);
    println!("  Actual:   {:?}", output_values);
    
    for (i, (&actual, &expected)) in output_values.iter().zip(expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "Chained operation output mismatch at index {}: expected {}, got {}", i, expected, actual);
    }
    
    println!("  ✓ Chained operations verified!");
    
    Ok(())
}
