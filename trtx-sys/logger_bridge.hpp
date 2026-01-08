#ifndef TRTX_LOGGER_BRIDGE_H
#define TRTX_LOGGER_BRIDGE_H

#include <NvInfer.h>

#ifdef __cplusplus
extern "C" {
#endif

// Rust callback function type
typedef void (*RustLogCallback)(void* user_data, int32_t severity, const char* msg);

// Opaque logger type for C interface
typedef struct RustLoggerBridge RustLoggerBridge;

// Create a logger bridge that calls back into Rust
RustLoggerBridge* create_rust_logger_bridge(RustLogCallback callback, void* user_data);

// Destroy the logger bridge
void destroy_rust_logger_bridge(RustLoggerBridge* logger);

// Get the ILogger pointer (for use with TensorRT C++ API)
nvinfer1::ILogger* get_logger_interface(RustLoggerBridge* logger);

// Factory functions for TensorRT (return raw pointers)
void* create_infer_builder(void* logger);
void* create_infer_runtime(void* logger);

// ONNX Parser factory function
void* create_onnx_parser(void* network, void* logger);

// Builder methods
void* builder_create_network_v2(void* builder, uint32_t flags);
void* builder_create_config(void* builder);
void* builder_build_serialized_network(void* builder, void* network, void* config, size_t* out_size);
void builder_config_set_memory_pool_limit(void* config, int32_t pool_type, size_t limit);

// Destruction methods
void delete_builder(void* builder);
void delete_network(void* network);
void delete_config(void* config);
void delete_runtime(void* runtime);
void delete_engine(void* engine);
void delete_context(void* context);
void delete_parser(void* parser);

// Runtime methods
void* runtime_deserialize_cuda_engine(void* runtime, const void* data, size_t size);

// Engine methods
int32_t engine_get_nb_io_tensors(void* engine);
const char* engine_get_tensor_name(void* engine, int32_t index);
void* engine_create_execution_context(void* engine);

// ExecutionContext methods
bool context_set_tensor_address(void* context, const char* name, void* data);
bool context_enqueue_v3(void* context, void* stream);

// Parser methods
bool parser_parse(void* parser, const void* data, size_t size);
int32_t parser_get_nb_errors(void* parser);
void* parser_get_error(void* parser, int32_t index);
const char* parser_error_desc(void* error);

// CUDA wrappers
int32_t cuda_malloc_wrapper(void** ptr, size_t size);
int32_t cuda_free_wrapper(void* ptr);
int32_t cuda_memcpy_wrapper(void* dst, const void* src, size_t count, int32_t kind);
int32_t cuda_device_synchronize_wrapper();
const char* cuda_get_error_string_wrapper(int32_t error);

#ifdef __cplusplus
}
#endif

#endif // TRTX_LOGGER_BRIDGE_H
