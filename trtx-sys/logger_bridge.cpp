#include "logger_bridge.hpp"
#include <NvOnnxParser.h>
#include <cstring>

// C++ implementation of ILogger that bridges to Rust
class RustLoggerImpl : public nvinfer1::ILogger {
public:
    RustLoggerImpl(RustLogCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (callback_) {
            callback_(user_data_, static_cast<int32_t>(severity), msg);
        }
    }

private:
    RustLogCallback callback_;
    void* user_data_;
};

// Opaque struct that holds the logger implementation
struct RustLoggerBridge {
    RustLoggerImpl* impl;
};

extern "C" {

RustLoggerBridge* create_rust_logger_bridge(RustLogCallback callback, void* user_data) {
    if (!callback) {
        return nullptr;
    }
    
    try {
        auto* bridge = new RustLoggerBridge();
        bridge->impl = new RustLoggerImpl(callback, user_data);
        return bridge;
    } catch (...) {
        return nullptr;
    }
}

void destroy_rust_logger_bridge(RustLoggerBridge* logger) {
    if (logger) {
        delete logger->impl;
        delete logger;
    }
}

nvinfer1::ILogger* get_logger_interface(RustLoggerBridge* logger) {
    return logger ? logger->impl : nullptr;
}

// Factory functions for TensorRT
void* create_infer_builder(void* logger) {
    if (!logger) {
        return nullptr;
    }
    try {
        auto* ilogger = static_cast<nvinfer1::ILogger*>(logger);
        return nvinfer1::createInferBuilder(*ilogger);
    } catch (...) {
        return nullptr;
    }
}

void* create_infer_runtime(void* logger) {
    if (!logger) {
        return nullptr;
    }
    try {
        auto* ilogger = static_cast<nvinfer1::ILogger*>(logger);
        return nvinfer1::createInferRuntime(*ilogger);
    } catch (...) {
        return nullptr;
    }
}

// ONNX Parser factory function
void* create_onnx_parser(void* network, void* logger) {
    if (!network || !logger) {
        return nullptr;
    }
    try {
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* ilogger = static_cast<nvinfer1::ILogger*>(logger);
        return nvonnxparser::createParser(*inetwork, *ilogger);
    } catch (...) {
        return nullptr;
    }
}

// Builder methods
void* builder_create_network_v2(void* builder, uint32_t flags) {
    if (!builder) return nullptr;
    try {
        auto* ibuilder = static_cast<nvinfer1::IBuilder*>(builder);
        return ibuilder->createNetworkV2(flags);
    } catch (...) {
        return nullptr;
    }
}

void* builder_create_config(void* builder) {
    if (!builder) return nullptr;
    try {
        auto* ibuilder = static_cast<nvinfer1::IBuilder*>(builder);
        return ibuilder->createBuilderConfig();
    } catch (...) {
        return nullptr;
    }
}

void builder_config_set_memory_pool_limit(void* config, int32_t pool_type, size_t limit) {
    if (!config) return;
    try {
        auto* iconfig = static_cast<nvinfer1::IBuilderConfig*>(config);
        iconfig->setMemoryPoolLimit(static_cast<nvinfer1::MemoryPoolType>(pool_type), limit);
    } catch (...) {
        // Ignore errors
    }
}

void* builder_build_serialized_network(void* builder, void* network, void* config, size_t* out_size) {
    if (!builder || !network || !config || !out_size) return nullptr;
    try {
        auto* ibuilder = static_cast<nvinfer1::IBuilder*>(builder);
        auto* inetwork = static_cast<nvinfer1::INetworkDefinition*>(network);
        auto* iconfig = static_cast<nvinfer1::IBuilderConfig*>(config);
        
        auto* serialized = ibuilder->buildSerializedNetwork(*inetwork, *iconfig);
        if (!serialized) return nullptr;
        
        *out_size = serialized->size();
        // Allocate and copy data
        void* data = malloc(*out_size);
        if (data) {
            memcpy(data, serialized->data(), *out_size);
        }
        delete serialized;
        return data;
    } catch (...) {
        return nullptr;
    }
}

// Runtime methods
void* runtime_deserialize_cuda_engine(void* runtime, const void* data, size_t size) {
    if (!runtime || !data) return nullptr;
    try {
        auto* iruntime = static_cast<nvinfer1::IRuntime*>(runtime);
        return iruntime->deserializeCudaEngine(data, size);
    } catch (...) {
        return nullptr;
    }
}

// Engine methods
int32_t engine_get_nb_io_tensors(void* engine) {
    if (!engine) return 0;
    try {
        auto* iengine = static_cast<nvinfer1::ICudaEngine*>(engine);
        return iengine->getNbIOTensors();
    } catch (...) {
        return 0;
    }
}

const char* engine_get_tensor_name(void* engine, int32_t index) {
    if (!engine) return nullptr;
    try {
        auto* iengine = static_cast<nvinfer1::ICudaEngine*>(engine);
        return iengine->getIOTensorName(index);
    } catch (...) {
        return nullptr;
    }
}

void* engine_create_execution_context(void* engine) {
    if (!engine) return nullptr;
    try {
        auto* iengine = static_cast<nvinfer1::ICudaEngine*>(engine);
        return iengine->createExecutionContext();
    } catch (...) {
        return nullptr;
    }
}

// ExecutionContext methods
bool context_set_tensor_address(void* context, const char* name, void* data) {
    if (!context || !name) return false;
    try {
        auto* icontex = static_cast<nvinfer1::IExecutionContext*>(context);
        return icontex->setTensorAddress(name, data);
    } catch (...) {
        return false;
    }
}

bool context_enqueue_v3(void* context, void* stream) {
    if (!context) return false;
    try {
        auto* icontext = static_cast<nvinfer1::IExecutionContext*>(context);
        return icontext->enqueueV3(static_cast<cudaStream_t>(stream));
    } catch (...) {
        return false;
    }
}

// Parser methods
bool parser_parse(void* parser, const void* data, size_t size) {
    if (!parser || !data) return false;
    try {
        auto* iparser = static_cast<nvonnxparser::IParser*>(parser);
        return iparser->parse(data, size);
    } catch (...) {
        return false;
    }
}

int32_t parser_get_nb_errors(void* parser) {
    if (!parser) return 0;
    try {
        auto* iparser = static_cast<nvonnxparser::IParser*>(parser);
        return iparser->getNbErrors();
    } catch (...) {
        return 0;
    }
}

void* parser_get_error(void* parser, int32_t index) {
    if (!parser) return nullptr;
    try {
        auto* iparser = static_cast<nvonnxparser::IParser*>(parser);
        return const_cast<nvonnxparser::IParserError*>(iparser->getError(index));
    } catch (...) {
        return nullptr;
    }
}

const char* parser_error_desc(void* error) {
    if (!error) return nullptr;
    try {
        auto* ierror = static_cast<nvonnxparser::IParserError*>(error);
        return ierror->desc();
    } catch (...) {
        return nullptr;
    }
}

// CUDA wrappers
int32_t cuda_malloc_wrapper(void** ptr, size_t size) {
    return static_cast<int32_t>(cudaMalloc(ptr, size));
}

int32_t cuda_free_wrapper(void* ptr) {
    return static_cast<int32_t>(cudaFree(ptr));
}

int32_t cuda_memcpy_wrapper(void* dst, const void* src, size_t count, int32_t kind) {
    return static_cast<int32_t>(cudaMemcpy(dst, src, count, static_cast<cudaMemcpyKind>(kind)));
}

int32_t cuda_device_synchronize_wrapper() {
    return static_cast<int32_t>(cudaDeviceSynchronize());
}

const char* cuda_get_error_string_wrapper(int32_t error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}

// Destruction methods
void delete_builder(void* builder) {
    if (builder) {
        delete static_cast<nvinfer1::IBuilder*>(builder);
    }
}

void delete_network(void* network) {
    if (network) {
        delete static_cast<nvinfer1::INetworkDefinition*>(network);
    }
}

void delete_config(void* config) {
    if (config) {
        delete static_cast<nvinfer1::IBuilderConfig*>(config);
    }
}

void delete_runtime(void* runtime) {
    if (runtime) {
        delete static_cast<nvinfer1::IRuntime*>(runtime);
    }
}

void delete_engine(void* engine) {
    if (engine) {
        delete static_cast<nvinfer1::ICudaEngine*>(engine);
    }
}

void delete_context(void* context) {
    if (context) {
        delete static_cast<nvinfer1::IExecutionContext*>(context);
    }
}

void delete_parser(void* parser) {
    if (parser) {
        delete static_cast<nvonnxparser::IParser*>(parser);
    }
}

} // extern "C"
