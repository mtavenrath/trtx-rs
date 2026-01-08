//! Builder for creating TensorRT engines

use crate::error::{Error, Result};
use crate::logger::Logger;

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

/// Network definition for building TensorRT engines
pub struct NetworkDefinition {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxNetworkDefinition,
}

/// Tensor handle (opaque pointer)
pub struct Tensor {
    inner: *mut std::ffi::c_void,
}

impl Tensor {
    /// Get the tensor name
    pub fn name(&self) -> Result<String> {
        #[cfg(not(feature = "mock"))]
        {
            let name_ptr = unsafe { trtx_sys::tensor_get_name(self.inner) };
            if name_ptr.is_null() {
                return Err(Error::Runtime("Failed to get tensor name".to_string()));
            }
            unsafe {
                Ok(std::ffi::CStr::from_ptr(name_ptr)
                    .to_str()?
                    .to_string())
            }
        }
        #[cfg(feature = "mock")]
        {
            Ok("mock_tensor".to_string())
        }
    }

    /// Set the tensor name
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            unsafe {
                trtx_sys::tensor_set_name(self.inner, name_cstr.as_ptr());
            }
            Ok(())
        }
        #[cfg(feature = "mock")]
        {
            Ok(())
        }
    }

    /// Get tensor dimensions
    pub fn dimensions(&self) -> Result<Vec<i32>> {
        #[cfg(not(feature = "mock"))]
        {
            let mut dims = [0i32; 8]; // MAX_DIMS
            let mut nb_dims = 0i32;
            let result = unsafe {
                trtx_sys::tensor_get_dimensions(
                    self.inner,
                    dims.as_mut_ptr(),
                    &mut nb_dims,
                )
            };
            if result.is_null() {
                return Err(Error::Runtime("Failed to get tensor dimensions".to_string()));
            }
            Ok(dims[..nb_dims as usize].to_vec())
        }
        #[cfg(feature = "mock")]
        {
            Ok(vec![1, 3, 224, 224])
        }
    }

    /// Get raw pointer (internal use)
    pub(crate) fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.inner
    }
}

impl NetworkDefinition {
    /// Get the raw pointer (for internal use)
    #[cfg(not(feature = "mock"))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.inner
    }

    #[cfg(feature = "mock")]
    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxNetworkDefinition {
        self.inner
    }
    
    #[cfg(feature = "mock")]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut trtx_sys::TrtxNetworkDefinition {
        self.inner
    }

    /// Add an input tensor to the network
    pub fn add_input(
        &mut self,
        name: &str,
        data_type: i32,
        dims: &[i32],
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let name_cstr = std::ffi::CString::new(name)?;
            let tensor_ptr = unsafe {
                trtx_sys::network_add_input(
                    self.inner,
                    name_cstr.as_ptr(),
                    data_type,
                    dims.as_ptr(),
                    dims.len() as i32,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(format!("Failed to add input: {}", name)));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Mark a tensor as a network output
    pub fn mark_output(&mut self, tensor: &Tensor) -> Result<()> {
        #[cfg(not(feature = "mock"))]
        {
            let success = unsafe {
                trtx_sys::network_mark_output(self.inner, tensor.inner)
            };
            if !success {
                return Err(Error::Runtime("Failed to mark tensor as output".to_string()));
            }
            Ok(())
        }
        #[cfg(feature = "mock")]
        {
            Ok(())
        }
    }

    /// Get the number of inputs
    pub fn get_nb_inputs(&self) -> i32 {
        #[cfg(not(feature = "mock"))]
        unsafe {
            trtx_sys::network_get_nb_inputs(self.inner)
        }
        #[cfg(feature = "mock")]
        {
            0
        }
    }

    /// Get the number of outputs
    pub fn get_nb_outputs(&self) -> i32 {
        #[cfg(not(feature = "mock"))]
        unsafe {
            trtx_sys::network_get_nb_outputs(self.inner)
        }
        #[cfg(feature = "mock")]
        {
            0
        }
    }

    /// Get an input tensor by index
    pub fn get_input(&self, index: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_get_input(self.inner, index)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(format!("Failed to get input at index {}", index)));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Get an output tensor by index
    pub fn get_output(&self, index: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_get_output(self.inner, index)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime(format!("Failed to get output at index {}", index)));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add an activation layer
    pub fn add_activation(&mut self, input: &Tensor, activation_type: i32) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_activation(self.inner, input.inner, activation_type)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add activation layer".to_string()));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add an elementwise operation layer
    pub fn add_elementwise(
        &mut self,
        input1: &Tensor,
        input2: &Tensor,
        op: i32,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_elementwise(self.inner, input1.inner, input2.inner, op)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add elementwise layer".to_string()));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add a pooling layer
    pub fn add_pooling(
        &mut self,
        input: &Tensor,
        pooling_type: i32,
        window_size: &[i32; 2],
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_pooling(
                    self.inner,
                    input.inner,
                    pooling_type,
                    window_size.as_ptr(),
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add pooling layer".to_string()));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add a shuffle layer (for reshaping/transposing)
    pub fn add_shuffle(&mut self, input: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_shuffle(self.inner, input.inner)
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add shuffle layer".to_string()));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }

    /// Add a matrix multiply layer
    pub fn add_matrix_multiply(
        &mut self,
        input0: &Tensor,
        op0: i32,
        input1: &Tensor,
        op1: i32,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mock"))]
        {
            let tensor_ptr = unsafe {
                trtx_sys::network_add_matrix_multiply(
                    self.inner,
                    input0.inner,
                    op0,
                    input1.inner,
                    op1,
                )
            };
            if tensor_ptr.is_null() {
                return Err(Error::Runtime("Failed to add matrix multiply layer".to_string()));
            }
            Ok(Tensor { inner: tensor_ptr })
        }
        #[cfg(feature = "mock")]
        {
            Ok(Tensor {
                inner: std::ptr::null_mut(),
            })
        }
    }
}

impl Drop for NetworkDefinition {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_network_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_network(self.inner);
            }
        }
    }
}

unsafe impl Send for NetworkDefinition {}

/// Builder configuration
pub struct BuilderConfig {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxBuilderConfig,
}

impl BuilderConfig {
    /// Set memory pool limit
    pub fn set_memory_pool_limit(&mut self, pool: MemoryPoolType, size: usize) -> Result<()> {
        #[cfg(feature = "mock")]
        {
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_config_set_memory_pool_limit(
                    self.inner,
                    pool as i32,
                    size,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(())
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder config".to_string()));
            }

            let trt_pool = match pool {
                MemoryPoolType::Workspace => 0, // kWORKSPACE
                MemoryPoolType::DlaManagedSram => 1, // kDLA_MANAGED_SRAM
                MemoryPoolType::DlaLocalDram => 2, // kDLA_LOCAL_DRAM
                MemoryPoolType::DlaGlobalDram => 3, // kDLA_GLOBAL_DRAM
            };

            unsafe {
                trtx_sys::builder_config_set_memory_pool_limit(self.inner, trt_pool, size);
            }

            Ok(())
        }
    }

    /// Get the raw pointer (for internal use)
    #[cfg(not(feature = "mock"))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.inner
    }

    #[cfg(feature = "mock")]
    pub(crate) fn as_ptr(&self) -> *mut trtx_sys::TrtxBuilderConfig {
        self.inner
    }
    
    #[cfg(feature = "mock")]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut trtx_sys::TrtxBuilderConfig {
        self.inner
    }
}

impl Drop for BuilderConfig {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_builder_config_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_config(self.inner);
            }
        }
    }
}

unsafe impl Send for BuilderConfig {}

/// Builder for creating optimized TensorRT engines
pub struct Builder<'a> {
    #[cfg(not(feature = "mock"))]
    inner: *mut std::ffi::c_void,
    #[cfg(feature = "mock")]
    inner: *mut trtx_sys::TrtxBuilder,
    _logger: &'a Logger,
}

impl<'a> Builder<'a> {
    /// Create a new builder
    pub fn new(logger: &'a Logger) -> Result<Self> {
        #[cfg(feature = "mock")]
        {
            let mut builder_ptr: *mut trtx_sys::TrtxBuilder = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_create(
                    logger.as_ptr(),
                    &mut builder_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(Builder {
                inner: builder_ptr,
                _logger: logger,
            })
        }

        #[cfg(not(feature = "mock"))]
        {
            let logger_ptr = logger.as_logger_ptr();
            let builder_ptr = unsafe {
                trtx_sys::create_infer_builder(logger_ptr)
            };

            if builder_ptr.is_null() {
                return Err(Error::Runtime("Failed to create builder".to_string()));
            }

            Ok(Builder {
                inner: builder_ptr,
                _logger: logger,
            })
        }
    }

    /// Create a network definition
    pub fn create_network(&self, flags: u32) -> Result<NetworkDefinition> {
        #[cfg(feature = "mock")]
        {
            let mut network_ptr: *mut trtx_sys::TrtxNetworkDefinition = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_create_network(
                    self.inner,
                    flags,
                    &mut network_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(NetworkDefinition { inner: network_ptr })
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder".to_string()));
            }
            
            let network_ptr = unsafe {
                trtx_sys::builder_create_network_v2(self.inner, flags)
            };

            if network_ptr.is_null() {
                return Err(Error::Runtime("Failed to create network".to_string()));
            }

            Ok(NetworkDefinition { inner: network_ptr })
        }
    }

    /// Create a builder configuration
    pub fn create_config(&self) -> Result<BuilderConfig> {
        #[cfg(feature = "mock")]
        {
            let mut config_ptr: *mut trtx_sys::TrtxBuilderConfig = std::ptr::null_mut();
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_create_builder_config(
                    self.inner,
                    &mut config_ptr,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            Ok(BuilderConfig { inner: config_ptr })
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder".to_string()));
            }
            
            let config_ptr = unsafe {
                trtx_sys::builder_create_config(self.inner)
            };

            if config_ptr.is_null() {
                return Err(Error::Runtime("Failed to create builder config".to_string()));
            }

            Ok(BuilderConfig { inner: config_ptr })
        }
    }

    /// Build a serialized network (engine)
    pub fn build_serialized_network(
        &self,
        network: &mut NetworkDefinition,
        config: &mut BuilderConfig,
    ) -> Result<Vec<u8>> {
        #[cfg(feature = "mock")]
        {
            let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut size: usize = 0;
            let mut error_msg = [0i8; 1024];

            let result = unsafe {
                trtx_sys::trtx_builder_build_serialized_network(
                    self.inner,
                    network.as_ptr(),
                    config.as_ptr(),
                    &mut data_ptr,
                    &mut size,
                    error_msg.as_mut_ptr(),
                    error_msg.len(),
                )
            };

            if result != trtx_sys::TRTX_SUCCESS as i32 {
                return Err(Error::from_ffi(result, &error_msg));
            }

            // Copy data to Vec and free C buffer
            let data = unsafe {
                let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
                let vec = slice.to_vec();
                trtx_sys::trtx_free_buffer(data_ptr);
                vec
            };

            Ok(data)
        }

        #[cfg(not(feature = "mock"))]
        {
            if self.inner.is_null() {
                return Err(Error::Runtime("Invalid builder".to_string()));
            }
            let network_ptr = network.as_mut_ptr();
            let config_ptr = config.as_mut_ptr();
            
            let mut size: usize = 0;
            let data_ptr = unsafe {
                trtx_sys::builder_build_serialized_network(
                    self.inner,
                    network_ptr,
                    config_ptr,
                    &mut size
                )
            };

            if data_ptr.is_null() {
                return Err(Error::Runtime("Failed to build serialized network".to_string()));
            }

            // Copy data to Vec and free C buffer
            let data = unsafe {
                let slice = std::slice::from_raw_parts(data_ptr as *const u8, size);
                let vec = slice.to_vec();
                libc::free(data_ptr);
                vec
            };

            Ok(data)
        }
    }
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            #[cfg(feature = "mock")]
            unsafe {
                trtx_sys::trtx_builder_destroy(self.inner);
            }
            #[cfg(not(feature = "mock"))]
            unsafe {
                trtx_sys::delete_builder(self.inner);
            }
        }
    }
}

unsafe impl Send for Builder<'_> {}
