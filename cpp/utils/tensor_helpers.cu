#define EIGEN_USE_GPU

// must be included after declaring EIGEN_USE_GPU to get definition of Eigen::GpuDevice
#include "tensorflow/core/framework/tensor.h"
#include "tensor_helpers.h"

namespace lng {

  using namespace tensorflow;

  /**
   * Copies from device to host, on the stream used by the context.
   * @returns true if ok.
   */
  bool copy_d_to_h(const Eigen::GpuDevice & d, const void * src, void * dest, size_t byte_length) {  // OpKernelContext* context   const Tensor & src, Tensor & dest
    d.memcpyDeviceToHost(const_cast<void*>(src), dest, byte_length);
    d.synchronize();
    return d.ok();
  }

}