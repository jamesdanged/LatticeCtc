#ifndef TFKERNELS_CONTEXT_HELPERS_H
#define TFKERNELS_CONTEXT_HELPERS_H

#include <string>
#include <sstream>
#include <iostream>

#include "tensorflow/core/framework/op_kernel.h"
#include "array_infos.h"
#include "tensor_helpers.h"


// op kernel context related code cannot be compiled by nvcc
// warnings will display


#define RET_IF_BAD if (!ctx->status().ok()) return


#define CHECK_CTX(CTX)     \
  do {                                  \
    if (!(CTX)->status().ok()) {        \
      return;                           \
    }                                   \
  } while (0)



// if not equal, returns early with error set
#define OP_REQUIRES_EQUALS(CTX, ARG1, ARG2) \
  do { \
      auto res1 = ARG1; \
      auto res2 = ARG2; \
      if (res1 != res2) { \
        std::stringstream ss; \
        ss << "Expected " << #ARG1 << " == " << #ARG2 << ", but got " << res1 << " != " << res2 << "."; \
        std::cerr << ss.str() << std::endl; \
        (CTX)->CtxFailure(tensorflow::errors::InvalidArgument(ss.str())); \
      } \
  } while (0);

// if not true, sets error. Caller should then return early.
#define OP_REQUIRES_TRUE_WITH_MSG(CTX, EXPR, MSG) \
  do { \
      if (!(EXPR)) { \
        std::string msg("Requirement failed: "); \
        msg += string(#EXPR) + ". " + MSG; \
        std::cerr << msg << std::endl; \
        (CTX)->CtxFailure(tensorflow::errors::InvalidArgument(msg)); \
      } \
  } while (0);

#define OP_REQUIRES_TRUE(CTX, EXPR) OP_REQUIRES_TRUE_WITH_MSG(CTX, EXPR, "")




namespace lng {


  template<typename T, IntSys NDIMS>
  const TensorWrapper<T, NDIMS> get_input(OpKernelContext * context, IntSys input_idx) {
    const Tensor & tensor = context->input(input_idx);
    return TensorWrapper<T, NDIMS>(const_cast<Tensor*>(&tensor));
  };

  template<typename T, IntSys NDIMS>
  TensorWrapper<T, NDIMS> alloc_temp(OpKernelContext* context, std::initializer_list<int64> dim_sizes, AllocatorAttributes attr = AllocatorAttributes()) {
    assert(dim_sizes.size() == NDIMS);

    Tensor t;
    DataType dt = get_dt<T>();

    Status st = context->allocate_temp(dt, dim_sizes, &t, attr);

    if (!st.ok()) {
      // relies upon caller to check context status and return early if error
      context->CtxFailureWithWarning(st);
      return TensorWrapper<T, NDIMS>();
    }
    return TensorWrapper<T, NDIMS>(t);
  };

  template<typename T, IntSys NDIMS>
  TensorWrapper<T, NDIMS> alloc_output(OpKernelContext* context, std::initializer_list<int64> dim_sizes, IntSys output_index, AllocatorAttributes attr = AllocatorAttributes()) {
    assert(dim_sizes.size() == NDIMS);

    Tensor * p_t = NULL;

    Status st = context->allocate_output(output_index, dim_sizes, &p_t, attr);
    if (!st.ok()) {
      // relies upon caller to check context status and return early if error
      context->CtxFailureWithWarning(st);
      return TensorWrapper<T, NDIMS>();
    }
    return TensorWrapper<T, NDIMS>(p_t);
  };


  template<typename T, IntSys NDIMS>
  TensorWrapper<T, NDIMS> alloc_output(OpKernelContext* context, std::initializer_list<int64> dim_sizes, StringPiece output_name, AllocatorAttributes attr = AllocatorAttributes()) {
    assert(dim_sizes.size() == NDIMS);

    Tensor * p_t = NULL;

    Status st = context->allocate_output(output_name, dim_sizes, &p_t, attr);
    if (!st.ok()) {
      // relies upon caller to check context status and return early if error
      context->CtxFailureWithWarning(st);
      return TensorWrapper<T, NDIMS>();
    }
    return TensorWrapper<T, NDIMS>(p_t);
  };





  void check_kernel_launch(OpKernelContext* context, bool launch_result, StringPiece message);

  /**
   * Synchronizes on the gpu stream of the context.
   * This must be done before accessing any results of a gpu kernel.
   * It doesn't have to be done before returning after every TF kernel, as TF seems to handle that.
   *
   * The gpu kernels must be launched on the context's stream for this synchronization to have any effect.
   *
   */
  void sync_gpu(OpKernelContext * context);
  void sync_gpu(const Eigen::GpuDevice & gpu_device);

  /**
   * THIS CANNOT BE USED IN TENSORFLOW!
   * Causes results to be inconsistent if multiple kernels of the same kind run sequentially.
   * First kernel will typically run ok.
   *
   * Copies from device to host, on the stream used by the context.
   *
   * relies upon caller to check context status and return early if error.
   *
   *
   */
  template<typename T, IntSys NDIMS>
  TensorWrapper<T, NDIMS> copy_d_to_h(OpKernelContext* context, const Tensor & src) {
    const Eigen::GpuDevice & d = context->eigen_gpu_device();

    AllocatorAttributes attr_hnd;   // host and device
    attr_hnd.set_on_host(true);
    attr_hnd.set_gpu_compatible(true);

    Tensor dest;
    Status st = context->allocate_temp(src.dtype(), src.shape(), &dest, attr_hnd);


    if (!st.ok()) {
      context->CtxFailureWithWarning(st);
      return TensorWrapper<T, NDIMS>();
    }


    // get byte size of contained type
    // TODO which tf function returns this for any dtype?
    auto dtype = src.dtype();
    size_t bytes_per = 0;
    if (dtype == DT_FLOAT) {
      bytes_per = 4;
    } else if (dtype == DT_DOUBLE) {
      bytes_per = 8;
    } else if (dtype == DT_INT32) {
      bytes_per = 4;
    } else if (dtype == DT_INT64) {
      bytes_per = 8;
    } else {
      printf("Unsupported dtype for copy_d_to_h!");
      abort();
    }

    const void* p_src = src.flat<T>().data();
    void* p_dest = dest.flat<T>().data();
    size_t byte_length = tensor_total_length(src) * bytes_per;

    if (!copy_d_to_h(d, p_src, p_dest, byte_length)) {
      context->CtxFailureWithWarning(Status(tensorflow::error::UNKNOWN, "Failed to copy from device to host."));
      return TensorWrapper<T, NDIMS>();
    }

    return TensorWrapper<T, NDIMS>(dest);
  }


}



#endif //TFKERNELS_CONTEXT_HELPERS_H
