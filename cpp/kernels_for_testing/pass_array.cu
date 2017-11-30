#include "../utils/decl_gpu_only.h"

//// EIGEN_USE_GPU must be defined at top of .cu files so that inclusion of
//// tensorflow/core/framework/tensor.h
//// causes Eigen::GpuDevice to be fully defined.
//#define EIGEN_USE_GPU

#include "pass_array.h"
// #include "../utils/cuda_includes_for_ide.h"


namespace lng {


  __global__ void pass_array_kernel(const Arr1Int arr_in, Arr1Int arr_out) {

    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;

    IntSys len = arr_in.dim0();

    for (IntSys i = 0; i < len; i++) {
      arr_out(i) = arr_out(i) * 2 + arr_in(i) * 10;
    }

    printf("pass array: block %d thread %d\n", blockIdx.x, threadIdx.x);
  }

  void pass_array_kernel_launcher(const Eigen::GpuDevice & d, const Arr1Int & arr_in, Arr1Int & arr_out) {
    pass_array_kernel<<<32, 256, 0, d.stream()>>>(arr_in, arr_out);
  }
}