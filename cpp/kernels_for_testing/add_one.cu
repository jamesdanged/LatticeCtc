#include "../utils/cuda_includes_for_ide.h"

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


namespace lng {

  __global__ void AddOneKernel(const int* in, const int N, int* out) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
      out[i] = in[i] + 1;
    }

  }

  void AddOneKernelLauncher(const int* in, const int N, int* out) {
    AddOneKernel<<<32, 256>>>(in, N, out);
  }


}
