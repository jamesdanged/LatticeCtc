#ifndef TFKERNELS_DECL_GPU_ONLY_H
#define TFKERNELS_DECL_GPU_ONLY_H

// EIGEN_USE_GPU must be defined at top of .cu files so that inclusion of
// tensorflow/core/framework/tensor.h
// causes Eigen::GpuDevice to be fully defined.
// Be sure to include this file before any other headers in .cu file.
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor.h"
#include "cuda_includes_for_ide.h"


#endif //TFKERNELS_DECL_GPU_ONLY_H
