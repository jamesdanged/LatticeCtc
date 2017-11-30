#ifndef TFKERNELS_CUDA_GENERAL_INCLUDES_H
#define TFKERNELS_CUDA_GENERAL_INCLUDES_H

// this file should only be referenced in .cu files,
// not .cpp nor .h files


#include "cuda_includes_minimal.h"

#ifndef USING_NVCC

// This block will be used by clion to parse code.
// Since they should be only used in .cu files, and .cu files are called with USING_NVCC defined,
// they should not matter.

#include <host_defines.h>
#include <stdint.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



#endif



#endif //TFKERNELS_CUDA_GENERAL_INCLUDES_H
