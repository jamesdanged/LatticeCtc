#ifndef TFKERNELS_HMM_SHARED_H
#define TFKERNELS_HMM_SHARED_H

#include "../utils/core_types.h"
#include "../utils/cuda_includes_minimal.h"

namespace lng {

  __device__ const IntSys MAX_NUM_TRANS_PER_STATE = 20;
  __device__ const IntSys HMM_BLOCK_DIM = 32;
  __device__ const IntSys HMM_BLOCK_DIM_EXPANDED = 128;
  __device__ const bool DO_NAN_CHECK = false; // true;
  // __device__ const bool DO_ZERO_SIGMA_CHECK = false; // true;




}
#endif //TFKERNELS_HMM_SHARED_H
