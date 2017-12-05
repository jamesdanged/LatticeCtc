cmake_minimum_required(VERSION 3.5.1)
project(LatticeCtc)

find_package(CUDA REQUIRED)


set(SOURCE_FILES
  utils/array_infos.h
  utils/context_helpers.h
  utils/context_helpers.cpp
  utils/context_helpers.cu
  utils/core_types.h
  utils/cuda_includes_minimal.h
  utils/cuda_includes_for_ide.h
  utils/decl_cpu_only.h
  utils/decl_gpu_only.h
  utils/error_handling_cpu.h
  utils/error_handling_cpu.cpp
  utils/math_utils.h
  utils/tensor_helpers.h
  utils/tensor_helpers.cpp
  utils/tensor_helpers.cu
  utils/timer.h
  kernels/backward_algo.h
  kernels/backward_algo.cpp
  kernels/backward_algo.cu
  kernels/ctc_loss.h
  kernels/ctc_loss.cpp
  kernels/ctc_loss.cu
  kernels/forward_algo.h
  kernels/forward_algo.cpp
  kernels/forward_algo.cu
  kernels/hmm_combined.cpp
  kernels/hmm_shared.h
  kernels/lattice.h
  kernels/lattice.cpp
  kernels/viterbi_algo.h
  kernels/viterbi_algo.cpp
  kernels/viterbi_algo.cu
  )

include_directories("/usr/local/lib/python3.5/dist-packages/tensorflow/include")
include_directories(SYSTEM "/usr/local/lib/cuda/include")

# Debug
# set(CUDA_NVCC_FLAGS "-g -G -O0 -DDO_BOUNDS_CHECK -arch sm_50 -lineinfo -DUSING_NVCC --expt-relaxed-constexpr -DFLOAT_SYS_IS_32_BIT")
# Release
set(CUDA_NVCC_FLAGS "-g -O2 -arch sm_50 -lineinfo -DNDEBUG -DUSING_NVCC --expt-relaxed-constexpr -DFLOAT_SYS_IS_32_BIT")

# Debug
# set(CMAKE_CXX_FLAGS "-g -O0 -DDO_BOUNDS_CHECK -fPIC -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -DFLOAT_SYS_IS_32_BIT")
# Release
set(CMAKE_CXX_FLAGS "-g -O2 -fPIC -std=c++11 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0 -DFLOAT_SYS_IS_32_BIT")

cuda_add_library(LatticeCtc SHARED ${SOURCE_FILES} )




