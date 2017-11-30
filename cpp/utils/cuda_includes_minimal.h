#ifndef TFKERNELS_CUDA_INCLUDES_MINIMAL_H
#define TFKERNELS_CUDA_INCLUDES_MINIMAL_H

#ifdef USING_NVCC

// __device__ will be defined by cuda

#else

// blank so ignored in host code
#define __device__



#endif


#endif //TFKERNELS_CUDA_INCLUDES_MINIMAL_H
