#include "mul_by_two.h"

#include <stdio.h>
#include "../utils/array_infos.h"
#include "../utils/cuda_includes_for_ide.h"


namespace lng {


  template<typename T>
  __global__ void mul_by_two_kernel(const Arr2<T> arr_in, Arr2<T> arr_out) {

    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;

    for (IntSys i = 0; i < arr_in.dim0(); i++) {
      for (IntSys j = 0; j < arr_in.dim1(); j++) {
        arr_out(i, j) = arr_in(i, j) * 2;
      }
    }

  }

  template<typename T>
  void mul_by_two_kernel_launcher(const Arr2<T> & arr_in, Arr2<T> & arr_out) {
    mul_by_two_kernel<<<32, 256>>>(arr_in, arr_out);
  }



#define MUL_BY_TWO_INSTANTIATE(T) \
  template __global__ void mul_by_two_kernel<T>(const Arr2<T> arr_in, Arr2<T> arr_out); \
  template void mul_by_two_kernel_launcher<T>(const Arr2<T> & arr_in, Arr2<T> & arr_out);

INSTANTIATE_WITH_ALL_FLT_TYPES(MUL_BY_TWO_INSTANTIATE)


}
