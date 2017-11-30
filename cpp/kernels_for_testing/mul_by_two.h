#ifndef TF_KERNELS__MUL_BY_TWO_H__
#define TF_KERNELS__MUL_BY_TWO_H__

#include "../utils/core_types.h"
#include "../utils/array_infos.h"

namespace lng {

  template<typename T>
  void mul_by_two_kernel_launcher(const Arr2<T> & arr_in, Arr2<T> & arr_out);

}

#endif