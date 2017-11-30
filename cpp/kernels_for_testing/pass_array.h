#ifndef TFKERNELS_PASS_ARRAY_H
#define TFKERNELS_PASS_ARRAY_H

#include "../utils/array_infos.h"
// for GpuDevice
#include "tensorflow/core/framework/tensor.h"

namespace lng {

  void pass_array_kernel_launcher(const Eigen::GpuDevice & d, const Arr1Int & arr_in, Arr1Int & arr_out);

}

#endif //TFKERNELS_PASS_ARRAY_H
