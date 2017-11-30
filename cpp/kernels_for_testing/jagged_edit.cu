#include "jagged_edit.h"

#include <stdio.h>
#include "../utils/cuda_includes_for_ide.h"

namespace lng {

  __global__ void jagged_edit_kernel(const Arrays2dFlt arrs_in, Arrays2dFlt arrs_out) {

    IntSys N = arrs_in.length;
    if (blockIdx.x != 0) return;
    if (threadIdx.x >= N) return;

    auto arr_in = arrs_in[threadIdx.x];
    auto arr_out = arrs_out[threadIdx.x];

    IntSys h = arr_in.dim0();
    IntSys w = arr_in.dim1();

    FloatSys counter = 0.5f;
    for (IntSys i = 0; i < h; i++) {
      for (IntSys j = 0; j < w; j++) {
        arr_out(i, j) = counter + arr_in(i, j);
        counter += 1;
      }
    }

  }

  void jagged_edit_kernel_launcher(const Arrays2dFlt & arrs_in, Arrays2dFlt & arrs_out) {
    jagged_edit_kernel<<<32, 256>>>(arrs_in, arrs_out);
  }


}
