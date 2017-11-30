#ifndef TFKERNELS_MATH_UTILS_H
#define TFKERNELS_MATH_UTILS_H

#include "core_types.h"
#include "cuda_includes_minimal.h"
#include "math.h"

namespace lng {

  __device__ inline
  IntSys div_round_up(IntSys numer, IntSys denom) { return (numer + denom - 1) / denom; }

  __device__ inline
  IntSys log2_round_up(IntSys x) { return (IntSys)log2((FloatSys) (x * 2 - 1)); }  // int floor by casting

  __device__ inline
  float add_log_space(float a, float b) {
    if (a == -INFINITY && b == -INFINITY) {
      return -INFINITY;
    }

    if (a > b) {
      return a + log1pf(expf(b - a));
    } else {
      return b + log1pf(expf(a - b));
    }
  }


  __device__ inline
  double add_log_space(double a, double b) {
    if (a == -INFINITY && b == -INFINITY) {
      return -INFINITY;
    }

    if (a > b) {
      return a + log1p(exp(b - a));
    } else {
      return b + log1p(exp(a - b));
    }
  }



}

#endif //TFKERNELS_MATH_UTILS_H
