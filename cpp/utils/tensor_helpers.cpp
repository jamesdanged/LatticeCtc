#include "tensor_helpers.h"


namespace lng {
  string shape_to_string(const Tensor & t) {
    stringstream ss;
    ss << "(";

    const TensorShape & shape = t.shape();
    for (IntSys i = 0; i < shape.dims(); i++) {
      ss << shape.dim_size(i);
      if (i != shape.dims() - 1) ss << ", ";
    }
    ss << ")";

    return ss.str();
  }

  IntSys tensor_total_length(const Tensor & t) {
    if (t.dims() == 0) return 0;
    IntSys total_length = 1;
    for (IntSys d = 0; d < t.dims(); d++) {
      total_length *= t.dim_size(0);
    }
    return total_length;
  }
}
