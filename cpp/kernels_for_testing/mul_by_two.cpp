#include "mul_by_two.h"

#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"

namespace lng {

  using std::cout;
  using std::endl;

  using namespace tensorflow;

  REGISTER_OP("MulByTwo")
      .Attr("T: {float, double}")
      .Input("input: T")
      .Output("output: T")
      .Doc(R"doc(
Multiplies by 2 all elements of the 2D tensor.

output: A Tensor.
  output = input .* 2
)doc");


  template<typename T>
  class MulByTwoOp : public OpKernel {
  public:
    explicit MulByTwoOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      auto input = get_input<T, 2>(context, 0);

      // create output tensor
      auto output = alloc_output<T, 2>(context, {input.dim0(), input.dim1()}, 0); CHECK_CTX(context);

      // call the launcher, which must be in a separate nvcc compiled file
      // even though it runs on the host
      mul_by_two_kernel_launcher<T>(input.info(), output.info());
    }

  };


#define REGISTER_KERNS(T)   \
  REGISTER_KERNEL_BUILDER(      \
    Name("MulByTwo")            \
      .Device(DEVICE_GPU)       \
      .TypeConstraint<T>("T"),  \
    MulByTwoOp<T>);


  // shows "error after macro substitution" in ide, but should compile
  INSTANTIATE_WITH_ALL_FLT_TYPES(REGISTER_KERNS);





}