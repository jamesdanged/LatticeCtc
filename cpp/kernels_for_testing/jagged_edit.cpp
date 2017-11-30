#include "jagged_edit.h"

#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"


namespace lng {

  using std::cout;
  using std::endl;

  using namespace tensorflow;

  REGISTER_OP("JaggedEdit")
      .Input("n_in: int32")  // N arrays
      .Input("shapes_in: int32")  // shapes
      .Input("offsets_in: int32")  // offsets
      .Input("underlying_in: float32")  // underlying
      .Output("underlying_out: float32")  // underlying
      .Doc(R"doc(
Adds an incrementing amount to each input.

underlying_out: A Tensor.

)doc");



  class JaggedEditOp : public OpKernel {
  public:
    explicit JaggedEditOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {

      auto n_in = context->input(0).scalar<IntSys>()(0);
      auto shapes_in = get_input<IntSys, 2>(context, 1);
      auto offsets_in = get_input<IntSys, 1>(context, 2);
      auto underlying_in = get_input<FloatSys, 1>(context, 3);
      auto jagged_in = Arrays<FloatSys, 2>(n_in, underlying_in.info(), shapes_in.info(), offsets_in.info());

      // create output tensor
      auto underlying_out = alloc_output<FloatSys, 1>(context, {underlying_in.dim0()}, 0); CHECK_CTX(context);
      auto jagged_out = Arrays<FloatSys, 2>(n_in, underlying_out.info(), shapes_in.info(), offsets_in.info());

      jagged_edit_kernel_launcher(jagged_in, jagged_out);

    }

  };




  REGISTER_KERNEL_BUILDER(
      Name("JaggedEdit")
          .Device(DEVICE_GPU)
          .HostMemory("n_in"),
      JaggedEditOp);
}