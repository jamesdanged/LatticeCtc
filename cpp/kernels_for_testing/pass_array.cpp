#include "pass_array.h"

#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"



namespace lng {


  using std::cout;
  using std::endl;

  using namespace tensorflow;

  REGISTER_OP("PassArray")
      .Input("arr_in: int32")
      .Output("arr_out: int32")
      .Doc(R"doc(
Test we don't need to synchronize before manipulating data on both host and device.

)doc");


  class PassArrayOp : public OpKernel {
  public:
    explicit PassArrayOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {

      AllocatorAttributes attr;
      attr.set_on_host(true);
      attr.set_gpu_compatible(true);

      auto arr_in = get_input<IntSys, 1>(ctx, 0); RET_IF_BAD;
      auto arr_out = alloc_output<IntSys, 1>(ctx, {arr_in.dim0()}, 0, attr); RET_IF_BAD;

      // edit output
      arr_out.info().fill(1);

      // run kernel without synchronizing first
      pass_array_kernel_launcher(ctx->eigen_gpu_device(), arr_in.info(), arr_out.info());

      // edit output, but must have synchronized first
      sync_gpu(ctx);
      for (IntSys i = 0; i < arr_out.dim0(); i++) {
        arr_out(i) += 5;
      }
    }

  };


  REGISTER_KERNEL_BUILDER(
      Name("PassArray")
          .Device(DEVICE_GPU)
          .HostMemory("arr_in"),
      PassArrayOp);

}

