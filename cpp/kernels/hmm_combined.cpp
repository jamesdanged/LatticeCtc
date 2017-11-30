#include <iostream>
#include <sstream>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"
#include "forward_algo.h"
#include "backward_algo.h"
#include "viterbi_algo.h"



namespace lng {

  using std::cout;
  using std::endl;
  using std::string;
  using std::to_string;
  using std::vector;
  using std::stringstream;

  using namespace tensorflow;

  REGISTER_OP("HmmCombined")
      .Attr("TFLT: {float, double}")
          LATTICE_INPUTS_TO_REGISTER_OP
      .Output("all_alphas: TFLT")
      .Output("all_betas: TFLT")
      .Output("all_deltas: TFLT")
      .Output("all_tracebacks: int32")
      .Output("all_best_paths: int32")
  .Doc(R"doc(

Simply runs forward, backward, and viterbi all at once
and returns all lattice results.

)doc");

  template<typename TFLT, typename DEVICE>
  class HmmCombinedOp : public OpKernel {
  public:
    explicit HmmCombinedOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      auto lat = Lattice<TFLT>::create(ctx); RET_IF_BAD;

      auto all_alphas = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 0, lat.attr_hnd); RET_IF_BAD;
      all_alphas.info().fill(NAN);
      auto all_betas = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 1, lat.attr_hnd); RET_IF_BAD;
      all_betas.info().fill(NAN);
      auto all_deltas = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 2, lat.attr_hnd); RET_IF_BAD;
      all_deltas.info().fill(NAN);
      auto all_tracebacks = alloc_output<IntSys, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 3, lat.attr_hnd); RET_IF_BAD;
      all_tracebacks.info().fill(-1);
      auto all_best_paths = alloc_output<IntSys, 2>(ctx, {lat.num_sequences, lat.max_T}, 4, lat.attr_hnd); RET_IF_BAD;
      all_best_paths.info().fill(-1);

      bool launch_res = ForwardFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_alphas.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch forward algorithm kernel.");

      launch_res = BackwardFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_betas.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch backward algorithm kernel.");

      launch_res = ViterbiFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_deltas.info(),
                                                    all_tracebacks.info(), all_best_paths.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch viterbi algorithm kernel.");
    }
  };


  #define REGISTER_KERNS(TFLT)   \
  REGISTER_KERNEL_BUILDER(      \
    Name("HmmCombined")            \
      .Device(DEVICE_GPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmCombinedOp<TFLT, Eigen::GpuDevice>); \
  REGISTER_KERNEL_BUILDER(      \
    Name("HmmCombined")            \
      .Device(DEVICE_CPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmCombinedOp<TFLT, Eigen::ThreadPoolDevice>);


  // shows "error after macro substitution" in ide, but should compile
  INSTANTIATE_WITH_ALL_FLT_TYPES(REGISTER_KERNS);

}
