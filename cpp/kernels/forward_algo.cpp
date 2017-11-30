#include <iostream>
#include <sstream>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"
#include "forward_algo.h"

#include "hmm_shared.h"
#include "../utils/math_utils.h"
#include <cmath>

namespace lng {

  using std::cout;
  using std::endl;
  using std::string;
  using std::to_string;
  using std::vector;
  using std::stringstream;
  using std::isnan;



  using namespace tensorflow;

  REGISTER_OP("HmmForward")
      .Attr("TFLT: {float, double}")
          LATTICE_INPUTS_TO_REGISTER_OP
      .Output("all_alphas: TFLT");         // 3d tensor: (num_samples, max_T, max_S)













  template<typename TFLT>
  void forward_probs_cpu_kernel(
      const IntSys num_sequences,
      const Arr1Int all_S,
      const Arr1Int all_T,

      // n -> (T, S)
      const Arr3<TFLT> all_emissions,

      // n -> (S,)
      const Arr2<TFLT> all_start_weights,
      const Arr2Int all_num_incoming_arcs,
      // n -> (MAX_NUM_TRANS_PER_STATE, S)
      const Arr3Int all_prev_states,
      const Arr3<TFLT> all_inc_arc_weights,

      // n -> (T, S)
      Arr3<TFLT> all_alphas
  )
  {

    const TFLT ZERO_LOG_SPACE = -INFINITY;
    // const TFLT ONE_LOG_SPACE = 0.0f;

    for (IntSys seq_idx = 0; seq_idx < num_sequences; seq_idx++) {

      IntSys S = all_S(seq_idx);
      IntSys T = all_T(seq_idx);

      auto alphas = all_alphas.subarray(seq_idx, {T, S});
      auto emissions = all_emissions.subarray(seq_idx, {T, S});

      auto start_weights      = all_start_weights.subarray(seq_idx, {S});
      auto num_incoming_arcs  = all_num_incoming_arcs.subarray(seq_idx, {S});
      auto prev_states        = all_prev_states.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});
      auto inc_arc_weights    = all_inc_arc_weights.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});

      for (IntSys t = 0; t < T; t++) {
        for (IntSys s = 0; s < S; s++) {

            TFLT alpha;
            if (t == 0) {
              alpha = start_weights(s) + emissions(t, s);
            } else {

              IntSys num_trans = num_incoming_arcs(s);
              alpha = ZERO_LOG_SPACE;
              for (IntSys i_trans = 0; i_trans < MAX_NUM_TRANS_PER_STATE; i_trans++) {  // static num of loop iters
                if (i_trans < num_trans) {
                  IntSys s_prev = prev_states(i_trans, s);
                  TFLT trans = inc_arc_weights(i_trans, s);
                  TFLT alpha_from = alphas(t - 1, s_prev);

                  alpha = add_log_space(alpha, alpha_from + trans);
                }
              }
              alpha = alpha + emissions(t, s);

            }

            if (DO_NAN_CHECK) {
              if (isnan(alpha)) {
                printf("Found NAN alpha for sequence %d (t, s) (%d, %d).", seq_idx, t, s);
              }
            }

            // store
            alphas(t, s) = alpha;

        } // s
      } // t

    }
  }

  template<typename TFLT>
  bool forward_algo_cpu_kernel_launcher(const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas) {

    forward_probs_cpu_kernel(
        l.num_sequences, l.all_S.info(), l.all_T.info(),
            l.all_log_emissions.info(),
            l.log_start_weights.info(), l.num_incoming_arcs.info(), l.prev_states.info(), l.inc_log_arc_weights.info(),
            all_alphas);

    return true;
  }


  template<typename TFLT, typename DEVICE>
  class HmmForwardOp : public OpKernel {
  public:
    explicit HmmForwardOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      auto lat = Lattice<TFLT>::create(ctx); RET_IF_BAD;

      auto all_alphas = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 0, lat.attr_hnd); RET_IF_BAD;
      all_alphas.info().fill(NAN);

//      bool launch_res = forward_algo_gpu_kernel_launcher(ctx->eigen_gpu_device(), lat, all_alphas.info());
      bool launch_res = ForwardFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_alphas.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch forward algorithm kernel.");
    }
  };





  #define REGISTER_KERNS(TFLT)   \
    template bool forward_algo_cpu_kernel_launcher<TFLT>(const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas); \
    REGISTER_KERNEL_BUILDER(      \
    Name("HmmForward")            \
      .Device(DEVICE_GPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmForwardOp<TFLT, Eigen::GpuDevice>); \
    \
    REGISTER_KERNEL_BUILDER(      \
    Name("HmmForward")            \
      .Device(DEVICE_CPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmForwardOp<TFLT, Eigen::ThreadPoolDevice>);



  // shows "error after macro substitution" in ide, but should compile
  INSTANTIATE_WITH_ALL_FLT_TYPES(REGISTER_KERNS);


}
