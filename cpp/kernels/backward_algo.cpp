#include <iostream>
#include <sstream>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"
#include "backward_algo.h"

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

  REGISTER_OP("HmmBackward")
      .Attr("TFLT: {float, double}")
          LATTICE_INPUTS_TO_REGISTER_OP
      .Output("all_betas: TFLT");         // 3d tensor: (num_samples, max_T, max_S)









  template<typename TFLT>
  void backward_probs_cpu_kernel(
      const IntSys num_sequences,
      const Arr1Int all_S,
      const Arr1Int all_T,

      // n -> (T, S)
      const Arr3<TFLT> all_emissions,

      // n -> (S,)
      const Arr2<TFLT> all_final_weights,
      const Arr2Int all_num_outgoing_arcs,
      // n -> (MAX_NUM_TRANS_PER_STATE, S)
      const Arr3Int all_next_states,
      const Arr3<TFLT> all_out_arc_weights,

      // n -> (T, S)
      Arr3<TFLT> all_betas
  )
  {
    const TFLT ZERO_LOG_SPACE = -INFINITY;
    // const TFLT ONE_LOG_SPACE = 0.0f;

    for (IntSys seq_idx = 0; seq_idx < num_sequences; seq_idx++) {
      IntSys S = all_S(seq_idx);
      IntSys T = all_T(seq_idx);

      auto betas = all_betas.subarray(seq_idx, {T, S});
      auto emissions = all_emissions.subarray(seq_idx, {T, S});

      auto final_weights      = all_final_weights.subarray(seq_idx, {S});
      auto num_outgoing_arcs  = all_num_outgoing_arcs.subarray(seq_idx, {S});
      auto next_states        = all_next_states.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});
      auto out_arc_weights    = all_out_arc_weights.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});

      for (IntSys t = T-1; t >= 0; t--) {
        for (IntSys s = 0; s < S; s++) {
          TFLT beta;
          if (t == T-1) {
            beta = final_weights(s);
          } else {

            IntSys num_trans = num_outgoing_arcs(s);
            beta = ZERO_LOG_SPACE;
            for (IntSys i_trans = 0; i_trans < MAX_NUM_TRANS_PER_STATE; i_trans++) {  // static num of loop iters
              if (i_trans < num_trans) {
                IntSys s_next = next_states(i_trans, s);
                TFLT trans = out_arc_weights(i_trans, s);
                TFLT beta_next = betas(t+1, s_next);
                TFLT emit_next = emissions(t+1, s_next);

                beta = add_log_space(beta, beta_next + trans + emit_next);
              }
            }
          }

          if (DO_NAN_CHECK) {
            if (isnan(beta)) {
              printf("Found NAN beta for sequence %d (t,s) (%d, %d).", seq_idx, t, s);
            }
          }

          // store
          betas(t, s) = beta;

        } // s
      } // t
    }

  }


  template<typename TFLT>
  bool backward_algo_cpu_kernel_launcher(const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas) {

    backward_probs_cpu_kernel(
        l.num_sequences, l.all_S.info(), l.all_T.info(),
            l.all_log_emissions.info(),
            l.log_final_weights.info(), l.num_outgoing_arcs.info(), l.next_states.info(), l.out_log_arc_weights.info(),
            all_betas);

    return true;
  }




  template<typename TFLT, typename DEVICE>
  class HmmBackwardOp : public OpKernel {
  public:
    explicit HmmBackwardOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      auto lat = Lattice<TFLT>::create(ctx); RET_IF_BAD;

      auto all_betas = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 0, lat.attr_hnd); RET_IF_BAD;
      all_betas.info().fill(NAN);

      bool launch_res = BackwardFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_betas.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch backward algorithm kernel.");
    }
  };


  #define REGISTER_KERNS(TFLT)   \
    template bool backward_algo_cpu_kernel_launcher<TFLT>(const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas); \
    REGISTER_KERNEL_BUILDER(      \
    Name("HmmBackward")            \
      .Device(DEVICE_GPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmBackwardOp<TFLT, Eigen::GpuDevice>); \
    REGISTER_KERNEL_BUILDER(      \
    Name("HmmBackward")            \
      .Device(DEVICE_CPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmBackwardOp<TFLT, Eigen::ThreadPoolDevice>);


  // shows "error after macro substitution" in ide, but should compile
  INSTANTIATE_WITH_ALL_FLT_TYPES(REGISTER_KERNS);






}
