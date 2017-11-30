#include "viterbi_algo.h"

#include <iostream>
#include <sstream>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"

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

  REGISTER_OP("HmmViterbi")
      .Attr("TFLT: {float, double}")
          LATTICE_INPUTS_TO_REGISTER_OP
      .Output("all_deltas: TFLT")         // 3d tensor: (num_samples, max_T, max_S)
      .Output("all_tracebacks: int32")    // 3d tensor: (num_samples, max_T, max_S)
      .Output("all_best_paths: int32")  ;  // 2d tensor: (num_samples, max_T)    Best state path (s's). Need to convert to symbols (k's).



  template<typename TFLT>
  void viterbi_cpu_kernel(
      const IntSys num_sequences,
      const Arr1Int all_S,
      const Arr1Int all_T,

      // n -> (T, S)
      const Arr3<TFLT> all_emissions,

      // n -> (S,)
      const Arr2<TFLT> all_start_weights,
      const Arr2<TFLT> all_final_weights,
      const Arr2Int all_num_incoming_arcs,
      // n -> (MAX_NUM_TRANS_PER_STATE, S)
      const Arr3Int all_prev_states,
      const Arr3<TFLT> all_inc_arc_weights,

      // n -> (T, S)
      Arr3<TFLT> all_deltas,
      Arr3Int all_tracebacks,
      Arr2Int all_best_paths
  ) {

    const TFLT ZERO_LOG_SPACE = -INFINITY;

    for (IntSys seq_idx = 0; seq_idx < num_sequences; seq_idx++) {
      IntSys S = all_S(seq_idx);
      IntSys T = all_T(seq_idx);

      auto deltas = all_deltas.subarray(seq_idx, {T, S});
      auto traceback = all_tracebacks.subarray(seq_idx, {T, S});
      auto best_path = all_best_paths.subarray(seq_idx, {T});
      auto emissions = all_emissions.subarray(seq_idx, {T, S});

      auto start_weights          = all_start_weights.subarray(seq_idx, {S});
      auto final_weights          = all_final_weights.subarray(seq_idx, {S});
      auto num_incoming_arcs      = all_num_incoming_arcs.subarray(seq_idx, {S});
      auto prev_states            = all_prev_states.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});
      auto inc_arc_weights        = all_inc_arc_weights.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});

      for (IntSys t = 0; t < T; t++) {
        for (IntSys s = 0; s < S; s++) {
          TFLT best_delta;
          IntSys best_s_prev = -1;
          if (t == 0) {
            best_delta = start_weights(s) + emissions(t, s);
            best_s_prev = -1;
          } else {

            IntSys num_trans = num_incoming_arcs(s);
            best_delta = ZERO_LOG_SPACE;

            for (IntSys i_trans = 0; i_trans < MAX_NUM_TRANS_PER_STATE; i_trans++) {  // static num of loop iters
              if (i_trans < num_trans) {
                IntSys s_prev = prev_states(i_trans, s);
                TFLT trans = inc_arc_weights(i_trans, s);
                TFLT delta_from = deltas(t - 1, s_prev);

                TFLT curr_delta = delta_from + trans;
                if (curr_delta > best_delta) {
                  best_delta = curr_delta;
                  best_s_prev = s_prev;
                }
              }
            }
            best_delta = best_delta + emissions(t, s);

          }

          if (DO_NAN_CHECK) {
            if (isnan(best_delta)) {
              printf("Found NAN delta for sequence %d (t, s) (%d, %d).", seq_idx, t, s);
            }
          }

          // store
          deltas(t, s) = best_delta;
          traceback(t, s) = best_s_prev;

        } // s
      } // t

      // extract best path
      // find best final state, incorporating final weights
      IntSys best_final_state = -1;
      TFLT best_delta_plus_final = ZERO_LOG_SPACE;
      for (IntSys s = 0; s < S; s++) {
        TFLT curr_delta_plus_final = deltas(T-1, s) + final_weights(s);
        if (curr_delta_plus_final > best_delta_plus_final) {
          best_delta_plus_final = curr_delta_plus_final;
          best_final_state = s;
        }
      }

      // follow traceback
      best_path(T-1) = best_final_state;
      IntSys curr_state = best_final_state;
      for (IntSys t = T-1; t > 0; t--) {
        curr_state = traceback(t, curr_state);
        best_path(t-1) = curr_state;
      }

    } // seq_idx

  }





  template<typename TFLT>
  bool viterbi_algo_cpu_kernel_launcher(
      const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l,
      Arr3<TFLT> &all_deltas, Arr3Int &all_tracebacks, Arr2Int &all_best_paths) {

    viterbi_cpu_kernel(l.num_sequences, l.all_S.info(), l.all_T.info(),
        l.all_log_emissions.info(),
        l.log_start_weights.info(), l.log_final_weights.info(),
        l.num_incoming_arcs.info(), l.prev_states.info(), l.inc_log_arc_weights.info(),
        all_deltas, all_tracebacks, all_best_paths);

    return true;
  }








  template<typename TFLT, typename DEVICE>
  class HmmViterbiOp : public OpKernel {
  public:
    explicit HmmViterbiOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      auto lat = Lattice<TFLT>::create(ctx); RET_IF_BAD;

      auto all_deltas = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 0, lat.attr_hnd); RET_IF_BAD;
      all_deltas.info().fill(NAN);
      auto all_tracebacks = alloc_output<IntSys, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, 1, lat.attr_hnd); RET_IF_BAD;
      all_tracebacks.info().fill(-1);
      auto all_best_paths = alloc_output<IntSys, 2>(ctx, {lat.num_sequences, lat.max_T}, 2, lat.attr_hnd); RET_IF_BAD;
      all_best_paths.info().fill(-1);

      bool launch_res = ViterbiFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_deltas.info(),
                                                         all_tracebacks.info(), all_best_paths.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch viterbi algorithm kernel.");
    }
  };


  #define REGISTER_KERNS(TFLT)   \
    template bool viterbi_algo_cpu_kernel_launcher<TFLT>( \
      const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, \
      Arr3<TFLT> &all_deltas, Arr3Int & all_tracebacks, Arr2Int & best_paths); \
    REGISTER_KERNEL_BUILDER(      \
    Name("HmmViterbi")            \
      .Device(DEVICE_GPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmViterbiOp<TFLT, Eigen::GpuDevice>); \
    REGISTER_KERNEL_BUILDER(      \
    Name("HmmViterbi")            \
      .Device(DEVICE_CPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmViterbiOp<TFLT, Eigen::ThreadPoolDevice>);


  // shows "error after macro substitution" in ide, but should compile
  INSTANTIATE_WITH_ALL_FLT_TYPES(REGISTER_KERNS);

}
