#define EIGEN_USE_GPU
#include "viterbi_algo.h"

#include "tensorflow/core/framework/register_types.h"
#include "../utils/array_infos.h"
#include "../utils/math_utils.h"
#include "../utils/cuda_includes_for_ide.h"
#include "hmm_shared.h"

namespace lng {


  /**
   * Calculates viterbi path (deltas).
   *
   * Uses one thread block per sequence.
   * Each thread moves across time.
   * If there are more than 32 states, the 32 threads in a block
   * iterate through the states in chunks of 32.
   *
   */
  template<typename TFLT>
  __global__
  void viterbi_gpu_kernel(
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
    // const TFLT ONE_LOG_SPACE = 0.0f;

    IntSys seq_idx = blockIdx.x;
    IntSys thread_idx = threadIdx.x;
    if (seq_idx >= num_sequences) return;

    IntSys S = all_S(seq_idx);
    IntSys T = all_T(seq_idx);
    IntSys num_s_blocks = div_round_up(S, HMM_BLOCK_DIM_EXPANDED);

    auto deltas = all_deltas.subarray(seq_idx, {T, S});
    auto traceback = all_tracebacks.subarray(seq_idx, {T, S});
    auto best_path = all_best_paths.subarray(seq_idx, {T});
    auto emissions = all_emissions.subarray(seq_idx, {T, S});

    auto start_weights = all_start_weights.subarray(seq_idx, {S});
    auto final_weights = all_final_weights.subarray(seq_idx, {S});
    auto num_incoming_arcs = all_num_incoming_arcs.subarray(seq_idx, {S});
    auto prev_states = all_prev_states.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});
    auto inc_arc_weights = all_inc_arc_weights.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});

    for (IntSys t = 0; t < T; t++) {
      for (IntSys s_block = 0; s_block < num_s_blocks; s_block++) {
        IntSys s = s_block * HMM_BLOCK_DIM_EXPANDED + thread_idx;
        if (s < S) {

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

        } // if
      } // s_block

      if (HMM_BLOCK_DIM_EXPANDED != 32) __syncthreads();
    } // t

    // extract best path
    if (thread_idx == 0) {
      // find best final state, incorporating final weights
      IntSys best_final_state = -1;
      TFLT best_delta_plus_final = ZERO_LOG_SPACE;
      for (IntSys s = 0; s < S; s++) {
        TFLT curr_delta_plus_final = deltas(T - 1, s) + final_weights(s);
        if (curr_delta_plus_final > best_delta_plus_final) {
          best_delta_plus_final = curr_delta_plus_final;
          best_final_state = s;
        }
      }

      // follow traceback
      best_path(T - 1) = best_final_state;
      IntSys curr_state = best_final_state;
      for (IntSys t = T - 1; t > 0; t--) {
        curr_state = traceback(t, curr_state);
        best_path(t - 1) = curr_state;
      }
    }


  }


  template<typename TFLT>
  bool viterbi_algo_gpu_kernel_launcher(
      const Eigen::GpuDevice &d, const Lattice<TFLT> &l,
      Arr3<TFLT> &all_deltas, Arr3Int &all_tracebacks, Arr2Int &all_best_paths) {

    viterbi_gpu_kernel << < l.num_sequences, HMM_BLOCK_DIM_EXPANDED, 0, d.stream() >> >
                                                                        (l.num_sequences, l.all_S.info(), l.all_T.info(),
                                                                            l.all_log_emissions.info(),
                                                                            l.log_start_weights.info(), l.log_final_weights.info(),
                                                                            l.num_incoming_arcs.info(), l.prev_states.info(), l.inc_log_arc_weights.info(),
                                                                            all_deltas, all_tracebacks, all_best_paths);

    return d.ok();
  }


#define KERNEL_INSTANTIATE(TFLT) \
  template __global__ void viterbi_gpu_kernel<TFLT>(  \
    const IntSys num_sequences, \
    const Arr1Int all_S,  \
    const Arr1Int all_T,   \
    const Arr3<TFLT> all_emissions,  \
    const Arr2<TFLT> all_start_weights, \
    const Arr2<TFLT> all_final_weights, \
    const Arr2Int all_num_incoming_arcs,  \
    const Arr3Int all_prev_states,  \
    const Arr3<TFLT> all_inc_arc_weights,  \
    Arr3<TFLT> all_deltas,  \
    Arr3Int all_tracebacks, \
    Arr2Int all_best_paths  \
  ); \
  template bool viterbi_algo_gpu_kernel_launcher<TFLT>( \
    const Eigen::GpuDevice & d,  \
    const Lattice<TFLT> & l,   \
    Arr3<TFLT> & all_deltas,  \
    Arr3Int & all_tracebacks, \
    Arr2Int & all_best_paths \
  );

  INSTANTIATE_WITH_ALL_FLT_TYPES(KERNEL_INSTANTIATE)

}







