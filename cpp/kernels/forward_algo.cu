#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "../utils/array_infos.h"
#include "../utils/math_utils.h"
#include "../utils/cuda_includes_for_ide.h"
#include "forward_algo.h"
#include "hmm_shared.h"

namespace lng {



  /**
   * Calculates alphas to populate the trellis.
   *
   * Uses one thread block per sequence.
   * Each thread moves across time.
   * If there are more than 32 states, the 32 threads in a block
   * iterate through the states in chunks of 32.
   *
   */
  template<typename TFLT>
  __global__
  void forward_probs_gpu_kernel(
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

    IntSys seq_idx = blockIdx.x;
    IntSys thread_idx = threadIdx.x;
    if (seq_idx >= num_sequences) return;

    IntSys S = all_S(seq_idx);
    IntSys T = all_T(seq_idx);
    IntSys num_s_blocks = div_round_up(S, HMM_BLOCK_DIM_EXPANDED);

    auto alphas = all_alphas.subarray(seq_idx, {T, S});
    auto emissions = all_emissions.subarray(seq_idx, {T, S});

    auto start_weights      = all_start_weights.subarray(seq_idx, {S});
    auto num_incoming_arcs  = all_num_incoming_arcs.subarray(seq_idx, {S});
    auto prev_states        = all_prev_states.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});
    auto inc_arc_weights    = all_inc_arc_weights.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});

    for (IntSys t = 0; t < T; t++) {
      for (IntSys s_block = 0; s_block < num_s_blocks; s_block++) {
        IntSys s = s_block * HMM_BLOCK_DIM_EXPANDED + thread_idx;
        if (s < S) {

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

        } // if
      } // s_block

      // omitting this assumes warp synchronous programming, specifically that the block size is 32, same as warp size
      if (HMM_BLOCK_DIM_EXPANDED != 32) __syncthreads();
    } // t

  }



  template<typename TFLT>
  bool forward_algo_gpu_kernel_launcher(const Eigen::GpuDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas) {

    forward_probs_gpu_kernel <<<l.num_sequences, HMM_BLOCK_DIM_EXPANDED, 0, d.stream()>>>(
      l.num_sequences, l.all_S.info(), l.all_T.info(),
      l.all_log_emissions.info(),
      l.log_start_weights.info(), l.num_incoming_arcs.info(), l.prev_states.info(), l.inc_log_arc_weights.info(),
      all_alphas);

    d.synchronize();
    return d.ok();
  }



#define KERNEL_INSTANTIATE(TFLT) \
  template __global__ void forward_probs_gpu_kernel<TFLT>(  \
    const IntSys num_sequences, \
    const Arr1Int all_S,  \
    const Arr1Int all_T,   \
    const Arr3<TFLT> all_emissions,  \
    const Arr2<TFLT> all_start_weights, \
    const Arr2Int all_num_incoming_arcs,  \
    const Arr3Int all_prev_states,  \
    const Arr3<TFLT> all_inc_arc_weights,  \
    Arr3<TFLT> all_alphas  \
  ); \
  template bool forward_algo_gpu_kernel_launcher<TFLT>( \
    const Eigen::GpuDevice & d,  \
    const Lattice<TFLT> & l,   \
    Arr3<TFLT> & all_alphas  \
  );


  INSTANTIATE_WITH_ALL_FLT_TYPES(KERNEL_INSTANTIATE)


}




