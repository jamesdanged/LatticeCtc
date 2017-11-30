#define EIGEN_USE_GPU

#include "backward_algo.h"

#include "tensorflow/core/framework/register_types.h"
#include "../utils/array_infos.h"
#include "../utils/math_utils.h"
#include "../utils/cuda_includes_for_ide.h"
#include "hmm_shared.h"


namespace lng {


  /**
   * Calculates betas to populate the trellis.
   *
   * Uses one thread block per sequence.
   * Each thread moves across time.
   * If there are more than 32 states, the 32 threads in a block
   * iterate through the states in chunks of 32.
   *
   */
  template<typename TFLT>
  __global__
  void backward_probs_gpu_kernel(
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


    IntSys seq_idx = blockIdx.x;
    IntSys thread_idx = threadIdx.x;
    if (seq_idx >= num_sequences) return;

    IntSys S = all_S(seq_idx);
    IntSys T = all_T(seq_idx);
    IntSys num_s_blocks = div_round_up(S, HMM_BLOCK_DIM_EXPANDED);

    auto betas = all_betas.subarray(seq_idx, {T, S});
    auto emissions = all_emissions.subarray(seq_idx, {T, S});

    auto final_weights      = all_final_weights.subarray(seq_idx, {S});
    auto num_outgoing_arcs  = all_num_outgoing_arcs.subarray(seq_idx, {S});
    auto next_states        = all_next_states.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});
    auto out_arc_weights    = all_out_arc_weights.subarray(seq_idx, {MAX_NUM_TRANS_PER_STATE, S});

    for (IntSys t = T-1; t >= 0; t--) {
      for (IntSys s_block = 0; s_block < num_s_blocks; s_block++) {
        IntSys s = s_block * HMM_BLOCK_DIM_EXPANDED + thread_idx;
        if (s < S) {

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
              printf("Found NAN beta for sequence %ld (t,s) (%ld, %ld).", seq_idx, t, s);
            }
          }

          // store
          betas(t, s) = beta;

        } // if
      } // s_block

      if (HMM_BLOCK_DIM_EXPANDED != 32) __syncthreads();
    } // t

  }




  template<typename TFLT>
  bool backward_algo_gpu_kernel_launcher(const Eigen::GpuDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas) {

    backward_probs_gpu_kernel <<<l.num_sequences, HMM_BLOCK_DIM_EXPANDED, 0, d.stream()>>>(
        l.num_sequences, l.all_S.info(), l.all_T.info(),
        l.all_log_emissions.info(),
        l.log_final_weights.info(), l.num_outgoing_arcs.info(), l.next_states.info(), l.out_log_arc_weights.info(),
        all_betas);

    d.synchronize();
    return d.ok();
  }


#define KERNEL_INSTANTIATE(TFLT) \
  template __global__ void backward_probs_gpu_kernel<TFLT>(  \
    const IntSys num_sequences, \
    const Arr1Int all_S,  \
    const Arr1Int all_T,   \
    const Arr3<TFLT> all_emissions,  \
    const Arr2<TFLT> all_final_weights, \
    const Arr2Int all_num_outgoing_arcs,  \
    const Arr3Int all_next_states,  \
    const Arr3<TFLT> all_out_arc_weights,  \
    Arr3<TFLT> all_betas \
  ); \
  template bool backward_algo_gpu_kernel_launcher<TFLT>( \
    const Eigen::GpuDevice & d,  \
    const Lattice<TFLT> & l,   \
    Arr3<TFLT> & all_betas  \
  );

  INSTANTIATE_WITH_ALL_FLT_TYPES(KERNEL_INSTANTIATE)



}
