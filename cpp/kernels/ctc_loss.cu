#define EIGEN_USE_GPU
#include "ctc_loss.h"

#include "tensorflow/core/framework/register_types.h"
#include "../utils/array_infos.h"
#include "../utils/math_utils.h"
#include "../utils/cuda_includes_for_ide.h"
#include "hmm_shared.h"


namespace lng {

  /**
   * Calculates gradients for ctc.
   *
   * Uses one thread block per sequence.
   * Each thread moves across states, ie each thread handles a different time t.
   * Don't need __syncthreads() because each t is independent, no clashing.
   *
   * If T > HMM_BLOCK_DIM, the HMM_BLOCK_DIM threads in a block
   * iterate through time in chunks of HMM_BLOCK_DIM.
   *
   */
  template<typename TFLT>
  __global__
  void ctc_gradient_gpu_kernel(
      const IntSys num_sequences,
      const Arr1Int all_S,
      const Arr1Int all_T,

      // n -> (S,)
      const Arr2Int all_s_to_k,

      // n -> (T, S)     log space
      const Arr3<TFLT> all_alphas,
      const Arr3<TFLT> all_betas,

      // (n,)    log space
      const Arr1<TFLT> all_log_p_l_x,

      // n x max_T x K, not log space
      const Arr3<TFLT> all_posteriors,

      // n x max_T x K
      Arr3<TFLT> all_gradients

  ) {
    const TFLT ZERO_LOG_SPACE = -INFINITY;

    IntSys seq_idx = blockIdx.x;
    IntSys thread_idx = threadIdx.x;
    if (seq_idx >= num_sequences) return;

    IntSys S = all_S(seq_idx);
    IntSys T = all_T(seq_idx);
    IntSys K = all_posteriors.dim2();
    IntSys num_t_blocks = div_round_up(T, HMM_BLOCK_DIM_EXPANDED);
    TFLT log_p_l_x = all_log_p_l_x(seq_idx);

    auto s_to_k = all_s_to_k.subarray(seq_idx, {S});
    auto alphas = all_alphas.subarray(seq_idx, {T, S});
    auto betas = all_betas.subarray(seq_idx, {T, S});
    auto posteriors = all_posteriors.subarray(seq_idx, {T, K});
    auto gradients = all_gradients.subarray(seq_idx, {T, K});

    // populate gradients with 0
    // (Only within T for each sequence. Rest leave as NaN to catch errors.)
    for (IntSys t_block = 0; t_block < num_t_blocks; t_block++) {
      IntSys t = t_block * HMM_BLOCK_DIM_EXPANDED + thread_idx;
      if (t < T) {
        for (IntSys k = 0; k < K; k++) {
          gradients(t, k) = ZERO_LOG_SPACE;
        }
      }
    }

    // sum over each s as it matches up with k
    for (IntSys s = 0; s < S; s++) {
      IntSys k = s_to_k(s);
      for (IntSys t_block = 0; t_block < num_t_blocks; t_block++) {
        IntSys t = t_block * HMM_BLOCK_DIM_EXPANDED + thread_idx;
        if (t < T) {
          TFLT alpha = alphas(t, s);
          TFLT beta = betas(t, s);

          gradients(t, k) = add_log_space(gradients(t, k), alpha + beta);
        }
      }
    }
    // apply rest of gradient formula (with respect to activations, not posteriors)
    // ie
    //   d L/d a_tk = y_tk - 1/p_l_x * sum_s:sym=k( alphas(s,t) * beta(s,t) )
    for (IntSys k = 0; k < K; k++) {
      for (IntSys t_block = 0; t_block < num_t_blocks; t_block++) {
        IntSys t = t_block * HMM_BLOCK_DIM_EXPANDED + thread_idx;
        if (t < T) {
          gradients(t, k) = posteriors(t, k) - exp(gradients(t, k) - log_p_l_x);
        }
      }
    }




  }


  template<typename TFLT>
  bool ctc_gradient_gpu_kernel_launcher(
      const Eigen::GpuDevice &d, const Lattice<TFLT> &l,
      const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
      Arr3<TFLT> &all_gradients
  ) {
    ctc_gradient_gpu_kernel <<<l.num_sequences, HMM_BLOCK_DIM_EXPANDED, 0, d.stream()>>>(
        l.num_sequences, l.all_S.info(), l.all_T.info(), l.all_s_to_k.info(),
        all_alphas, all_betas, all_log_p_l_x, l.all_posteriors.info(), all_gradients);

    d.synchronize();
    return d.ok();
  }



#define KERNEL_INSTANTIATE(TFLT) \
  template __global__ void ctc_gradient_gpu_kernel<TFLT>(  \
    const IntSys num_sequences, \
    const Arr1Int all_S,  \
    const Arr1Int all_T,   \
    const Arr2Int all_s_to_k, \
    const Arr3<TFLT> all_alphas,  \
    const Arr3<TFLT> all_betas,  \
    const Arr1<TFLT> all_log_p_l_x, \
    const Arr3<TFLT> all_posteriors,  \
    Arr3<TFLT> all_gradients \
  ); \
  template bool ctc_gradient_gpu_kernel_launcher<TFLT>( \
    const Eigen::GpuDevice & d,  \
    const Lattice<TFLT> & l, \
    const Arr3<TFLT> & all_alphas,  \
    const Arr3<TFLT> & all_betas,  \
    const Arr1<TFLT> & all_log_p_l_x, \
    Arr3<TFLT> & all_gradients \
  );

  INSTANTIATE_WITH_ALL_FLT_TYPES(KERNEL_INSTANTIATE)


}
