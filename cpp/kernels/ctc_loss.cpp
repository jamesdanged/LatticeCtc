#include "ctc_loss.h"

#include <iostream>
#include <sstream>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"
#include "../utils/math_utils.h"
#include "../utils/timer.h"
#include "forward_algo.h"
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

  REGISTER_OP("HmmCtcLoss")
      .Attr("TFLT: {float, double}")
          LATTICE_INPUTS_TO_REGISTER_OP
      .Output("loss: TFLT")               // 1d tensor: (num_samples,)
      .Output("all_gradients: TFLT")      // 3d tensor: (num_samples, max_T, K)
      .Doc(R"doc(
)doc");




  template<typename TFLT>
  void ctc_gradient_cpu_kernel(
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

    for (IntSys seq_idx = 0; seq_idx < num_sequences; seq_idx++) {
      IntSys S = all_S(seq_idx);
      IntSys T = all_T(seq_idx);
      IntSys K = all_posteriors.dim2();
      TFLT log_p_l_x = all_log_p_l_x(seq_idx);

      auto s_to_k = all_s_to_k.subarray(seq_idx, {S});
      auto alphas = all_alphas.subarray(seq_idx, {T, S});
      auto betas = all_betas.subarray(seq_idx, {T, S});
      auto posteriors = all_posteriors.subarray(seq_idx, {T, K});
      auto gradients = all_gradients.subarray(seq_idx, {T, K});

      // populate gradients with 0
      // (Only within T for each sequence. Rest leave as NaN to catch errors.)
      for (IntSys t = 0; t < T; t++) {
        for (IntSys k = 0; k < K; k++) {
          gradients(t, k) = ZERO_LOG_SPACE;
        }
      }

      // sum over each s as it matches up with k
      for (IntSys s = 0; s < S; s++) {
        IntSys k = s_to_k(s);
        for (IntSys t = 0; t < T; t++) {
          TFLT alpha = alphas(t, s);
          TFLT beta = betas(t, s);

          gradients(t, k) = add_log_space(gradients(t, k), alpha + beta);
        }
      }
      // apply rest of gradient formula (with respect to activations, not posteriors)
      // ie
      //   d L/d a_tk = y_tk - 1/p_l_x * sum_s:sym=k( alphas(s,t) * beta(s,t) )
      for (IntSys k = 0; k < K; k++) {
        for (IntSys t = 0; t < T; t++) {
          gradients(t, k) = posteriors(t, k) - exp(gradients(t, k) - log_p_l_x);
        }
      }

    }

  }



  template<typename TFLT>
  bool ctc_gradient_cpu_kernel_launcher(
      const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l,
      const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
      Arr3<TFLT> &all_gradients
  ) {
    ctc_gradient_cpu_kernel(
        l.num_sequences, l.all_S.info(), l.all_T.info(), l.all_s_to_k.info(),
            all_alphas, all_betas, all_log_p_l_x, l.all_posteriors.info(), all_gradients);

    return true;
  }





  template<typename TFLT, typename DEVICE>
  class HmmCtcLossOp : public OpKernel {
  public:
    explicit HmmCtcLossOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {

      const TFLT ZERO_LOG_SPACE = -INFINITY;
      SimpleTimer timer;


      auto lat = Lattice<TFLT>::create(ctx); RET_IF_BAD;

      auto all_alphas = alloc_temp<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, lat.attr_hnd); RET_IF_BAD;
      auto all_betas = alloc_temp<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.max_S}, lat.attr_hnd); RET_IF_BAD;
      all_alphas.info().fill(NAN);
      all_betas.info().fill(NAN);
      // printf("Created alphas, betas: %0.3f sec.\n", timer.since_last());

      // p(l | x) for each sequence
      // log space
      auto all_log_p_l_x = alloc_temp<TFLT, 1>(ctx, {lat.num_sequences}, lat.attr_hnd); RET_IF_BAD;
      all_log_p_l_x.info().fill(NAN);

      // ctc loss
      auto total_loss = alloc_output<TFLT, 1>(ctx, {lat.num_sequences}, 0, lat.attr_hnd); RET_IF_BAD;
      total_loss.info().fill(ZERO_LOG_SPACE);

      // gradients
      auto all_gradients = alloc_output<TFLT, 3>(ctx, {lat.num_sequences, lat.max_T, lat.K}, 1, lat.attr_hnd); RET_IF_BAD;
      // fill with 0 because the extra times for shorter sequences should
      // backprop a gradient of 0.
      all_gradients.info().fill(0);
      // printf("Created other arrays: %0.3f sec.\n", timer.since_last());

      // const GPUDevice & device = ctx->eigen_device<GPUDevice>();
      bool launch_res = ForwardFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_alphas.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch forward algorithm kernel.");
      // printf("\t\tRan forward: %0.3f sec.\n", timer.since_last());

      launch_res = BackwardFunctor<TFLT, DEVICE>()(ctx->eigen_device<DEVICE>(), lat, all_betas.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch backward algorithm kernel.");
      // printf("\t\tRan backward: %0.3f sec.\n", timer.since_last());

      // loss (negative log likelihood)
      // sum over all states (all valid final states),
      // take the final alpha and final state weight, take negative
      // sync_gpu(ctx);

      for (IntSys i = 0; i < lat.num_sequences; i++) {
        IntSys t_to_use = 0; // lat.all_T(i) - 1;
        // TODO maybe use eigen to reduce sum faster and simpler on gpu
        IntSys iS = lat.all_S(i);
        TFLT sum_over_s = ZERO_LOG_SPACE;
        for (IntSys s = 0; s < iS; s++) {
          TFLT final_alpha = all_alphas(i, t_to_use, s);
          TFLT final_beta = all_betas(i, t_to_use, s);
          sum_over_s = add_log_space(sum_over_s, final_alpha + final_beta);
        }
        all_log_p_l_x(i) = sum_over_s;
        total_loss(i) = -sum_over_s;
      }
      // printf("Loss sum on cpu: %0.3f sec.\n", timer.since_last());

      // gradients
      launch_res = CtcGradientFunctor<TFLT, DEVICE>()(
          ctx->eigen_device<DEVICE>(), lat,
          all_alphas.info(), all_betas.info(), all_log_p_l_x.info(), all_gradients.info());
      check_kernel_launch(ctx, launch_res, "Failed to launch ctc gradient kernel.");
      // sync_gpu(ctx);
      // printf("\t\tGradients kernel: %0.3f sec.\n", timer.since_last());

    }



  };


  #define REGISTER_KERNS(TFLT)   \
  template bool ctc_gradient_cpu_kernel_launcher( \
      const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, \
      const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x, \
      Arr3<TFLT> &all_gradients); \
  REGISTER_KERNEL_BUILDER(      \
    Name("HmmCtcLoss")            \
      .Device(DEVICE_GPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmCtcLossOp<TFLT, Eigen::GpuDevice>); \
  REGISTER_KERNEL_BUILDER(      \
    Name("HmmCtcLoss")            \
      .Device(DEVICE_CPU)       \
      LATTICE_INPUTS_TO_REGISTER_KERNEL \
      .TypeConstraint<TFLT>("TFLT"),  \
    HmmCtcLossOp<TFLT, Eigen::ThreadPoolDevice>);


  // shows "error after macro substitution" in ide, but should compile
  INSTANTIATE_WITH_ALL_FLT_TYPES(REGISTER_KERNS);





}

