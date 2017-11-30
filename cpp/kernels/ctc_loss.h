#ifndef TFKERNELS_CTC_LOSS_H
#define TFKERNELS_CTC_LOSS_H

#include "../utils/array_infos.h"
#include "lattice.h"

namespace lng {

  template<typename TFLT>
  bool ctc_gradient_cpu_kernel_launcher(
      const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l,
      const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
      Arr3<TFLT> &all_gradients);

  template<typename TFLT>
  bool ctc_gradient_gpu_kernel_launcher(
      const Eigen::GpuDevice &d, const Lattice<TFLT> &l,
      const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
      Arr3<TFLT> &all_gradients);

  template<typename TFLT, typename DEVICE>
  class CtcGradientFunctor {
  public:
    bool operator()(
        const DEVICE & device, const Lattice<TFLT> &l,
        const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
        Arr3<TFLT> &all_gradients);
  };

  template<typename TFLT>
  class CtcGradientFunctor<TFLT, Eigen::ThreadPoolDevice> {
  public:
    bool operator()(
        const Eigen::ThreadPoolDevice & device, const Lattice<TFLT> &l,
        const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
        Arr3<TFLT> &all_gradients) {
      return ctc_gradient_cpu_kernel_launcher(device, l, all_alphas, all_betas, all_log_p_l_x, all_gradients);
    }
  };

  template<typename TFLT>
  class CtcGradientFunctor<TFLT, Eigen::GpuDevice> {
  public:
    bool operator()(
        const Eigen::GpuDevice & device, const Lattice<TFLT> &l,
        const Arr3<TFLT> &all_alphas, const Arr3<TFLT> &all_betas, const Arr1<TFLT> &all_log_p_l_x,
        Arr3<TFLT> &all_gradients) {
      return ctc_gradient_gpu_kernel_launcher(device, l, all_alphas, all_betas, all_log_p_l_x, all_gradients);
    }
  };




}

#endif //TFKERNELS_CTC_LOSS_H
