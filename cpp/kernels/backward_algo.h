#ifndef TFKERNELS_BACKWARD_ALGO_H
#define TFKERNELS_BACKWARD_ALGO_H

#include "../utils/array_infos.h"
#include "lattice.h"

namespace lng {

  template<typename TFLT>
  bool backward_algo_cpu_kernel_launcher(const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas);

  template<typename TFLT>
  bool backward_algo_gpu_kernel_launcher(const Eigen::GpuDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas);

  template<typename TFLT, typename DEVICE>
  class BackwardFunctor {
  public:
    bool operator()(const DEVICE & device, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas);
  };

  template<typename TFLT>
  class BackwardFunctor<TFLT, Eigen::ThreadPoolDevice> {
  public:
    bool operator()(const Eigen::ThreadPoolDevice & device, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas) {
      return backward_algo_cpu_kernel_launcher(device, l, all_betas);
    }
  };

  template<typename TFLT>
  class BackwardFunctor<TFLT, Eigen::GpuDevice> {
  public:
    bool operator()(const Eigen::GpuDevice & device, const Lattice<TFLT> &l, Arr3<TFLT> &all_betas) {
      return backward_algo_gpu_kernel_launcher(device, l, all_betas);
    }
  };



}
#endif //TFKERNELS_BACKWARD_ALGO_H
