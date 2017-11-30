#ifndef TFKERNELS_FORWARD_ALGO_H
#define TFKERNELS_FORWARD_ALGO_H

#include "../utils/array_infos.h"
#include "lattice.h"


namespace lng {

  template<typename TFLT>
  bool forward_algo_cpu_kernel_launcher(const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas);

  template<typename TFLT>
  bool forward_algo_gpu_kernel_launcher(const Eigen::GpuDevice &d, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas);

  template<typename TFLT, typename DEVICE>
  class ForwardFunctor {
  public:
    bool operator()(const DEVICE & device, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas);
  };

  template<typename TFLT>
  class ForwardFunctor<TFLT, Eigen::ThreadPoolDevice> {
  public:
    bool operator()(const Eigen::ThreadPoolDevice & device, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas) {
      return forward_algo_cpu_kernel_launcher(device, l, all_alphas);
    }
  };

  template<typename TFLT>
  class ForwardFunctor<TFLT, Eigen::GpuDevice> {
  public:
    bool operator()(const Eigen::GpuDevice & device, const Lattice<TFLT> &l, Arr3<TFLT> &all_alphas) {
      return forward_algo_gpu_kernel_launcher(device, l, all_alphas);
    }
  };







}
#endif //TFKERNELS_FORWARD_ALGO_H
