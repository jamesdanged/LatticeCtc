#ifndef TFKERNELS_VITERBI_ALGO_H
#define TFKERNELS_VITERBI_ALGO_H

#include "../utils/array_infos.h"
#include "lattice.h"


namespace lng {

  template<typename TFLT>
  bool viterbi_algo_cpu_kernel_launcher(
      const Eigen::ThreadPoolDevice &d, const Lattice<TFLT> &l,
      Arr3<TFLT> &all_deltas, Arr3Int & all_tracebacks, Arr2Int & best_paths);

  template<typename TFLT>
  bool viterbi_algo_gpu_kernel_launcher(
      const Eigen::GpuDevice &d, const Lattice<TFLT> &l,
      Arr3<TFLT> &all_deltas, Arr3Int & all_tracebacks, Arr2Int & best_paths);

  template<typename TFLT, typename DEVICE>
  class ViterbiFunctor {
  public:
    bool operator()(
        const DEVICE & device, const Lattice<TFLT> &l,
        Arr3<TFLT> &all_deltas, Arr3Int & all_tracebacks, Arr2Int & best_paths);
  };

  template<typename TFLT>
  class ViterbiFunctor<TFLT, Eigen::ThreadPoolDevice> {
  public:
    bool operator()(
        const Eigen::ThreadPoolDevice & device, const Lattice<TFLT> &l,
        Arr3<TFLT> &all_deltas, Arr3Int & all_tracebacks, Arr2Int & best_paths) {
      return viterbi_algo_cpu_kernel_launcher(device, l, all_deltas, all_tracebacks, best_paths);
    }
  };

  template<typename TFLT>
  class ViterbiFunctor<TFLT, Eigen::GpuDevice> {
  public:
    bool operator()(
        const Eigen::GpuDevice & device, const Lattice<TFLT> &l,
        Arr3<TFLT> &all_deltas, Arr3Int & all_tracebacks, Arr2Int & best_paths) {
      return viterbi_algo_gpu_kernel_launcher(device, l, all_deltas, all_tracebacks, best_paths);
    }
  };


}

#endif //TFKERNELS_VITERBI_ALGO_H
