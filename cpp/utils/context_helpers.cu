#include "../utils/decl_gpu_only.h"



namespace lng {

  void sync_gpu(const Eigen::GpuDevice & gpu_device) {
    gpu_device.synchronize();
  }

}