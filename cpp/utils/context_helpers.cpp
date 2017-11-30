#include "context_helpers.h"

#include "tensorflow/core/framework/op_kernel.h"


namespace lng {

  using namespace tensorflow;

  void check_kernel_launch(OpKernelContext* context, bool launch_result, StringPiece message) {
    if (!launch_result) {
      context->CtxFailureWithWarning(Status(tensorflow::error::UNKNOWN, message));
    }
  }

  void sync_gpu(OpKernelContext * context) {
    sync_gpu(context->eigen_gpu_device());
  }

}