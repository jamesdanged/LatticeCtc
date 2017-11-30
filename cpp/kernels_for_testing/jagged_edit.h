#ifndef TFKERNELS_JAGGED_EDIT_H
#define TFKERNELS_JAGGED_EDIT_H

#include "../utils/array_infos.h"

namespace lng {

  void jagged_edit_kernel_launcher(const Arrays2dFlt & arrs_in, Arrays2dFlt & arrs_out);

}

#endif //TFKERNELS_JAGGED_EDIT_H
