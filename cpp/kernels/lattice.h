#ifndef TFKERNELS_LATTICE_H
#define TFKERNELS_LATTICE_H

#include "../utils/core_types.h"
#include "../utils/cuda_includes_minimal.h"
#include "../utils/array_infos.h"
#include "../utils/decl_cpu_only.h"
#include "../utils/tensor_helpers.h"

// All inputs are in normal (non-log) space.
//
//      inputs: TFLT            //  3d tensor: num_samples * max_T * K.   Logits, pre softmax. Gradient is taken wrt this.
//      num_samples: int32      //  scalar
//      sequence_lengths: int32 //  1d tensor: num_samples
//      priors: TFLT            //  1d tensor: K
//
//      graphs_num_states: int32       // 1d tensor: num_samples
//      graphs_num_arcs: int32         // 1d tensor: num_samples
//      graphs_state_symbols: int32    // 2d tensor: num_samples * max_S
//      graphs_state_weights: TFLT     // 3d tensor: num_samples * max_S x 2          (start_weight, final_weight)
//      graphs_arcs: int32             // 3d tensor: num_samples * max_num_arcs x 2   (prev_state, next_state)
//      graphs_arc_weights: TFLT       // 2d tensor: num_samples * max_num_arcs



#define LATTICE_INPUTS_TO_REGISTER_OP \
  .Input("inputs: TFLT") \
  .Input("num_sequences: int32") \
  .Input("sequence_lengths: int32") \
  .Input("priors: TFLT")     \
  .Input("graphs_num_states: int32") \
  .Input("graphs_num_arcs: int32") \
  .Input("graphs_state_symbols: int32") \
  .Input("graphs_state_weights: TFLT") \
  .Input("graphs_arcs: int32") \
  .Input("graphs_arc_weights: TFLT")



#define LATTICE_INPUTS_TO_REGISTER_KERNEL \
  .HostMemory("inputs")  \
  .HostMemory("num_sequences")  \
  .HostMemory("sequence_lengths")  \
  .HostMemory("priors")  \
  .HostMemory("graphs_num_states") \
  .HostMemory("graphs_num_arcs") \
  .HostMemory("graphs_state_symbols")  \
  .HostMemory("graphs_state_weights")  \
  .HostMemory("graphs_arcs")  \
  .HostMemory("graphs_arc_weights")



// decl
namespace tensorflow {
  struct OpKernelContext;
}

namespace lng {

  template<typename TFLT>
  class Lattice {
  private:
    Lattice() {};
    Lattice(tensorflow::OpKernelContext * ctx);
  public:
    static const Lattice<TFLT> create(tensorflow::OpKernelContext *);

    template<IntSys NDIMS> using TwFlt = TensorWrapper<TFLT, NDIMS>;
    template<IntSys NDIMS> using TwInt = TensorWrapper<IntSys, NDIMS>;

    IntSys num_sequences = 0;
    TwInt<1> sequence_lengths;
    TwFlt<3> inputs;            // not log
    TwFlt<1> priors;            // not log

    IntSys max_S = 0;
    IntSys max_T = 0;   // T and max_T include start, final non-emitting states. Posteriors is therefore N x (max_T-2) x K
    IntSys K = 0;
    IntSys max_num_arcs = 0;

    TwInt<1> graphs_num_states;       // (N,)
    TwInt<1> graphs_num_arcs;         // (N,)
    TwInt<2> graphs_state_symbols;    // (N, max_S)
    TwFlt<3> graphs_state_weights;    // (N, max_S, 2)         not log
    TwInt<3> graphs_arcs;             // (N, max_num_arcs, 2)
    TwFlt<2> graphs_arc_weights;      // (N, max_num_arcs)     not log

    TwInt<1> all_S;                   // (N,)
    TwInt<1> all_T;                   // (N,)
    TwInt<2> all_s_to_k;              // (N, max_S)

    TwFlt<3> all_posteriors;          // (N, max_T, K)         not log
    TwFlt<3> all_log_posteriors;      // (N, max_T, K)         log
    TwFlt<3> all_log_emissions;       // (N, max_T, max_S)     log

    TwFlt<2> log_start_weights;       // (N, max_S)
    TwFlt<2> log_final_weights;       // (N, max_S)
    TwInt<2> num_incoming_arcs;       // (N, max_S)
    TwInt<2> num_outgoing_arcs;       // (N, max_S)
    TwInt<3> prev_states;             // (N, MAX_NUM_TRANS_PER_STATE, max_S)
    TwInt<3> next_states;             // (N, MAX_NUM_TRANS_PER_STATE, max_S)
    TwFlt<3> inc_log_arc_weights;     // (N, MAX_NUM_TRANS_PER_STATE, max_S)      log
    TwFlt<3> out_log_arc_weights;     // (N, MAX_NUM_TRANS_PER_STATE, max_S)      log

    AllocatorAttributes attr_hnd;
  };

}

#endif //TFKERNELS_LATTICE_H
