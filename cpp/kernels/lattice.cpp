#include "lattice.h"

#include <iostream>
#include <sstream>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../utils/tensor_helpers.h"
#include "../utils/context_helpers.h"
#include "../utils/timer.h"
#include "hmm_shared.h"


namespace lng {

  using std::cout;
  using std::endl;
  using std::string;
  using std::to_string;
  using std::vector;
  using std::stringstream;

  using namespace tensorflow;


  // Only allow const creation publicly.
  template<typename TFLT>
  const Lattice<TFLT> Lattice<TFLT>::create(OpKernelContext * ctx) {
    return Lattice<TFLT>(ctx);
  }

  template<typename TFLT>
  Lattice<TFLT>::Lattice(OpKernelContext * ctx) {

    const TFLT ZERO_LOG_SPACE = -INFINITY;
    const TFLT ONE_LOG_SPACE = 0.0f;
    SimpleTimer timer;

    inputs= get_input<TFLT, 3>(ctx, 0);
    num_sequences = ctx->input(1).scalar<IntSys>()(0);
    sequence_lengths = get_input<IntSys, 1>(ctx, 2);
    priors = get_input<TFLT, 1>(ctx, 3);

    // get and validate certain constants
    max_T = inputs.dim1();
    K = inputs.dim2();

    // OP_REQUIRES_EQUALS(ctx, num_inputs, num_inputs_expected); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, sequence_lengths.dim0(), num_sequences); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, inputs.dim0(), num_sequences); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, inputs.dim2(), priors.dim0()); RET_IF_BAD;

    // parse lists of graph components

    graphs_num_states = get_input<IntSys, 1>(ctx, 4);
    graphs_num_arcs = get_input<IntSys, 1>(ctx, 5);
    graphs_state_symbols = get_input<IntSys, 2>(ctx, 6);
    graphs_state_weights = get_input<TFLT, 3>(ctx, 7);
    graphs_arcs = get_input<IntSys, 3>(ctx, 8);
    graphs_arc_weights = get_input<TFLT, 2>(ctx, 9);

    max_S = graphs_state_symbols.dim1();
    max_num_arcs = graphs_arcs.dim1();
    OP_REQUIRES_EQUALS(ctx, graphs_state_symbols.dim0(), num_sequences); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, graphs_state_weights.dim1(), max_S); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, graphs_state_weights.dim2(), 2); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, graphs_arcs.dim0(), num_sequences); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, graphs_arcs.dim2(), 2); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, graphs_arc_weights.dim0(), num_sequences); RET_IF_BAD;
    OP_REQUIRES_EQUALS(ctx, graphs_arc_weights.dim1(), max_num_arcs); RET_IF_BAD;
    // printf("Got inputs for lattice: %0.3f sec.\n", timer.since_last());


    // construct arrays to pass to cuda

    // host and device
    attr_hnd.set_on_host(true);
    attr_hnd.set_gpu_compatible(true);

    // all_S, all_T
    all_S = alloc_temp<IntSys, 1>(ctx, {num_sequences}, attr_hnd); RET_IF_BAD;
    all_T = alloc_temp<IntSys, 1>(ctx, {num_sequences}, attr_hnd); RET_IF_BAD;
    for (IntSys i = 0; i < num_sequences; i++) {
      all_S(i) = graphs_num_states(i);
      all_T(i) = sequence_lengths(i);
    }
    // printf("Lattice all_S all_T: %0.3f sec.\n", timer.since_last());


    // validate graph related inputs
    for (IntSys i = 0; i < num_sequences; i++) {
      // validate num states, num arcs, sequence lengths
      IntSys iT = all_T(i);
      IntSys iS = all_S(i);
      IntSys i_num_arcs = graphs_num_arcs(i);
      OP_REQUIRES_TRUE_WITH_MSG(ctx, iT <= max_T, string("Invalid seq length for sample ") + to_string(i)); RET_IF_BAD;
      OP_REQUIRES_TRUE_WITH_MSG(ctx, iS <= max_S, string("Invalid num states for sample ") + to_string(i)); RET_IF_BAD;
      OP_REQUIRES_TRUE_WITH_MSG(ctx, i_num_arcs <= max_num_arcs, string("Invalid num arcs for sample ") + to_string(i)); RET_IF_BAD;

      // validate state symbols, arcs, arc weights
      for (IntSys s = 0; s < iS; s++) {
        IntSys k = graphs_state_symbols(i, s);
        OP_REQUIRES_TRUE_WITH_MSG(ctx, k < K, string("For sample ") + to_string(i) + ", at s " + to_string(s) + ", k must be < " + to_string(K) + ", but was " + to_string(k));
      }
      for (IntSys j = 0; j < i_num_arcs; j++) {
        IntSys prev_state = graphs_arcs(i, j, 0);
        IntSys next_state = graphs_arcs(i, j, 1);;
        TFLT arc_weight = graphs_arc_weights(i, j);
        OP_REQUIRES_TRUE_WITH_MSG(ctx, prev_state < iS, string("For sample ") + to_string(i) + ", at arc " + to_string(j) + ", prev_state must be < " + to_string(iS) + ", but was " + to_string(prev_state)); RET_IF_BAD;
        OP_REQUIRES_TRUE_WITH_MSG(ctx, next_state < iS, string("For sample ") + to_string(i) + ", at arc " + to_string(j) + ", next_state must be < " + to_string(iS) + ", but was " + to_string(next_state)); RET_IF_BAD;
        OP_REQUIRES_TRUE_WITH_MSG(ctx, arc_weight > ZERO_LOG_SPACE, string("For sample ") + to_string(i) + ", at arc " + to_string(j) + ", arc weight must be > -INF, but was " + to_string(arc_weight)); RET_IF_BAD;
      }
    }
    // printf("Lattice validation: %0.3f sec.\n", timer.since_last());

    // mapping used for ctc gradient calcs
    all_s_to_k = alloc_temp<IntSys, 2>(ctx, {num_sequences, max_S}, attr_hnd); RET_IF_BAD;
    all_s_to_k.info().fill(-1);
    for (IntSys i = 0; i < num_sequences; i++) {
      IntSys iS = all_S(i);
      for (IntSys s = 0; s < iS; s++) {
        all_s_to_k(i, s) = graphs_state_symbols(i, s);
      }
    }
    // printf("Lattice s_to_k: %0.3f sec.\n", timer.since_last());


    // TODO calc softmax, log posteriors, log emissions on gpu

    // calc softmax to get posteriors, log posteriors
    all_posteriors = alloc_temp<TFLT, 3>(ctx, {num_sequences, max_T, K}, attr_hnd); RET_IF_BAD;
    all_log_posteriors = alloc_temp<TFLT, 3>(ctx, {num_sequences, max_T, K}, attr_hnd); RET_IF_BAD;
    all_posteriors.info().fill(NAN);
    all_log_posteriors.info().fill(NAN);
    // printf("Lattice alloc posteriors: %0.3f sec.\n", timer.since_last());
    for (IntSys i = 0; i < num_sequences; i++) {
      IntSys iT = all_T(i);
      auto i_input = inputs.info().subarray(i, {iT, K});
      auto i_posteriors = all_posteriors.info().subarray(i, {iT, K});  // TODO use eigen to simplify arraywise ops
      auto i_log_posteriors = all_log_posteriors.info().subarray(i, {iT, K});
      for (IntSys t = 0; t < iT; t++) {

        // take softmax
        // first find max value, to protect against overflow
        TFLT max_logit = -INFINITY;
        for (IntSys k = 0; k < K; k++) {
          if (i_input(t, k) > max_logit) {
            max_logit = i_input(t, k);
          }
        }
        // find sum for denominator
        // subtract max value
        double denom = 0;  // use double precision for summing denom
        for (IntSys k = 0; k < K; k++) {
          i_posteriors(t, k) = exp(i_input(t, k) - max_logit);
          denom += i_posteriors(t, k);
        }
        // check
        OP_REQUIRES_TRUE_WITH_MSG(ctx, denom != 0, string("Inputs summed to 0 for sample ") + to_string(i) + " at time " + to_string(t) );
        RET_IF_BAD;
        // normalize
        for (IntSys k = 0; k < K; k++) {
          i_posteriors(t, k) = i_posteriors(t, k) / denom;

          // log posteriors
          i_log_posteriors(t, k) = log(i_posteriors(t, k));
        }
      }
    }
    // printf("Lattice posteriors softmax and log: %0.3f sec.\n", timer.since_last());



    // emissions
    // = posterior / prior
    // = log(posterior) - log(prior)
    //
    // need to map from k -> s,
    //   ie map from N x max_T x K posteriors to N x max_T x max_S emissions
    all_log_emissions = alloc_temp<TFLT, 3>(ctx, {num_sequences, max_T, max_S}, attr_hnd); RET_IF_BAD;
    all_log_emissions.info().fill(NAN);
    for (IntSys i = 0; i < num_sequences; i++) {
      IntSys iS = all_S(i);
      IntSys iT = all_T(i);

      // calc emissions
      for (IntSys s = 0; s < iS; s++) {
        IntSys k = graphs_state_symbols(i, s);
        TFLT k_prior = priors(k);
        for (IntSys t = 0; t < iT; t++) {
          all_log_emissions(i, t, s) = all_log_posteriors(i, t, k) - log(k_prior);
        }
      }
    }
    // printf("Lattice emissions: %0.3f sec.\n", timer.since_last());

    // graph related arrays

    // tracks start state weights and final state weights
    // n -> s
    log_start_weights = alloc_temp<TFLT, 2>(ctx, {num_sequences, max_S}, attr_hnd); RET_IF_BAD;
    log_final_weights = alloc_temp<TFLT, 2>(ctx, {num_sequences, max_S}, attr_hnd); RET_IF_BAD;

    // tracks number of arcs leading into a state
    // n -> s x 1
    // (all_transitions_from_prev_num)
    num_incoming_arcs = alloc_temp<IntSys, 2>(ctx, {num_sequences, max_S}, attr_hnd); RET_IF_BAD;
    // tracks number of arcs going out of a state
    // n -> s x 1
    // (all_transitions_onto_next_num)
    num_outgoing_arcs = alloc_temp<IntSys, 2>(ctx, {num_sequences, max_S}, attr_hnd); RET_IF_BAD;

    // tracks which was the prev state along kth arc (ith sample, jth next_state, kth arc)
    // n -> s x MAX_NUM_TRANS_PER_STATE
    // (all_transitions_from_prev_s)
    prev_states = alloc_temp<IntSys, 3>(ctx, {num_sequences, MAX_NUM_TRANS_PER_STATE, max_S}, attr_hnd); RET_IF_BAD;
    // tracks which was the next state along kth arc (ith sample, jth prev_state, kth arc)
    // n -> s x MAX_NUM_TRANS_PER_STATE
    // (all_transitions_onto_next_s)
    next_states = alloc_temp<IntSys, 3>(ctx, {num_sequences, MAX_NUM_TRANS_PER_STATE, max_S}, attr_hnd); RET_IF_BAD;

    // tracks arc weight along kth arc
    // n -> s x MAX_NUM_TRANS_PER_STATE
    // (all_transitions_from_prev_val)
    inc_log_arc_weights = alloc_temp<TFLT, 3>(ctx, {num_sequences, MAX_NUM_TRANS_PER_STATE, max_S}, attr_hnd); RET_IF_BAD;
    // tracks arc weight along kth arc
    // n -> s x MAX_NUM_TRANS_PER_STATE
    // (all_transitions_onto_next_val)
    out_log_arc_weights = alloc_temp<TFLT, 3>(ctx, {num_sequences, MAX_NUM_TRANS_PER_STATE, max_S}, attr_hnd); RET_IF_BAD;

    // pre-populate with 0s or invalid values
    log_start_weights.info().fill(NAN);
    log_final_weights.info().fill(NAN);
    num_incoming_arcs.info().fill(0);
    num_outgoing_arcs.info().fill(0);
    prev_states.info().fill(-1);
    next_states.info().fill(-1);
    inc_log_arc_weights.info().fill(NAN);
    out_log_arc_weights.info().fill(NAN);
    // populate
    for (IntSys isample = 0; isample < num_sequences; isample++) {
      IntSys iS = all_S(isample);
      for (IntSys jstate = 0; jstate < iS; jstate++) {
        log_start_weights(isample, jstate) = log(graphs_state_weights(isample, jstate, 0));
        log_final_weights(isample, jstate) = log(graphs_state_weights(isample, jstate, 1));
      }


      for (IntSys jarc = 0; jarc < graphs_num_arcs(isample); jarc++) {
        IntSys prev_state = graphs_arcs(isample, jarc, 0);
        IntSys next_state = graphs_arcs(isample, jarc, 1);;
        TFLT log_arc_weight = log(graphs_arc_weights(isample, jarc));

        // incoming arcs
        IntSys inc_arc_idx = num_incoming_arcs(isample, next_state);
        num_incoming_arcs(isample, next_state)++;
        prev_states(isample, inc_arc_idx, next_state) = prev_state;
        inc_log_arc_weights(isample, inc_arc_idx, next_state) = log_arc_weight;

        // outgoing arcs
        IntSys out_arc_idx = num_outgoing_arcs(isample, prev_state);
        num_outgoing_arcs(isample, prev_state)++;
        next_states(isample, out_arc_idx, prev_state) = next_state;
        out_log_arc_weights(isample, out_arc_idx, prev_state) = log_arc_weight;
      }
    }
    // printf("Lattice graph arrays: %0.3f sec.\n", timer.since_last());

  }


  // instantiate templates
  #define INSTANTIATE(TFLT) \
    template const Lattice<TFLT> Lattice<TFLT>::create(OpKernelContext * ctx); \
    template Lattice<TFLT>::Lattice(OpKernelContext * ctx); \

  INSTANTIATE_WITH_ALL_FLT_TYPES(INSTANTIATE)

}