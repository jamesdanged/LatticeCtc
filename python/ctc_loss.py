import tensorflow as tf
from tensorflow.python.framework import ops

from latticectc.base import hmm_ctc_loss_kernel


def hmm_ctc_loss(inputs, num_samples, sequence_lengths, priors,
                 graphs_num_states, graphs_num_arcs,
                 graphs_state_symbols, graphs_state_weights,
                 graphs_arcs, graphs_arc_weights,):
    """Runs ctc loss."""

    # ignore the gradients, they are used internally
    loss, _ = hmm_ctc_loss_kernel(
        inputs, num_samples, sequence_lengths, priors,
        graphs_num_states, graphs_num_arcs,
        graphs_state_symbols, graphs_state_weights,
        graphs_arcs, graphs_arc_weights,)

    return loss


@ops.RegisterGradient("HmmCtcLoss")
def _ctc_loss_grad(op, back_prop_grad, _):
    """The derivative provided by CTC loss.

    Args:
      op: the HmmCtcLoss op.
      back_prop_grad: ctc loss backpropped gradient.
      _: grad of the gradient (since gradient is the 2nd output). Just ignore.

    Returns:
      The ctc loss gradient.
    """

    internal_gradient = op.outputs[1]
    grad_result = tf.reshape(
        back_prop_grad,
        [tf.shape(back_prop_grad)[0], 1, 1]
    ) * internal_gradient

    return [
        grad_result,     # inputs
        None,            # num_samples
        None,            # sequence_lengths
        None,            # priors
        None,            # num states
        None,            # num arcs
        None,            # state symbols
        None,            # state weights
        None,            # arcs
        None,            # arc weights
    ]

