import tensorflow as tf

lng_kernels = tf.load_op_library('libLatticeCtc.so')

# kernels for testing
zero_out_kernel = lng_kernels.zero_out
add_one_kernel = lng_kernels.add_one
mul_by_two_kernel = lng_kernels.mul_by_two
jagged_edit_kernel = lng_kernels.jagged_edit
pass_array_kernel = lng_kernels.pass_array

hmm_forward_kernel = lng_kernels.hmm_forward
hmm_backward_kernel = lng_kernels.hmm_backward
hmm_viterbi_kernel = lng_kernels.hmm_viterbi
hmm_combined_kernel = lng_kernels.hmm_combined
hmm_ctc_loss_kernel = lng_kernels.hmm_ctc_loss

