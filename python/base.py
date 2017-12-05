import tensorflow as tf

lng_kernels = tf.load_op_library('libLatticeCtc.so')

hmm_forward_kernel = lng_kernels.hmm_forward
hmm_backward_kernel = lng_kernels.hmm_backward
hmm_viterbi_kernel = lng_kernels.hmm_viterbi
hmm_combined_kernel = lng_kernels.hmm_combined
hmm_ctc_loss_kernel = lng_kernels.hmm_ctc_loss

