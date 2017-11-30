# LatticeCtc

A Tensorflow extension to calculate CTC loss against lattices instead of linear sequences. 

The CTC Loss function currently provided in Tensorflow allows you to optimize sequential models like RNNs without having to provide an alignment of targets against every frame of input. However, you must provide a linear sequence of targets. This extension allows you to provide a lattice of targets, ie a directed, possibly cyclic, graph such as outputs from FST composition. The CTC loss is similarly calculated using the forward-backward algorithm over all valid paths through the lattice.

For instance, in speech recognition, you would have to train some initial models to infer the correct pronunciation of a transcript. Either you would train a full recipe starting from HMM GMMs or you could train an initial DNN by making initial assumptions about pronunciations, then redecode to get a better inference about the pronunciations. This extension allows you to encode the multiple possible pronunciations into your FST, and train immediately on all the possibilities. It is essentially similar to the way HMMs are trained.

Tensorflow's implementation is fully CPU based. This extension is compiled both for CPU and GPUs. However, in order to fit on a GPU, the lattices should not be too large. Long sequences may need to be partitioned.

This lattice training could theoretically be used to train non CTC forward backwards algorithms (eg typical HMM EM training where there is no CTC blank) or for the numerator and denominator lattices in sequence discriminiative training, like MMI or MPE. These have not been explored yet.

Documentation is currently limited, but if you're interested, raise an issue and express your interest, and I'll work to flesh it out.


