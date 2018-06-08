#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular base name for the weights and
# predictions files. Each single repetition checks if it was already run or is
# currently being run, creates a lockfile, trains the network, computes the
# predictions, and removes the lockfile. To distribute runs between multiple
# GPUs, just run this script multiple times with different THEANO_FLAGS.

train_if_free() {
	modelfile="$1"
	echo "$modelfile"
	if [ ! -f "$modelfile" ] && [ ! -f "$modelfile.lock" ]; then
		echo "$HOSTNAME: $THEANO_FLAGS" > "$modelfile.lock"
		OMP_NUM_THREADS=1 ./train.py "$modelfile" --dataset=lehner --augment --cache=cache --load-spectra=on-demand --var eta_decay=0.1 --var eta_decay_every=trial_of_patience --var patience=10 --var trials_of_patience=3 --var epochs=1000 --no-validate --save-errors "${@:2}"
		OMP_NUM_THREADS=1 THEANO_FLAGS=allow_gc=1,dnn.conv.algo_fwd=guess_on_shape_change,dnn.conv.algo_bwd_data=guess_on_shape_change,dnn.conv.algo_bwd_filter=guess_on_shape_change,"$THEANO_FLAGS" ./predict.py "$modelfile" "${modelfile%.npz}.pred.pkl" --dataset=lehner --cache=cache
		rm "$modelfile.lock"
	fi
}

train() {
	repeats="$1"
	name="$2"
	for (( r=1; r<=$repeats; r++ )); do
		train_if_free "$name"$r.npz "${@:3}"
	done
}


mkdir -p ismir2018
train 5 ismir2018/baseline_
train 5 ismir2018/loudaugment_ --var max_loud=10
train 5 ismir2018/instnorm_ --var input_norm=instance
train 5 ismir2018/timediff_ --var arch.timediff=1
train 5 ismir2018/convzeromean_ --var arch.firstconv_zeromean=params
#train 5 ismir2018/pcen_ --var magscale=pcen
train 5 ismir2018/pcenfixalpha_ --var magscale=pcen --var pcen_fix_alpha=1
