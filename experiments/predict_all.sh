#!/bin/bash

# Computes network predictions for all trained networks that do not have their
# predictions computed yet, for the original recordings and six different
# input gains. If all predictions are available, nothing happens.

predict() {
	if [ ! -f "$2" ]; then
		touch "$2"
		echo "$2"
		THEANO_FLAGS=allow_gc=1,dnn.conv.algo_fwd=guess_on_shape_change,dnn.conv.algo_bwd_data=guess_on_shape_change,dnn.conv.algo_bwd_filter=guess_on_shape_change,"$THEANO_FLAGS" ./predict.py --dataset=lehner --cache=cache "$@" || rm "$2"
	fi
}

predict_all_gains() {
	modelfile="$1"
	predict "$modelfile" "${modelfile%.npz}.pred.pkl" "${@:2}"
	for l in 3 6 9; do
		predict "$modelfile" "${modelfile%.npz}.p$l.pred.pkl" --loudness +$l "${@:2}"
		predict "$modelfile" "${modelfile%.npz}.m$l.pred.pkl" --loudness -$l "${@:2}"
	done
}

for i in {1..5}; do
	predict_all_gains ismir2018/baseline_$i.npz
	predict_all_gains ismir2018/loudaugment_$i.npz
	predict_all_gains ismir2018/instnorm_$i.npz --mem-use=low
	predict_all_gains ismir2018/timediff_$i.npz
	predict_all_gains ismir2018/convzeromean_$i.npz
	#predict_all_gains ismir2018/pcen_$i.npz
	predict_all_gains ismir2018/pcenfixalpha_$i.npz
done
