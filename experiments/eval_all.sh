#!/bin/bash

# Evaluates all available predictions in sets of five repetitions and six gain
# factors. For each network, will determine a classification threshold on the
# validation set at the original sound level, then apply it to the test set at
# different gains.
# Note that this will produce a ton of output, you may want to tee it into a
# text file with: ./eval_all.sh | tee eval_all.txt

evaluate() {
	name="$1"
	predfile="$2"
	part="$3"
#	for subset in jamendo msd100 rwc_pop yt_classics_song all_song rwc_classical rwc_jazz yt_classics_instr yt_guitars yt_heavy_instr yt_wind_flute yt_wind_sax all_instr; do
#		printf "%s\t%s\t" "$name" "$subset"
#		./eval.py "$predfile" --dataset=lehner --test-list="$part"."$subset" "${@:3}" | cut -d' ' -f12
#	done
	printf "%s\t%s\t" "$name" "all_$part"
	./eval.py "$predfile" --dataset=lehner --test-list="$part" "${@:4}" --save-rawdata="${predfile%.pred.pkl}.eval.npy" | cut -d' ' -f12
}

evaluate_all_gains() {
	modelfile="$1"
	results=$(./eval.py "${modelfile%.npz}.pred.pkl" --dataset=lehner --test-list=valid --save-rawdata="${modelfile%npz}eval.npy")
	threshold=$(echo $results | cut -c6-9)
	for part in test valid; do
		evaluate "$modelfile 0dB" "${modelfile%.npz}.pred.pkl" "$part" --threshold "$threshold"
		for l in 3 6 9; do
			evaluate "$modelfile +${l}dB" "${modelfile%.npz}.p$l.pred.pkl" "$part" --threshold "$threshold"
			evaluate "$modelfile -${l}dB" "${modelfile%.npz}.m$l.pred.pkl" "$part" --threshold "$threshold"
		done
	done
}

for i in {1..5}; do
	evaluate_all_gains ismir2018/baseline_$i.npz
	evaluate_all_gains ismir2018/loudaugment_$i.npz
	evaluate_all_gains ismir2018/instnorm_$i.npz
	evaluate_all_gains ismir2018/timediff_$i.npz
	evaluate_all_gains ismir2018/convzeromean_$i.npz
	#evaluate_all_gains ismir2018/pcen_$i.npz
	evaluate_all_gains ismir2018/pcenfixalpha_$i.npz
done
