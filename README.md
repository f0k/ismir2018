Zero-Mean Convolutions for Level-Invariant Singing Voice Detection
==================================================================

This is the implementation of the experiments presented in the paper
"Zero-Mean Convolutions for Level-Invariant Singing Voice Detection" by
Jan Schlüter and Bernhard Lehner at the 19th International Society for
Music Information Retrieval conference (ISMIR 2018).
[[Paper](http://ofai.at/~jan.schlueter/pubs/2018_ismir.pdf),
[BibTeX](http://ofai.at/~jan.schlueter/pubs/2018_ismir.bib)]

Specifically, it includes experiments for predicting singing voice solely from
framewise total magnitudes, and for making a CNN-based singing voice detector
invariant to the sound level.


Preliminaries
-------------

The code requires the following software:
* Python 2.7+ or 3.4+
* Python packages: numpy, scipy, Theano, Lasagne
* bash or a compatible shell with wget and tar
* ffmpeg or avconv

For better performance, the following Python packages are recommended:
* pyfftw (for much faster spectrogram computation)
* scipy version 0.15+ (to allow time stretching and pitch shifting
  augmentations to be parallelized by multithreading, not only by
  multiprocessing, https://github.com/scipy/scipy/pull/3943)

For Theano and Lasagne, you may need the bleeding-edge versions from github.
In short, they can be installed with:
```bash
pip install --upgrade --no-deps https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip
```
(Add `--user` to install in your home directory, or `sudo` to install globally.)
For more complete installation instructions including GPU setup, please refer
to the [From Zero to Lasagne](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne)
guides.

On Ubuntu, pyfftw can be installed with the following two commands:
```bash
sudo apt-get install libfftw3-dev
pip install pyfftw
```


Setup
-----

For preparing the experiments, clone the repository somewhere and checkout the
`phd_extra` branch:
```bash
git clone https://github.com/f0k/ismir2018.git
git checkout phd_extra
```
If you do not have `git` available, download the code from
https://github.com/f0k/ismir2018/archive/master.zip and extract it.

The experiments use the public [Jamendo dataset by Mathieu Ramona](www.mathieuramona.com/wp/data/jamendo/).
To download and prepare it, open the cloned or extracted repository in a
bash terminal and execute the following scripts (in this order):
```bash
./datasets/jamendo/audio/recreate.sh
./datasets/jamendo/filelists/recreate.sh
./datasets/jamendo/labels/recreate.sh
```

The experiments also use an extended dataset described in ["Online,
Loudness-Invariant Vocal Detection in Mixed Music
Signals"](http://www.ofai.at/~jan.schlueter/pubs/2018_tasl.pdf).
We are currently working on making it available.


Experiments
-----------

### Motivational Example

To demonstrate the dependency of singing voice ground truth to input magnitude,
we evaluated how well we can classify singing voice with a simple threshold on
the summarized spectral magnitudes of the Jamendo dataset. To reproduce this
toy experiment, run:
```bash
cd experiments
mkdir -p cache/jamendo
mkdir loudnesses
./loudness_extract.py --dataset=jamendo --cache=cache/jamendo loudnesses/
./loudness_eval.py loudnesses/jamendo_gt.pkl loudnesses/jamendo_*sum.pkl
```
This will extract summarized magnitudes and store them in pickle files in the
`loudnesses` subdirectory, for different spectrogram variants. It will then
output classification accuracies for an all-vocal baseline and for thresholds
chosen on the training set. Spectrograms will be cached in the `cache` directory
in a subfolder that will be reused in the next experiment -- if you'd like to
store them somewhere else or not at all, change or omit the `--cache` argument.
When passing `--plot`, it will also create histogram plots in the `loudnesses`
subdirectory akin to Figure 1 of the paper. Passing `--smooth=56` shows how
results can be slightly improved by temporal smoothing before thresholding.

### Level-Invariant CNNs

To train the baseline network and five methods for improving invariance to the
sound level, simply run:
```bash
cd experiments
mkdir cache
./train_all.sh
```
This will train 6 network variants with 5 repetitions each. On an Nvidia GTX
970 GPU, a single training run will take about 4 hours. Spectrograms will
be cached in the `cache` directory -- if you'd like to store them in `/tmp`
instead, replace `mkdir cache` with `ln -s /tmp cache` or edit the
`train_all.sh` file where it says `--cache=cache`.

If you have multiple GPUs, you can distribute runs over these GPUs by running
the script multiple times in multiple terminals with different target devices,
e.g., `THEANO_FLAGS=device=cuda1 ./train_all.sh`. If you have multiple servers
that can access the same directory via NFS, you can also run the script on
each server for further distribution of runs (runs are blocked with lockfiles).

The script will also compute network predictions after each training run, but
for the original recording level only. To compute predictions for all networks
at the original level as well as six different input gains, run:
```bash
./predict_all.sh
```
This will only compute missing network predictions (if none are missing, nothing
happens). On a GTX 970, it will take about 3 minutes per network and gain for
most variants, 9 minutes for PCEN and 100 minutes with instance normalization.
Just as before, this can be distributed over multiple GPUs or multiple servers
with shared NFS access.

Finally, to evaluate all networks at all gain factors, run:
```bash
./eval_all.sh | tee eval_all.txt
```
For each network, this computes the optimal classification threshold on the
validation set at the original sound level and uses that threshold to evaluate
the test set at all gain factors (-9dB, -6dB, -3dB, 0dB, +3dB, +6dB, +9dB), as
well as the validation set (not reported in paper). Results are printed to
screen and written to a text file in tab-separated format, so they are easy to
read in using a script or spreadsheet application. Those are the results
visualized in Figure 3 of the paper.

### Benchmark

Table 1 of the paper compares the different network variants in terms of
computation time required to process one hour of audio. To reproduce these
numbers, you'll first need an hour of audio. We'll set up a fake dataset for
this purpose, using `sox` to create the audio file:
```bash
cd ../datasets
mkdir onehour
cd onehour
mkdir audio filelists
sox -n audio/onehour.wav synth 3600 sin 400 sin 401
for _ in {1..10}; do echo onehour.wav >> filelists/tenhours; done
```
We also created a file list repeating that audio file name ten times, for better
performance measurement. Now we can compute predictions for all models:
```bash
for modelfile in ismir2018/{baseline,pcenfixalpha,zeromeanconv}_1.npz; do
  ./predict.py --dataset=onehour --filelists=tenhours --cache=cache "$m" foo.pkl
done
```
The very first one will have to compute the spectrogram, the others will read it
from the cache, so you may want to run the first one again. For instance
normalization, we need to add a flag to predict excerpt by excerpt:
```bash
./predict.py --dataset=onehour --filelists=tenhours --cache=cache \
  ismir2018/instnorm_1.npz foo.pkl --mem-use=low
```
We omit `loudaugment_1.npz` since it is equivalent to the baseline at test time.
By repeating these calls with different `THEANO_FLAGS` for `device=cpu` and
`device=cuda` (or `device=cuda1`, `device=cuda2`) you can benchmark the CPU and
different GPU devices if available.


About...
--------

### ... the code

This is an adaption of the code for singing voice detection experiments in the
first author's PhD thesis: https://github.com/f0k/ismir2015/tree/phd_extra
The commit history retains all modifications from that starting point.

This code can serve as a template for your own sequence labelling experiments.
Some interesting features are:
* Datasets can easily be added to the `datasets` directory and their name be
  passed as the `--dataset` argument of `train.py`, `predict.py` and `eval.py`.
* With `--load-spectra=on-demand`, `train.py` can efficiently handle datasets
  too large to fit into main memory; when placing the spectrogram cache on a
  SSD, it can be fast enough to not stall the GPU.
* Both `train.py` and `predict.py` accept key-value settings in the form
  `--var key=value`, these can be thought of global environment variables that
  you can access anywhere in the code. The `defaults.vars` files stores
  default key-value settings to be overridden via `--var key=value`, and
  `train.py` stores the settings used for training along with the trained
  network model, automatically obeyed by `predict.py`. This allows to easily
  extend any part of the code without breaking existing experiments: implement
  a new feature conditioned on a key-value setting, add this setting to
  `defaults.vars` such that default behaviour stays the same, add a new line
  to `train_all.sh` that passes different `--var key=value` settings, and run
  the script. See the changeset of commit d5158ad for an example.

The advantage of using such a template over creating a more generic experiment
framework is that you have direct control of all dataset preparation, model
creation and training code and a low hurdle to modify things, the disadvantage
is that it becomes harder to reuse code.

### ... the results

Results will vary depending on the random initialization of the networks. Even
with fixed random seeds, results will not be exactly reproducible due to the
multi-threaded data augmentation. Furthermore, when training on GPU with cuDNN,
the backward pass is nondeterministic by default, introducing further noise.

Furthermore, it makes a difference which tool is used for decoding and
resampling the audio files. We used `ffmpeg` 2.8.14 (Ubuntu 16.04) for all
experiments, which produces slightly different outputs than `avconv` 9.20
(Ubuntu 14.04) -- be careful not to mix different tools when working in a
heterogeneous environment.
