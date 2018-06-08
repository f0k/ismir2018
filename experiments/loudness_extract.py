#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes summarized magnitudes for some spectrogram variants.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from progress import progress
from simplecache import cached
import audio
import znorm
from labels import create_aligned_targets
import config


def opts_parser():
    descr = "Computes summarized magnitudes for some spectrogram variants."
    parser = ArgumentParser(description=descr)
    parser.add_argument('outdir',
            type=str,
            help='Directy to save summarized magnitudes to')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--load-spectra',
            choices=('memory', 'memmap', 'on-demand'), default='memory',
            help='By default, spectrograms are loaded to memory. Large '
                 'datasets can be read as memory-mapped files (if not '
                 'exceeding the allowable number of open files) or read '
                 'on-demand. The latter two require --cache-spectra.')
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE '
                 'lines. Can be given multiple times, settings from later '
                 'files overriding earlier ones. Will read defaults.vars, '
                 'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
                 'settings from --vars options. Can be given multiple times.')
    return parser


def save_spectral_sums(outfile, spects):
    print(outfile)
    with io.open(outfile, 'wb') as f:
        pickle.dump({'sums': [spect[:,].sum(axis=-1) for spect in spects]}, f,
                    protocol=-1)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    outdir = options.outdir
    if options.load_spectra != 'memory' and not options.cache_spectra:
        parser.error('option --load-spectra=%s requires --cache-spectra' %
                     options.load_spectra)

    # read configuration files and immediate settings
    cfg = {}
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))
    cfg.update(config.parse_variable_assignments(options.var))

    # read some settings into local variables
    sample_rate = cfg['sample_rate']
    frame_len = cfg['frame_len']
    fps = cfg['fps']
    mel_bands = cfg['mel_bands']
    mel_min = cfg['mel_min']
    mel_max = cfg['mel_max']

    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    # - load filelist
    filelist = []
    ranges = {}
    for part in 'train', 'valid', 'test':
        a = len(filelist)
        with io.open(os.path.join(datadir, 'filelists',
                                  cfg.get('filelist.%s' % part, part))) as f:
            filelist.extend(l.rstrip() for l in f if l.rstrip())
        ranges[part] = slice(a, len(filelist))

    # - compute spectra
    print("Computing%s spectra..." %
          (" or loading" if options.cache_spectra else ""))
    spects = []
    for fn in progress(filelist, 'File '):
        cache_fn = (options.cache_spectra and
                    os.path.join(options.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn,
                             audio.extract_spect,
                             os.path.join(datadir, 'audio', fn),
                             sample_rate, frame_len, fps,
                             loading_mode=options.load_spectra))

    # - load and convert corresponding labels
    print("Loading labels...")
    labels = []
    for fn, spect in zip(filelist, spects):
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(spect)) / float(fps)
        labels.append(create_aligned_targets(segments, timestamps, np.bool))

    # compute and save different variants of summarized magnitudes
    print("Saving files...")

    # - ground truth
    outfile = os.path.join(outdir, '%s_gt.pkl' % options.dataset)
    print(outfile)
    with io.open(outfile, 'wb') as f:
        pickle.dump({'labels': labels, 'splits': ranges}, f, protocol=-1)
    
    # - summarized spectra
    save_spectral_sums(
            os.path.join(outdir, '%s_spect_sum.pkl' % options.dataset),
            spects)
    
    # - summarized mel spectra
    bank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                       mel_min, mel_max).astype(np.float32)
    spects = [np.dot(spect[:,], bank) for spect in spects]
    save_spectral_sums(
            os.path.join(outdir, '%s_spect_mel_sum.pkl' % options.dataset),
            spects)

    # - summarized log-mel spectra
    spects = [np.log(np.maximum(1e-7, spect)) for spect in spects]
    save_spectral_sums(
            os.path.join(outdir, '%s_spect_mel_log_sum.pkl' % options.dataset),
            spects)

    # - summarized standardized log-mel spectra
    m, s = znorm.compute_mean_std(spects[ranges['train']], axis=0)
    spects = [((spect - m) / s).astype(np.float32) for spect in spects]
    save_spectral_sums(
            os.path.join(outdir, '%s_spect_mel_log_std_sum.pkl' %
                         options.dataset),
            spects)

if __name__=="__main__":
    main()
