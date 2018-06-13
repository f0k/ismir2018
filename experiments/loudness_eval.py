#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains and tests a loudness-threshold classifier for every given .pkl file.

For usage information, call without any parameters.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
from argparse import ArgumentParser

import numpy as np
import scipy.optimize


def opts_parser():
    descr = ("Trains and tests a loudness-threshold classifier for every given "
             ".pkl file.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('targets',
            type=str,
            help='a pickle file with ground truth targets')
    parser.add_argument('inputs', nargs='+',
            type=str,
            help='a pickle file with loudnesses (summarized magnitudes)')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, saves a histogram plot next to each input file.')
    parser.add_argument('--plot-data',
            action='store_true',
            help='If given along with --plot, saves the histogram values next '
                 'to each plot.')
    parser.add_argument('--smooth-width', metavar='WIDTH',
            type=int, default=0,
            help='Optionally apply temporal median smoothing over WIDTH frames '
                 '(default: (default)s)')
    return parser


def error(thr, inputs, targets):
    return 1. - np.mean((inputs >= thr) == targets)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    targetfile = options.targets
    inputfiles = options.inputs

    # load targets
    targetdata = np.load(targetfile)
    targets = targetdata['labels']
    splits = targetdata['splits']
    targets_train = np.concatenate(targets[splits['train']])

    # print baseline
    for t in 0, 1:
        print("baseline (all %d)" % t)
        for part in 'train', 'valid', 'test':
            print("%s err: %.4g" %
                  (part,
                   np.abs(t - np.mean(np.concatenate(targets[splits[part]])))))
    
    # iterate over the inputs
    for inputfile in inputfiles:
        print(inputfile)
        inp = np.load(inputfile)['sums']
        if options.smooth_width:
            from scipy.ndimage.filters import median_filter
            inp = [median_filter(p, options.smooth_width, mode='nearest')
                   for p in inp]
        inp_train = np.concatenate(inp[splits['train']])
        thr = .5 * (inp_train.min() + inp_train.max())
        thr = scipy.optimize.fmin(error, thr, args=(inp_train, targets_train),
                                  disp=False)[0]
        print("threshold: %f" % thr)
        for part in 'train', 'valid', 'test':
            print("%s err: %.4g" %
                  (part, error(thr, np.concatenate(inp[splits[part]]),
                               np.concatenate(targets[splits[part]]))))

        if options.plot:
            import matplotlib.pyplot as plt
            if not 'log' in inputfile:
                low = inp_train.max() / 10**(50. / 20)  # cutoff at -50 dB FS
            else:
                k = int(len(inp_train) * .02)  # cutoff at 2-percentile
                low = np.partition(inp_train, k)[k]
            np.maximum(low, inp_train, inp_train)  # apply cutoff
            if not 'log' in inputfile:
                inp_train = 20 * np.log10(inp_train)
                low = inp_train.min()
                thr = 20 * np.log10(thr)
            high = inp_train.max()
            plt.figure()
            posvals, bins, _ = plt.hist(inp_train[targets_train], 50,
                                        range=(low, high), alpha=.8)
            negvals, _, _ = plt.hist(inp_train[~targets_train], bins, alpha=.8)
            plt.axvline(thr, linestyle=':', color='r')
            plt.savefig(inputfile[:-3] + 'png')
            if options.plot_data:
                np.savez(inputfile[:-3] + 'hist.npz', bins=bins, thr=thr,
                         posvals=posvals, negvals=negvals)


if __name__=="__main__":
    main()
