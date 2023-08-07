#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pylab as plt

path = "../results/qunfold/"


def qunfold_plot_results(true, meas, unfolded, distr, binning, ext="png"):
    """
    Save quantum unfolding results plot to local filesystem.

    Args:
        true (numpy.ndarray): true distribution histogram.
        meas (numpy.ndarray): measured distribution histogram.
        unfolded (numpy.ndarray): unfolded distribution histogram.
        distr (str): name of the distribution.
        binning (numpy.ndarray): output histogram binning.
        ext (str): output file extension (png, pdf)
    """
    from analysis import overflow, num_left_bins, num_right_bins

    if not os.path.exists(f"{path}{distr}"):
        os.makedirs(f"{path}{distr}")

    _, ax = plt.subplots(figsize=(9, 6))
    for histo, label in [(true, "True"), (meas, "Meas")]:
        y = histo[num_left_bins:-num_right_bins] if overflow else histo
        steps = np.append(y, [y[-1]])
        ax.step(binning, steps, label=label, where="post")

    binwidth = binning[1] - binning[0]
    x = binning[:-1] + (binwidth / 2)
    y = unfolded[num_left_bins:-num_right_bins] if overflow else unfolded
    ax.scatter(x, y, label="Unfolded (SA)", marker="o", s=30, c="limegreen")
    ax.legend()
    plt.savefig(f"{path}{distr}/unfolded_SA.{ext}")
    print(
        f"Info in <plt.savefig>: file {path}{distr}/unfolded_SA.{ext} has been created"
    )
