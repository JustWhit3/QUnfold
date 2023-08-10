#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pylab as plt
import seaborn as sns

path = "../results/qunfold/"


def qunfold_plot_results(
    true, meas, unfolded, error, distr, binning, overflow=False, ext="png"
):
    """
    Save quantum unfolding results plot to local filesystem.

    Args:
        true (numpy.ndarray): true distribution histogram.
        meas (numpy.ndarray): measured distribution histogram.
        unfolded (numpy.ndarray): unfolded distribution histogram.
        distr (str): name of the distribution.
        binning (numpy.ndarray): output histogram binning.
        overflow (bool): enable/disable first and last bins overflow.
        ext (str): output file extension (png, pdf)
    """
    sns.set()

    if not os.path.exists(f"{path}{distr}"):
        os.makedirs(f"{path}{distr}")

    _, ax = plt.subplots(figsize=(9, 6))
    for histo, label in [(true, "True"), (meas, "Meas")]:
        y = histo[1:-1] if overflow else histo
        steps = np.append(y, [y[-1]])
        ax.step(binning, steps, label=label, where="post")

    binwidth = binning[1] - binning[0]
    x = binning[:-1] + (binwidth / 2)
    y = unfolded[1:-1] if overflow else unfolded
    yerr = error[1:-1] if overflow else error
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        label="Unfolded QUBO",
        marker="o",
        ms=5,
        color="black",
        linestyle="None",
    )
    ax.legend()
    plt.savefig(f"{path}{distr}/unfolded_QUBO.{ext}")
    print(
        f"Info in <plt.savefig>: file {path}{distr}/unfolded_QUBO.{ext} has been created"
    )
