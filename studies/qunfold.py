#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pylab as plt

sys.path.append("../src")
from QUnfold.QUnfoldQUBO import QUnfoldQUBO

path = "../results/qunfold/"


def qunfold_unfolder(response, meas):
    """
    Apply QUnfold algorithm to unfold the given measured distribution.

    Args:
        response (numpy.ndarray): response matrix.
        meas (numpy.ndarray): measured histogram.

    Returns:
        numpy.ndarray: unfolded histogram.
    """
    unfolder = QUnfoldQUBO(response, meas)
    result = unfolder.solve_simulated_annealing(lam=0.1, num_reads=100)
    return result


def qunfold_plot_results(
    true, meas, unfolded, distr, binning, overflow=False, ext="png"
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
    ax.scatter(x, y, label="Unfolded (SA)", marker="o", s=30, c="limegreen")
    ax.legend()
    plt.savefig(f"{path}{distr}/unfolded_SA_with_overflow.{ext}")
