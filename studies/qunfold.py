#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pylab as plt

path = "../results/qunfold/"


def qunfold_plot_results(true, meas, unfolded, distr, binning, solver, ext="png"):
    """
    Save quantum unfolding results plot to local filesystem.

    Args:
        true (numpy.ndarray): true distribution histogram.
        meas (numpy.ndarray): measured distribution histogram.
        unfolded (numpy.ndarray): unfolded distribution histogram.
        distr (str): name of the distribution.
        binning (numpy.ndarray): output histogram binning.
        solver (str): D-Wave Ocean solver (SA, HYB).
        ext (str): output file extension (png, pdf)
    """
    if not os.path.exists(f"{path}{distr}"):
        os.makedirs(f"{path}{distr}")

    _, ax = plt.subplots(figsize=(9, 6))
    for histo, label in [(true, "True"), (meas, "Meas")]:
        y = histo[1:-1]
        steps = np.append(y, [y[-1]])
        ax.step(binning, steps, label=label, where="post")

    binwidth = binning[2] - binning[1]
    x = binning[:-1] + (binwidth / 2)
    y = unfolded[1:-1]
    chi2 = round(compute_chi2_dof(y, true[1:-1]), 2)
    label = rf"Unfolded {solver} $\chi^2 = {chi2}$"
    ax.scatter(x, y, label=label, marker="o", s=30, c="limegreen")
    ax.legend()
    plt.savefig(f"{path}{distr}/unfolded_{solver}.{ext}")
    print(
        f"Info in <plt.savefig>: file {path}{distr}/unfolded_{solver}.{ext} has been created"
    )


def compute_chi2_dof(observed, expected):
    """
    Compute chi-squared per degree of freedom (chi2/dof) between two histograms.

    Args:
        observed (numpy.ndarray): observed histogram.
        expected (numpy.ndarray): expected histogram.

    Returns:
        float: chi-squared per degree of freedom.
    """
    obs = observed[expected != 0]
    exp = expected[expected != 0]  # avoid division by 0 error
    chi2 = np.sum((obs - exp) ** 2 / exp)
    dof = len(exp)
    return chi2 / dof
