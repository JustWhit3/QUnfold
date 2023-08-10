#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pylab as plt
import seaborn as sns

path = "../results/comparison/"


def plot_comparison(true, data, distr, binning, ext="png"):
    """
    Save comparison plot for different unfolding methods to local filesystem.

    Args:
        true (numpy.ndarray): true distribution histogram.
        data (dict): dictionary mapping each unfolding method to the corresponding result.
        distr (str): name of the distribution.
        binning (numpy.ndarray): output histogram binning.
        ext (str): output file extension (png, pdf)
    """
    sns.set()

    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")

    _, ax = plt.subplots(figsize=(9, 6))
    steps = np.append(true, [true[-1]])
    ax.step(binning, steps, label="True", where="post", color="black", linewidth=1)

    binwidth = binning[1] - binning[0]
    x_offset = binwidth / (len(data) + 5)
    for i, (method, result) in enumerate(data.items()):
        x = binning[:-1] + (i + 3) * x_offset
        y = result["unfolded"]
        yerr = result["error"]
        chi2 = round(compute_chi2_dof(y, true), 2)
        label = rf"{method} $\chi^2 = {chi2}$"
        ax.errorbar(x, y, yerr=yerr, label=label, marker="o", ms=3, linestyle="None")
    ax.legend()

    plt.savefig(f"{path}{distr}.{ext}")
    print(f"Info in <plt.savefig>: file {path}{distr}.{ext} has been created")


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
