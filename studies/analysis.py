#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from time import time

from generator import *
from qunfold import *

sys.path.append("../src")
from QUnfold.QUnfoldQUBO import QUnfoldQUBO


# Set random seed for numpy
seed = 9
np.random.seed(seed)

############################## CONFIG INPUT VARIABLES ############################
##################################################################################
distributions = {
    "normal": {
        "parameters": (5.3, 1.4),  # (mu, sigma)
        "lambda": 0.18,  # naivly optimized for 15 bins
    },
    "breit-wigner": {
        "parameters": (5.3, 2.1),  # (mu, gamma)
        "lambda": 0.039,  # naivly optimized for 15 bins
    },
    "exponential": {
        "parameters": (2.5,),  # (tau)
        "lambda": 0.0027,  # naivly optimized for 15 bins
    },
    "double-peaked": {
        "parameters": (3.3, 0.9, 6.4, 1.2),  # ((mu1, sigma1), (mu2, sigma2))
        "lambda": 0.022,  # naivly optimized for 15 bins
    },
}
samples = 10000
bins = 40
min_bin = 0.0
max_bin = 10.0
bias = -1.7
smear = 0.5
eff = 0.92
##################################################################################
##################################################################################


def main():
    for distr in distributions:
        true, meas, resp = generate_data(
            distr, samples, bins, min_bin, max_bin, bias, smear, eff
        )

        unfolder = QUnfoldQUBO(resp, meas, lam=0.1)
        t1 = time()
        unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100, seed=seed)
        t2 = time()
        unfolded_HYB = unfolder.solve_hybrid_sampler()  # don't need any hyper-parameter
        t3 = time()

        binning = np.linspace(min_bin, max_bin, bins + 1)
        qunfold_plot_results(true, meas, unfolded_SA, distr, binning, solver="SA")
        print(f"Elapsed time for SA solver = {round(t2 - t1, 2)} seconds")
        qunfold_plot_results(true, meas, unfolded_HYB, distr, binning, solver="HYB")
        print(f"Elapsed time for HYB solver = {round(t3 - t2, 2)} seconds")


if __name__ == "__main__":
    main()
