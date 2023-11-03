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
        "parameters": (5.3, 1.4)  # (mu, sigma)
    },
    "breit-wigner": {
        "parameters": (5.3, 2.1)  # (mu, gamma)
    },
    "exponential": {
        "parameters": (2.5,)  # (tau)
    },
    "double-peaked": {
        "parameters": (3.3, 0.9, 6.4, 1.2)  # ((mu1, sigma1), (mu2, sigma2))
    },
}
samples = 10000
bins = 20
min_bin = 0.0
max_bin = 10.0
bias = -0.13
smear = 0.21
eff = 0.92
##################################################################################
##################################################################################


if __name__ == "__main__":

    for distr in distributions:
        true, meas, resp = generate_data(
            distr, samples, bins, min_bin, max_bin, bias, smear, eff
        )

        unfolder = QUnfoldQUBO(resp, meas, lam=0.05)
        energy_true = unfolder.compute_energy(x=true)

        t1 = time()
        unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100, seed=seed)
        t2 = time()
        unfolded_HYB = unfolder.solve_hybrid_sampler()  # don't need any hyper-parameter
        t3 = time()
        energy_SA = unfolder.compute_energy(x=unfolded_SA)
        energy_HYB = unfolder.compute_energy(x=unfolded_HYB)

        print("Energy true =", energy_true)
        print("Energy SA =", energy_SA)
        print("Energy HYB =", energy_HYB)

        binning = np.linspace(min_bin, max_bin, bins + 1)
        qunfold_plot_results(true, meas, unfolded_SA, distr, binning, solver="SA")
        print(f"Elapsed time for SA solver = {round(t2 - t1, 2)} seconds")
        qunfold_plot_results(true, meas, unfolded_HYB, distr, binning, solver="HYB")
        print(f"Elapsed time for HYB solver = {round(t3 - t2, 2)} seconds")
        print()
