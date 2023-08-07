#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

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
    },
    "breit-wigner": {
        "parameters": (5.3, 2.1),  # (mu, gamma)
    },
    "exponential": {
        "parameters": (2.5,),  # (tau,)
    },
    "double-peaked": {
        "parameters": (3.3, 0.9, 6.4, 1.2),  # (mu1, sigma1, mu2, sigma2)
    },
}
samples = 10000
bins = 40
min_bin = 0.0
max_bin = 10.0
bias = 0.9
smear = 0.5
eff = 0.92

overflow = True
num_left_bins = 5
num_right_bins = 6
##################################################################################
##################################################################################


def main():
    for distr in distributions:
        true, meas, resp = generate_data(
            distr, samples, bins, min_bin, max_bin, bias, smear, eff
        )

        unfolder = QUnfoldQUBO(resp, meas, lam=0.04)
        unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100, seed=seed)

        binning = np.linspace(min_bin, max_bin, bins + 1)
        qunfold_plot_results(true, meas, unfolded_SA, distr, binning)


if __name__ == "__main__":
    main()
