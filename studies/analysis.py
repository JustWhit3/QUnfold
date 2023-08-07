#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ROOT
import numpy as np

from generator import *
from roounfold import *
from qunfold import *
from root_converter import *

import sys

sys.path.append("../src")
from QUnfold.QUnfoldQUBO import QUnfoldQUBO


# Load RooUnfold library from local installation
ROOT.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")

# Get ROOT random generator and set random seed
seed = 42
gRandom = ROOT.gRandom
gRandom.SetSeed(seed)

############################## CONFIG INPUT VARIABLES ############################
##################################################################################
distributions = {
    "normal": {
        "generator": gRandom.Gaus,
        "parameters": (4.9, 1.3),  # (mu, sigma)
    },
    "breit-wigner": {
        "generator": gRandom.BreitWigner,
        "parameters": (4.7, 1.7),  # (mu, gamma)
    },
    "double-peaked": {
        "generator": gRandom.Gaus,
        "parameters": ((3.5, 1.0), (6.2, 0.8)),  # ((mu1, sigma1), (mu2, sigma2))
    },
}
samples = 10000
bins = 30
min_bin = 0.0
max_bin = 10.0
bias = 0.8
smear = 0.5
eff = 0.9
##################################################################################
##################################################################################


def main():
    for distr in distributions:
        true, meas, response = generate_data(
            distr, samples, bins, min_bin, max_bin, bias, smear, eff
        )
        roounfold_plot_response(response, distr)

        ########################## Classical ##########################
        # Iterative Bayesian Unfolding (IBU)
        unfolded_IBU = roounfold_unfolder(response, meas, method="IBU")
        roounfold_plot_results(true, meas, unfolded_IBU, distr)

        # SVD Tikhonov unfolding (SVD)
        unfolded_SVD = roounfold_unfolder(response, meas, method="SVD")
        roounfold_plot_results(true, meas, unfolded_SVD, distr)

        # Bin-by-Bin unfolding (B2B)
        unfolded_B2B = roounfold_unfolder(response, meas, method="B2B")
        roounfold_plot_results(true, meas, unfolded_B2B, distr)

        ########################### Quantum ###########################
        resp = TMatrix_to_array(response.Mresponse(norm=True))
        meas = TH1_to_array(meas)
        true = TH1_to_array(true)

        # Simulated Annealing unfolding (SA)
        unfolder = QUnfoldQUBO(resp, meas, lam=0.04)
        unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100, seed=seed)

        binning = np.linspace(min_bin, max_bin, bins + 1)
        qunfold_plot_results(true, meas, unfolded_SA, distr, binning)


if __name__ == "__main__":
    main()
