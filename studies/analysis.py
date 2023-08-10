#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ROOT
import numpy as np

from generator import *
from roounfold import *
from qunfold import *
from comparison import *
from root_converter import *

import sys

sys.path.append("../src")
from QUnfold.QUnfoldQUBO import QUnfoldQUBO

# Load RooUnfold library from local installation
ROOT.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")

############################## CONFIG INPUT VARIABLES ############################
##################################################################################
distributions = {
    "normal": {
        "generator": ROOT.gRandom.Gaus,
        "parameters": (4.9, 1.3),  # (mu, sigma)
        "lambda": 0.18,  # naivly optimized for 15 bins
    },
    "breit-wigner": {
        "generator": ROOT.gRandom.BreitWigner,
        "parameters": (4.9, 1.5),  # (mu, gamma)
        "lambda": 0.039,  # naivly optimized for 15 bins
    },
    "double-peaked": {
        "generator": ROOT.gRandom.Gaus,
        "parameters": ((3.5, 1.0), (6.2, 0.8)),  # ((mu1, sigma1), (mu2, sigma2))
        "lambda": 0.022,  # naivly optimized for 15 bins
    },
    "exponential": {
        "generator": ROOT.gRandom.Exp,
        "parameters": (2.3,),  # (tau)
        "lambda": 0.0027,  # naivly optimized for 15 bins
    },
}
samples = 4000
bins = 15
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
        binning = np.linspace(min_bin, max_bin, bins + 1)

        # Simulated annealing unfolding
        unfolder = QUnfoldQUBO(resp, meas, lam=distributions[distr]["lambda"])
        unfolded_QUBO, error_QUBO = unfolder.solve_simulated_annealing(num_reads=20)
        qunfold_plot_results(true, meas, unfolded_QUBO, error_QUBO, distr, binning)
        ###############################################################

        data = {
            "IBU": {
                "unfolded": TH1_to_array(unfolded_IBU),
                "error": TH1_to_error(unfolded_IBU),
            },
            "SVD": {
                "unfolded": TH1_to_array(unfolded_SVD),
                "error": TH1_to_error(unfolded_SVD),
            },
            "B2B": {
                "unfolded": TH1_to_array(unfolded_B2B),
                "error": TH1_to_error(unfolded_B2B),
            },
            "QUBO": {"unfolded": unfolded_QUBO, "error": error_QUBO},
        }
        plot_comparison(true, data, distr, binning)


if __name__ == "__main__":
    main()
