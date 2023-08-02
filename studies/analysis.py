#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------- Metadata ----------------------
#
# File name:  analysis.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-08-02
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Input variables
distributions = ["breit-wigner", "normal", "double-peaked", "exponential"]
samples = 10000
max_bin = 10
min_bin = -10
bins = 40

# STD modules
import sys

# Data science modules
import ROOT as r

# My modules
from functions.ROOT_converter import TH1_to_array, TH2_to_array
from functions.custom_logger import INFO, ERROR
from functions.generator import generate
from functions.RooUnfold import (
    RooUnfold_unfolder,
    RooUnfold_plot,
    RooUnfold_plot_response,
)
from functions.QUnfolder import QUnfold_unfolder_and_plot
from functions.comparisons import plot_comparisons

# RooUnfold settings
loaded_RooUnfold = r.gSystem.Load("../HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    ERROR("RooUnfold not found!")
    sys.exit(0)


def main():

    # Global variables
    global min_bin

    # Iterate over distributions
    for distr in distributions:

        # Exponential distribution settings
        if distr == "exponential":
            min_bin = 0

        # Generate data
        INFO("Unfolding the {} distribution".format(distr))
        truth, meas, response = generate(distr, bins, min_bin, max_bin, samples)

        ########################## Classic ###########################

        # RooUnfold settings
        r_response = response
        RooUnfold_plot_response(r_response, distr)
        r_response.UseOverflow(False)

        # Matrix inversion (MI)
        unfolded_MI = RooUnfold_unfolder("MI", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_MI, distr)

        # Iterative Bayesian unfolding (IBU)
        unfolded_IBU = RooUnfold_unfolder("IBU", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_IBU, distr)

        # Tikhonov unfolding (SVD)
        unfolded_SVD = RooUnfold_unfolder("SVD", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_SVD, distr)

        # Bin-to-Bin unfolding (B2B)
        unfolded_B2B = RooUnfold_unfolder("B2B", r_response, meas)
        RooUnfold_plot(truth, meas, unfolded_B2B, distr)

        ########################## Quantum ###########################

        # QUnfold settings
        truth = TH1_to_array(truth, overflow=True)
        meas = TH1_to_array(meas, overflow=True)
        response = TH2_to_array(response.Hresponse(), overflow=True)

        # Simulated annealing (SA)
        unfolded_SA = QUnfold_unfolder_and_plot(
            "SA", response, meas, truth, distr, bins, min_bin, max_bin
        )

        ########################## Compare ###########################

        # Comparison settings
        data = {
            "IBU4": TH1_to_array(unfolded_IBU, overflow=False),
            "B2B": TH1_to_array(unfolded_B2B, overflow=False),
            "MI": TH1_to_array(unfolded_MI, overflow=False),
            "SVD": TH1_to_array(unfolded_SVD, overflow=False),
            "SA": unfolded_SA[1:-1],
        }

        # Plot comparisons
        plot_comparisons(data, distr, truth[1:-1], bins, min_bin, max_bin)
        print("Done", end="\n\n")


if __name__ == "__main__":
    main()