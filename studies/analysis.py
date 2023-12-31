# Main modules
import sys
import ROOT as r
from analysis_functions.custom_logger import get_custom_logger
from analysis_functions.generator import generate
from analysis_functions.RooUnfold import (
    RooUnfold_unfolder,
    RooUnfold_plot,
    RooUnfold_plot_response,
)
from analysis_functions.QUnfolder import QUnfold_unfolder_and_plot
from analysis_functions.comparisons import plot_comparisons
from QUnfold.utility import TH1_to_array, TH2_to_array, normalize_response

# Settings
log = get_custom_logger(__name__)
loaded_RooUnfold = r.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    log.error("RooUnfold not found!")
    sys.exit(0)

# Input variables
distributions = ["normal", "gamma", "exponential", "breit-wigner", "double-peaked"]
samples = 10000
max_bin = 10
min_bin = 0
bins = 20
bias = -0.13
smearing = 0.21
eff = 0.92


if __name__ == "__main__":
    # Iterate over distributions
    for distr in distributions:
        # Generate data
        log.info("Unfolding the {} distribution".format(distr))
        truth, measured, response = generate(
            distr, bins, min_bin, max_bin, samples, bias, smearing, eff
        )

        ########################## Classic ###########################

        # RooUnfold settings
        r_response = response
        RooUnfold_plot_response(r_response, distr)
        r_response.UseOverflow(False)

        # Matrix inversion (MI)
        unfolded_MI, error_MI = RooUnfold_unfolder("MI", r_response, measured)
        RooUnfold_plot(truth, measured, unfolded_MI, distr)

        # Iterative Bayesian unfolding (IBU)
        unfolded_IBU, error_IBU = RooUnfold_unfolder("IBU", r_response, measured)
        RooUnfold_plot(truth, measured, unfolded_IBU, distr)

        # Tikhonov unfolding (SVD)
        unfolded_SVD, error_SVD = RooUnfold_unfolder("SVD", r_response, measured)
        RooUnfold_plot(truth, measured, unfolded_SVD, distr)

        ########################## Quantum ###########################

        # QUnfold settings
        truth = TH1_to_array(truth, overflow=False)
        measured = TH1_to_array(measured, overflow=False)
        response = normalize_response(
            TH2_to_array(response.Hresponse()), TH1_to_array(response.Htruth())
        )

        # Simulated annealing (SA)
        unfolded_SA, error_SA = QUnfold_unfolder_and_plot(
            "SA", response, measured, truth, distr, bins, min_bin, max_bin
        )

        # Hybrid solver (HYB)
        unfolded_HYB, error_HYB = QUnfold_unfolder_and_plot(
            "HYB", response, measured, truth, distr, bins, min_bin, max_bin
        )

        ########################## Compare ###########################

        # Comparison settings
        data = {
            "IBU4": TH1_to_array(unfolded_IBU, overflow=False),
            "MI": TH1_to_array(unfolded_MI, overflow=False),
            "SVD": TH1_to_array(unfolded_SVD, overflow=False),
            "SA": unfolded_SA,
            "HYB": unfolded_HYB,
        }
        errors = {
            "IBU4": error_IBU,
            "MI": error_MI,
            "SVD": error_SVD,
            "SA": error_SA,
            "HYB": error_HYB,
        }

        # Plot comparisons
        plot_comparisons(data, errors, distr, truth, bins, min_bin, max_bin)
        log.info("Done\n")
