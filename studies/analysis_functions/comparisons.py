import os
import matplotlib.pyplot as plt
import numpy as np
from QUnfold.utility import compute_chi2


def plot_errorbar(
    ax1, ax2, bin_edges, bin_contents, bin_errors, color, mark, method, chi2, truth
):
    # Plot stuff
    ax1.errorbar(
        x=bin_edges,
        y=bin_contents,
        yerr=bin_errors,
        color=color,
        marker=mark,
        ms=3.5,
        label=r"{} ($\chi^2 = {:.2f}$)".format(method, chi2),
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_edges,
        y=bin_contents / truth,
        yerr=bin_errors / truth,
        color=color,
        marker=mark,
        ms=3.5,
        linestyle="None",
    )


def plot_comparisons(data, errors, cov, distr, truth, measured, binning):
    # Binning
    bin_edges = binning
    binwidths = np.diff(bin_edges)
    bin_midpoints = bin_edges[:-1] + binwidths / 2

    # Divide into subplots
    fig = plt.figure(figsize=(8.6, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot truth
    truth_steps = np.append(truth, [truth[-1]])
    ax1.step(
        bin_edges,
        truth_steps,
        label="Truth",
        where="post",
        color="tab:blue",
    )
    ax1.fill_between(bin_edges, truth_steps, step="post", alpha=0.3, color="tab:blue")
    ax2.axhline(y=1, color="tab:blue")

    # Plot measured
    meas_steps = np.append(measured, [measured[-1]])
    ax1.step(
        bin_edges,
        meas_steps,
        label="Measured",
        where="post",
        color="tab:orange",
    )
    ax1.fill_between(bin_edges, meas_steps, step="post", alpha=0.3, color="tab:orange")

    # Iterate over the unfolding methods
    for method, unfolded in data.items():
        # Plot each unfolding method
        chi2_dof = round(compute_chi2(unfolded, truth, cov[method]), 4)
        if method == "MI":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "green",
                "s",
                r"$\mathtt{RooUnfold}$ (MI)",
                chi2_dof,
                truth,
            )
        elif method == "IBU4":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "red",
                "o",
                r"$\mathtt{RooUnfold}$ (IBU)",
                chi2_dof,
                truth,
            )
        elif method == "HYB":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "orange",
                "*",
                r"$\mathtt{QUnfold}$ (HYB)",
                chi2_dof,
                truth,
            )
        elif method == "SA":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "purple",
                "*",
                r"$\mathtt{QUnfold}$ (SIM)",
                chi2_dof,
                truth,
            )

        # Plot settings
        ax1.tick_params(axis="x", which="both", bottom=True, top=False, direction="in")
        ax2.tick_params(axis="x", which="both", bottom=True, top=True, direction="in")
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax1.set_ylim(0, ax1.get_ylim()[1])
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        ax2.set_yticklabels(["", "0.5", "", "1.0", "", "1.5", ""])
        ax1.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax2.set_ylabel("Ratio to\ntruth")
        ax2.set_xlabel("Bins", loc="center")
        ax1.set_ylabel("Entries", loc="center")
        ax1.legend(loc="best")

        # Save plot
        plt.tight_layout()
        if not os.path.exists("studies/img/analysis"):
            os.makedirs("studies/img/analysis/png")
            os.makedirs("studies/img/analysis/pdf")
        plt.savefig("studies/img/analysis/png/{}.png".format(distr))
        plt.savefig("studies/img/analysis/pdf/{}.pdf".format(distr))

    plt.close()
