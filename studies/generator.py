#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def _normal(mu, sigma, size):
    """
    Generate random numbers from a normal/gaussian distribution.

    Args:
        mu (float): center of the distribution.
        sigma (float): spread of the distribution.
        size (int): number of random samples to generate.

    Returns:
        numpy.ndarray: array of random numbers following the normal distribution.
    """
    return np.random.normal(mu, sigma, size)


def _breit_wigner(mu, gamma, size):
    """
    Generate random numbers from a Breit-Wigner distribution.

    Args:
        mu (float): center of the distribution.
        gamma (float): width at half-maximum, determining the spread of the distribution.
        size (int): number of random samples to generate.

    Returns:
        numpy.ndarray: array of random numbers following the Breit-Wigner distribution.
    """
    u = np.random.rand(size)
    return mu + (0.5 * gamma) * np.tan(np.pi * (u - 0.5))


def _exponential(tau, size):
    """
    Generate random numbers from a falling exponential distribution.

    Args:
        tau (float): inverse of the rate parameter.
        size (int): number of random samples to generate.

    Returns:
        numpy.ndarray: array of random numbers following the exponetial distribution.
    """
    return np.random.exponential(tau, size)


def _double_peaked(mu1, sigma1, mu2, sigma2, size):
    """
    Generate random numbers from a double-peaked distribution.

    Args:
        mu1 (float): center of the first normal distribution.
        sigma1 (float): spread of the first normal distribution.
        mu2 (float): center of the second normal distribution.
        sigma2 (float): spread of the second normal distribution.
        size (int): number of random samples to generate.

    Returns:
        numpy.ndarray: array of random numbers following the double-peaked distribution.
    """
    normal1 = np.random.normal(mu1, sigma1, size=size // 2)
    normal12 = np.random.normal(mu2, sigma2, size=size // 2)
    return np.concatenate([normal1, normal12])


def _smearing(arr, bias, smear, eff):
    """
    Apply gaussian smearing to the input sample array.

    Args:
        arr (numpy.ndarray): input sample array.
        bias (float): gaussian distortion mean.
        smear (float): gaussian distortion sigma.
        eff (float): smearing efficiency.

    Returns:
        numpy.ndarray: smeared output sample array.
        numpy.ndarray: masking array (drop values due to limited efficiency)
    """
    mask = np.random.rand(len(arr)) < eff
    out = arr[mask]  # drop values due to limited efficiency
    out += np.random.normal(bias, smear, size=len(out))  # apply gaussian smearing
    return out, mask


def generate_data(distr, num_samples, num_bins, min_bin, max_bin, bias, smear, eff):
    """
    Generate true/measured histograms and response matrix for a given distribution.

    Args:
        distr (str): name of the distribution.
        num_samples (int): number of data samples.
        num_bins (int): number of bins in the histograms.
        min_bin (float): minimum value of the histogram range.
        max_bin (float): maximum value of the histogram range.
        bias (float): distortion bias.
        smear (float): distortion variance.
        eff (float): smearing efficiency.

    Returns:
        numpy.ndarray: true distribution array.
        numpy.ndarray: measured distribution array.
        numpy.ndarray: response matrix.
    """
    from analysis import distributions

    generators = {
        "normal": _normal,
        "breit-wigner": _breit_wigner,
        "exponential": _exponential,
        "double-peaked": _double_peaked,
    }

    # Get generator and parameters of the distribution
    generator = generators[distr]
    parameters = distributions[distr]["parameters"]

    # Set histograms binning
    bins = np.linspace(min_bin, max_bin, num_bins + 1)

    # Generate true and meas histograms
    true_data = generator(*parameters, size=num_samples)
    meas_data, _ = _smearing(true_data, bias=bias, smear=smear, eff=eff)
    true, _ = np.histogram(true_data, bins=bins)
    meas, _ = np.histogram(meas_data, bins=bins)
    true = true.astype(float)
    meas = meas.astype(float)

    # Generate response matrix
    mc_data = generator(*parameters, size=num_samples * 100)
    reco_data, mask = _smearing(mc_data, bias=bias, smear=smear, eff=eff)
    response, _, _ = np.histogram2d(reco_data, mc_data[mask], bins=bins)

    # Normalize response columns using Monte Carlo data
    colnorms, _ = np.histogram(mc_data, bins=bins)
    response /= colnorms + 1e-6  # add epsilon to avoid division by 0 error

    return true, meas, response
