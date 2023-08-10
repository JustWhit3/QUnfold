#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler


class QUnfoldQUBO:
    def __init__(self, response, meas, lam=0.0):
        self.R = response
        self.d = meas
        self.lam = lam

    @staticmethod
    def _get_laplacian(dim):
        diag = np.ones(dim) * -2
        ones = np.ones(dim - 1)
        D = np.diag(diag) + np.diag(ones, k=1) + np.diag(ones, k=-1)
        return D

    def _define_variables(self):
        # Get largest power of 2 integer below the total number of entries
        n = int(2 ** np.floor(np.log2(sum(self.d)))) - 1
        # Encode integer variables using logarithmic binary encoding
        vars = [LogEncInteger(f"x{i}", value_range=(0, n)) for i in range(len(self.d))]
        return vars

    def _define_hamiltonian(self, x):
        hamiltonian = 0
        dim = len(x)
        # Add linear terms
        a = -2 * (self.R.T @ self.d)
        for i in range(dim):
            hamiltonian += a[i] * x[i]
        # Add quadratic terms
        G = self._get_laplacian(dim)
        B = (self.R.T @ self.R) + self.lam * (G.T @ G)
        for i in range(dim):
            for j in range(dim):
                hamiltonian += B[i, j] * x[i] * x[j]
        return hamiltonian

    def _define_pyqubo_model(self):
        x = self._define_variables()
        h = self._define_hamiltonian(x)
        labels = [x[i].label for i in range(len(x))]
        model = h.compile()
        return labels, model

    def solve_simulated_annealing(self, num_reads=10, seed=None):
        labels, model = self._define_pyqubo_model()
        sampler = SimulatedAnnealingSampler()
        samples = model.decode_sampleset(
            sampler.sample(model.to_bqm(), num_reads=num_reads, seed=seed)
        )
        solutions = np.array(
            [[sample.subh[label] for label in labels] for sample in samples]
        )
        solution = np.mean(solutions, axis=0)
        error = np.sqrt(np.var(solutions, axis=0) / num_reads)
        return solution, error
