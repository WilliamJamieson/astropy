"""Shared helpers for astropy.modeling ASV benchmarks."""

import numpy as np


def rng(seed=24680):
    return np.random.default_rng(seed)


def linspace(n, start=-5.0, stop=5.0):
    return np.linspace(start, stop, n, dtype=float)


def grid(shape, span=5.0):
    x = np.linspace(-span, span, shape, dtype=float)
    y = np.linspace(-span, span, shape, dtype=float)
    return np.meshgrid(x, y, indexing="xy")


def noisy(values, sigma=0.05, seed=24680):
    return values + rng(seed).normal(0.0, sigma, size=np.shape(values))
