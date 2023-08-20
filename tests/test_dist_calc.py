
import numpy as np
from distribution_diff import DistCalculator, jensen_shannon_divergence


def test_PSI():
    rng = np.random.default_rng(2023)
    d1 = rng.normal(size=1000)
    d2 = rng.normal(size=1000)
    dc = DistCalculator(d1)   
    assert np.abs(dc.PSI(d1)) < 1e-6
    assert np.abs(dc.PSI(d2)) < 1e-1
    d3 = rng.normal(size=1000) * 1.3 + 0.5
    assert dc.PSI(d3) > 1e-2


def test_JSD():
    rng = np.random.default_rng(2023)
    d1 = rng.normal(size=1000)
    d2 = rng.normal(size=1000)
    assert jensen_shannon_divergence(d1, d1) >= 0
    assert jensen_shannon_divergence(d1, d1) < 1e-6
    assert jensen_shannon_divergence(d1, d2) >= 0
    assert jensen_shannon_divergence(d1, d2) < 1e-1
    d3 = rng.normal(size=1000) * 1.3 + 0.5
    assert jensen_shannon_divergence(d1, d3) > 1e-2


def test_JSD_disjoint():
    rng = np.random.default_rng(2023)
    d1 = rng.uniform(size=1000, low=0.6, high=1.0)
    d2 = rng.uniform(size=1000, low=0.0, high=0.4)
    assert jensen_shannon_divergence(d1, d1) >= 0
    assert jensen_shannon_divergence(d1, d1) < 1e-6
    assert jensen_shannon_divergence(d1, d2) >= 0
    assert jensen_shannon_divergence(d1, d2) >= 1 - 1e-6
