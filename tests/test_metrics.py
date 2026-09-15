import numpy as np
import pytest

from video2laeq.metrics import Sample, compute, laeq, validate


def test_laeq_constant():
    assert laeq(np.array([60.0] * 10)) == pytest.approx(60.0)


def test_laeq_energy_average():
    # 50 και 70 dB μισό-μισό: 10·log10((10^5 + 10^7)/2) ≈ 67.04
    assert laeq(np.array([50.0, 70.0])) == pytest.approx(67.04, abs=0.01)


def test_percentiles():
    samples = [Sample(float(i), float(v)) for i, v in enumerate(range(40, 61))]  # 40..60
    r = compute(samples)
    assert r.laf10 == pytest.approx(58.0)
    assert r.laf90 == pytest.approx(42.0)
    assert r.lmin == 40.0 and r.lmax == 60.0
    assert r.n_valid == r.n_total == 21


def test_validate_rejects_outliers_and_none():
    samples = [Sample(0, 55.0), Sample(1, None), Sample(2, 5.5), Sample(3, 56.0), Sample(4, 95.0), Sample(5, 55.5)]
    validate(samples)
    assert [s.valid for s in samples] == [True, False, False, True, False, True]
    assert "εκτός" in samples[2].reason and "άλμα" in samples[4].reason


def test_compute_needs_valid():
    with pytest.raises(ValueError):
        compute([Sample(0, None, False)])
