"""Operator (UT-centred, elongated) search pattern added in milestone 8."""
from pathlib import Path

import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS
from cislunar_custody.custody import (predict_gap, strip_pattern, containment_horizon, strip_horizon,
                                      custody_horizons, search_radius_deg)
from cislunar_custody.timeutil import jd_from_iso

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"


def _P(sp_km, sv_ms):
    return np.diag([(sp_km / LU_KM) ** 2] * 3 + [(sv_ms * 1e-3 / VU_KMS) ** 2] * 3)


def test_strip_pattern_uses_at_most_n_fields():
    for n in (1, 10, 100):
        for ratio in (1.0, 0.3, 0.05, 1e-4):
            L, W = strip_pattern(1.0, n, ratio)
            assert L >= W > 0 and round(L * W) <= n
    assert strip_pattern(1.0, 100, 1.0) == (10.0, 10.0)
    assert strip_pattern(1.0, 100, 1e-4) == (100.0, 1.0)


def test_containment_horizon():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    assert containment_horizon(t, [1.0, 1.0, 1.0, 1.0]) == np.inf
    assert containment_horizon(t, [0.5, 1.0, 1.0, 1.0]) == 0.0
    assert 2.0 < containment_horizon(t, [1.0, 1.0, 0.995, 0.9]) < 4.0


def test_strip_option_leaves_other_outputs_unchanged():
    orb = load_named(CAT, "NRHO_9:2")
    dt = np.concatenate([[0.0], np.geomspace(0.05, 2.0, 8)])
    kw = dict(n_samples=200, jd0=jd_from_iso("2027-03-01T00:00"), mu=MU_EM)
    x0 = orb.states_at(0.3)[0]
    a = predict_gap(x0, _P(20.0, 0.1), dt, rng=np.random.default_rng(3), **kw)
    b = predict_gap(x0, _P(20.0, 0.1), dt, rng=np.random.default_rng(3), strips=[(1.0, 10)], **kw)
    for k in ("theta_ideal", "theta_ut_actual", "contain_lin", "contain_ut"):
        assert np.array_equal(a[k], b[k])
    assert np.all((b["contain_strip_N10"] >= 0) & (b["contain_strip_N10"] <= 1))
    # a strip the same area as the circle, centred and aligned, should not be much worse than the circle
    circle = custody_horizons(b, search_radius_deg(1.0, 10))["ut_actual"]
    assert strip_horizon(b, 10) >= 0.5 * circle
