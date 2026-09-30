"""Operator (UT-centred, elongated) search pattern added in milestone 8."""
from pathlib import Path

import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS
from cislunar_custody.custody import (predict_gap, strip_pattern, containment_horizon, strip_horizon,
                                      custody_horizons, search_radius_deg, strip_coordinates,
                                      strip_containment, MAX_STRIP_DEG)
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
    assert strip_pattern(1.0, 10, 1e-4) == (10.0, 1.0)
    # no strip longer than MAX_STRIP_DEG, whatever the ellipse
    for n in (1, 10, 100):
        assert strip_pattern(1.0, n, 1e-6)[0] <= MAX_STRIP_DEG
    assert strip_pattern(1.0, 100, 1e-4) == (10.0, 10.0)
    assert strip_pattern(1.0, 100, 1e-4, max_len_deg=None) == (100.0, 1.0)


def test_strip_coordinates_match_gnomonic_for_small_angles():
    rng = np.random.default_rng(0)
    u = np.array([0.3, -0.5, 0.8]); u /= np.linalg.norm(u)
    e1 = np.cross(u, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    xy = np.radians(0.5) * rng.uniform(-1, 1, (500, 2))
    v = u + xy[:, :1] * e1 + xy[:, 1:] * e2
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    along, cross = strip_coordinates(v, u, e1)
    q = (v @ np.stack([e1, e2]).T) / (v @ u)[:, None]
    assert np.allclose(along, q[:, 0], atol=1e-4) and np.allclose(cross, q[:, 1], atol=1e-4)


def test_strip_containment_tilt_is_conservative():
    rng = np.random.default_rng(1)
    along = np.radians(rng.uniform(-4.5, 4.5, 2000))
    cross = np.radians(rng.normal(0, 0.05, 2000))
    straight = strip_containment(along, cross, 10.0, 1.0, tilt_deg=0.0)
    tilted = strip_containment(along, cross, 10.0, 1.0, tilt_deg=2.0)
    assert straight == 1.0 and tilted <= straight
    assert strip_containment(along, cross, 10.0, 1.0, tilt_deg=10.0) < 0.9


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
