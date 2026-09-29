from pathlib import Path

import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS
from cislunar_custody.custody import predict_gap, crossing_time, custody_horizons, search_radius_deg
from cislunar_custody.dynamics import propagate_many, propagate_many_dense
from cislunar_custody.timeutil import jd_from_iso

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"


def _P(sp_km, sv_ms):
    return np.diag([(sp_km / LU_KM) ** 2] * 3 + [(sv_ms * 1e-3 / VU_KMS) ** 2] * 3)


def test_dense_propagation_matches_final_state():
    orb = load_named(CAT, "NRHO_9:2")
    S0 = orb.states_at(np.array([0.2, 0.6]))
    dense = propagate_many_dense(S0, np.array([0.0, 0.3, 0.7]), MU_EM)
    assert np.allclose(dense[0], S0)
    assert np.allclose(dense[-1], propagate_many(S0, 0.7, MU_EM), atol=1e-9)


def test_crossing_time():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    y = np.array([1.0, 2.0, 4.0, 16.0])
    assert crossing_time(t, y, 8.0) == pytest.approx(np.exp(np.log(2) + 0.5 * np.log(2)))
    assert crossing_time(t, y, 100.0) == np.inf and crossing_time(t, y, 0.5) == 0.0


def test_small_uncertainty_gaussian_predictions_match_monte_carlo():
    orb = load_named(CAT, "NRHO_9:2")
    x0 = orb.states_at(0.5 * orb.period)[0]                 # apolune
    pred = predict_gap(x0, _P(1.0, 1e-3), np.array([0.0, 0.1, 0.3]), jd_from_iso("2027-03-01T00:00"),
                       MU_EM, n_samples=2000, rng=np.random.default_rng(0))
    for tag in ("lin", "ut"):
        assert np.allclose(pred[f"theta_{tag}_claim"][1:], pred["theta_ideal"][1:], rtol=0.15)
        assert np.all(pred[f"contain_{tag}"][1:] > 0.97)
    assert pred["smax"][0] == pytest.approx(1.0) and np.all(np.diff(pred["smax"]) > 0)


def test_custody_horizons_are_ordered_by_search_radius():
    orb = load_named(CAT, "NRHO_9:2")
    dt = np.concatenate([[0.0], np.geomspace(0.05, 3.0, 12)])
    pred = predict_gap(orb.states_at(0.3)[0], _P(20.0, 0.1), dt, jd_from_iso("2027-03-01T00:00"),
                       MU_EM, n_samples=300, rng=np.random.default_rng(1))
    small = custody_horizons(pred, search_radius_deg(1.0, 1))["ideal"]
    large = custody_horizons(pred, search_radius_deg(1.0, 100))["ideal"]
    assert small <= large
