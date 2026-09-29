from pathlib import Path

import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS
from cislunar_custody.dynamics import propagate, propagate_many
from cislunar_custody.filters import AnglesModel, EKF, UKF, to_physical_sigma
from cislunar_custody.scenario import build_measurements, run_filter
from cislunar_custody.timeutil import jd_from_iso

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"
JD0 = jd_from_iso("2027-01-01T00:00")


def _P(sig_pos_km, sig_vel_ms):
    return np.diag([(sig_pos_km / LU_KM) ** 2] * 3 + [(sig_vel_ms * 1e-3 / VU_KMS) ** 2] * 3)


def test_propagate_many_matches_single():
    orb = load_named(CAT, "NRHO_9:2")
    S0 = orb.states_at(np.array([0.0, 0.3, 0.9]))
    many = propagate_many(S0, 0.5, MU_EM, rtol=1e-12, atol=1e-12)
    for s0, sm in zip(S0, many):
        assert np.allclose(propagate(s0, 0.5, MU_EM).y[:, -1], sm, atol=1e-10)


def test_measurement_jacobian_matches_finite_difference():
    orb = load_named(CAT, "NRHO_9:2")
    meas, _, _ = build_measurements(orb, JD0, 20.0, 1.0, rng=np.random.default_rng(0))
    m = meas[len(meas) // 2]
    model = AnglesModel(MU_EM)
    x = orb.states_at(m.t)[0]
    H = model.H(x, m)
    h = 1e-7
    cols = [model.residual(model.h(x + h * e, m)[0], model.h(x - h * e, m)[0]) / (2 * h) for e in np.eye(6)]
    assert np.allclose(H, np.array(cols).T, rtol=1e-5, atol=1e-9)


def test_ukf_prediction_matches_linear_covariance_for_small_uncertainty():
    """In the small-uncertainty limit the unscented and linearised (STM) predictions agree.
    (With 1 m/s over 1 TU the cloud grows to ~1000 km and they differ by ~0.2%: that is
    genuine NRHO nonlinearity, not an error.)"""
    orb = load_named(CAT, "NRHO_9:2")
    x0 = orb.states_at(0.2)[0]
    P0 = _P(1.0, 1e-3)                      # 1 km, 1 mm/s
    model = AnglesModel(MU_EM)
    ekf = EKF(MU_EM, model, rtol=1e-12, atol=1e-14)
    ukf = UKF(MU_EM, model, rtol=1e-12, atol=1e-14)
    _, Pe = ekf.predict(x0, P0, 0.3)
    _, Pu = ukf.predict(x0, P0, 0.3)
    assert np.max(np.abs(Pu - Pe)) / np.max(np.abs(Pe)) < 1e-3


@pytest.mark.parametrize("Filter", [EKF, UKF])
def test_filter_converges_on_nrho_arc(Filter):
    orb = load_named(CAT, "NRHO_9:2")
    rng = np.random.default_rng(1)
    meas, t0, t1 = build_measurements(orb, JD0, 20.0, 3.0, rng=rng)
    assert len(meas) > 100
    P0 = _P(100.0, 1.0)
    x0 = orb.states_at(t0)[0] + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    out = run_filter(Filter(MU_EM, AnglesModel(MU_EM)), x0, P0, t0, meas, orb)
    assert not out["diverged"]
    sp, _ = to_physical_sigma(out["P"][-1])
    err_km = np.linalg.norm(out["err"][-1, :3]) * LU_KM
    assert sp < 50.0 and err_km < 3.0 * sp
