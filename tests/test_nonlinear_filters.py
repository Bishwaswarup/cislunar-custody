from pathlib import Path

import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS
from cislunar_custody.filters import AnglesModel, UKF, GMUKF, ParticleFilter, to_physical_sigma
from cislunar_custody.scenario import build_measurements, run_filter
from cislunar_custody.timeutil import jd_from_iso

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"
JD0 = jd_from_iso("2027-01-01T00:00")


def _P(sp_km, sv_ms):
    return np.diag([(sp_km / LU_KM) ** 2] * 3 + [(sv_ms * 1e-3 / VU_KMS) ** 2] * 3)


def test_split_preserves_mixture_moments():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    P = A @ A.T + 6 * np.eye(6)
    m = rng.standard_normal(6)
    W, M, Ps = GMUKF.split(0.7, m, P, rng.standard_normal((6, 6)))
    gm = GMUKF(MU_EM, AnglesModel(MU_EM))
    mm, PP = gm.moments((np.array(W) / 0.7, np.array(M), np.array(Ps)))
    assert np.allclose(sum(W), 0.7)
    assert np.allclose(mm, m) and np.allclose(PP, P)


def test_gmukf_without_splitting_equals_ukf():
    orb = load_named(CAT, "NRHO_9:2")
    meas, t0, _ = build_measurements(orb, JD0, 20.0, 0.5, rng=np.random.default_rng(2))
    model = AnglesModel(MU_EM)
    P0 = _P(50.0, 0.5)
    x0 = orb.states_at(t0)[0]
    a = run_filter(UKF(MU_EM, model), x0, P0, t0, meas, orb)
    b = run_filter(GMUKF(MU_EM, model, split_tol=np.inf), x0, P0, t0, meas, orb)
    assert np.allclose(a["x"][-1], b["x"][-1], atol=1e-12) and np.allclose(a["P"][-1], b["P"][-1], rtol=1e-8)
    assert np.all(b["n_comp"] == 1)


def test_gmukf_splits_across_a_long_gap():
    orb = load_named(CAT, "NRHO_9:2")
    gm = GMUKF(MU_EM, AnglesModel(MU_EM))
    t_apo = 0.5 * orb.period                      # start at apolune: a 1000 km cloud at
    b = gm.predict_belief(gm.init(orb.states_at(t_apo)[0], _P(1000.0, 10.0)), 0.6)   # perilune would hit the Moon
    assert gm.size(b) > 1
    assert np.isclose(b[0].sum(), 1.0)


def test_particle_prediction_matches_ukf_moments():
    orb = load_named(CAT, "NRHO_9:2")
    x0, P0 = orb.states_at(0.3)[0], _P(1.0, 1e-3)
    model = AnglesModel(MU_EM)
    pf = ParticleFilter(MU_EM, model, n=4000, rng=np.random.default_rng(3), handoff_km=None)
    mp, Pp = pf.moments(pf.predict_belief(pf.init(x0, P0), 0.3))
    mu_, Pu = UKF(MU_EM, model).predict(x0, P0, 0.3)
    s = np.sqrt(np.diag(Pu))
    assert np.all(np.abs(mp - mu_) < 0.1 * s)                     # sampling error ~ 1/sqrt(N)
    assert np.allclose(np.sqrt(np.diag(Pp)), s, rtol=0.1)


def test_particle_filter_converges_on_short_arc():
    orb = load_named(CAT, "NRHO_9:2")
    rng = np.random.default_rng(4)
    meas, t0, _ = build_measurements(orb, JD0, 20.0, 1.0, rng=rng)
    P0 = _P(100.0, 1.0)
    x0 = orb.states_at(t0)[0] + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    pf = ParticleFilter(MU_EM, AnglesModel(MU_EM), n=3000, rng=rng)
    out = run_filter(pf, x0, P0, t0, meas, orb)
    sp, _ = to_physical_sigma(out["P"][-1])
    err = np.linalg.norm(out["err"][-1, :3]) * LU_KM
    assert not out["diverged"] and sp < 100.0 and err < 4.0 * sp


def test_particle_filter_hands_over_to_ukf_on_long_arc():
    orb = load_named(CAT, "NRHO_9:2")
    rng = np.random.default_rng(8)
    meas, t0, _ = build_measurements(orb, JD0, 202.0, 3.0, rng=rng)
    P0 = _P(1000.0, 10.0)
    x0 = orb.states_at(t0)[0] + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    out = run_filter(ParticleFilter(MU_EM, AnglesModel(MU_EM), n=2000, rng=rng), x0, P0, t0, meas, orb)
    assert out["n_comp"][0] == 2000 and out["n_comp"][-1] == 1        # started as PF, ended as UKF
    sp, _ = to_physical_sigma(out["P"][-1])
    assert np.linalg.norm(out["err"][-1, :3]) * LU_KM < 4.0 * sp
