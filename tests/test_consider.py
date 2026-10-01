"""Milestone 9: consider covariance (site biases, timing offsets, SRP area-to-mass error)."""
from pathlib import Path

import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import LU_KM, MU_EM, TU_S, DAY_S, VU_KMS
from cislunar_custody.custody import predict_gap
from cislunar_custody.dynamics import propagate_many_dense
from cislunar_custody.frames import Ephemeris
from cislunar_custody.observability import arc_information
from cislunar_custody.observability.consider import SRPModel, arc_consider, propagate_srp_sensitivity
from cislunar_custody.sensors import SITES, radec
from cislunar_custody.timeutil import jd_from_iso

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"
JD0 = jd_from_iso("2027-02-01T00:00")


def _arc(span_tu=0.7, n=40):
    orb = load_named(CAT, "NRHO_9:2")
    t = np.linspace(0.0, span_tu, n)
    keys = ["Hanle", "Haleakala"]
    T, R, B, sid, rate = [], [], [], [], []
    h = 60.0 / TU_S
    for i, key in enumerate(keys):
        tt = t[i::2]
        eph = Ephemeris(JD0 + tt * TU_S / DAY_S)
        rs = eph.site_eci(SITES[key])[0]
        T.append(tt), R.append(rs), B.append(eph.basis), sid.append(np.full(len(tt), i))
        ang = []
        for d in (-h, h):
            e2 = Ephemeris(JD0 + (tt + d) * TU_S / DAY_S)
            ra, dec = radec(e2.to_eci(orb.states_at(tt + d), MU_EM) - e2.site_eci(SITES[key])[0])
            ang.append(np.stack([np.unwrap(ra), dec], 1))
        rate.append((ang[1] - ang[0]) / (2 * h * TU_S))
    basis = tuple(np.concatenate([b[j] for b in B]) for j in range(3))
    return orb, np.concatenate(T), np.concatenate(R), basis, np.concatenate(sid), np.concatenate(rate), span_tu


def test_zero_consider_reproduces_crlb():
    orb, t, rs, bm, sid, rate, te = _arc()
    be = tuple(b[0] for b in Ephemeris(np.array([JD0 + te * TU_S / DAY_S])).basis)
    ref = arc_information(orb, 0.0, t, rs, bm, [te], [be])[0]
    srp = SRPModel(JD0, -1.0, 2.0)
    ca = arc_consider(orb, 0.0, t, rs, bm, te, site_id=sid, rate=rate, srp=srp)
    P, aug = ca.covariance()
    assert ca.observable == ref.observable
    assert np.allclose(P, ref.P_end, rtol=1e-10, atol=0)
    assert np.allclose(aug[:6, :6], ref.P_end, rtol=1e-10, atol=0) and aug[6, 6] == 0.0


def test_consider_terms_scale_with_variance_and_are_psd():
    orb, t, rs, bm, sid, rate, te = _arc()
    ca = arc_consider(orb, 0.0, t, rs, bm, te, site_id=sid, rate=rate, srp=SRPModel(JD0, -1.0, 2.0))
    P1, _ = ca.covariance(bias_arcsec=0.2)
    P2, _ = ca.covariance(bias_arcsec=0.4)
    assert np.allclose(P2 - ca.P_crlb_end, 4.0 * (P1 - ca.P_crlb_end), rtol=1e-8, atol=1e-30)
    assert ca.sigma_pos_km(P1) > ca.sigma_pos_crlb_km
    _, aug = ca.covariance(bias_arcsec=0.5, timing_s=0.01, srp_sigma=0.003)
    assert np.allclose(aug, aug.T) and np.linalg.eigvalsh(aug).min() > -1e-12 * np.abs(aug).max()
    assert aug[6, 6] == pytest.approx(0.003 ** 2)


def test_srp_acceleration_magnitude():
    m = SRPModel(JD0, 0.0, 1.0)
    a_ms2 = np.linalg.norm(m(0.5)) * LU_KM / TU_S ** 2 * 1e3          # per unit A/m
    assert a_ms2 * 0.01 == pytest.approx(1.3 * 4.56e-6 * 0.01, rel=0.04)


def test_srp_sensitivity_matches_finite_difference():
    orb = load_named(CAT, "NRHO_9:2")
    x0 = orb.states_at(0.3)[0]
    grid = np.array([0.0, 0.4, 1.0])
    srp = SRPModel(JD0, 0.0, 1.5)
    psi = propagate_srp_sensitivity(x0, grid, MU_EM, srp)
    eps = 1e-3
    Y = propagate_many_dense(np.vstack([x0, x0]), grid, MU_EM, 1e-12, 1e-14, p=np.array([-eps, eps]), accel=srp)
    fd = (Y[:, 1] - Y[:, 0]) / (2 * eps)
    assert np.allclose(fd[1:], psi[1:], rtol=1e-5, atol=1e-5 * np.abs(psi).max())


def test_predict_gap_consider_with_same_covariance_is_unchanged():
    orb = load_named(CAT, "NRHO_9:2")
    x0 = orb.states_at(1.0)[0]
    P0 = np.diag([(5 / LU_KM) ** 2] * 3 + [(5e-5 / VU_KMS) ** 2] * 3)
    dt = np.array([0.0, 0.5, 1.0])
    a = predict_gap(x0, P0, dt, JD0, MU_EM, n_samples=50, rng=np.random.default_rng(1))
    b = predict_gap(x0, P0, dt, JD0, MU_EM, n_samples=50, rng=np.random.default_rng(1), consider={"P_mc": P0})
    assert np.array_equal(a["theta_ideal"], b["theta_ideal"])
    aug = np.zeros((7, 7))
    aug[:6, :6] = P0
    aug[6, 6] = 0.003 ** 2
    c = predict_gap(x0, P0, dt, JD0, MU_EM, n_samples=50, rng=np.random.default_rng(1),
                    consider={"P_mc": aug, "accel": SRPModel(JD0, 0.0, 1.5)})
    assert np.all(np.isfinite(c["theta_ideal"])) and np.array_equal(c["theta_ut_claim"], a["theta_ut_claim"])
