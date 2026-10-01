import numpy as np
import pytest

from cislunar_custody.catalogue import load_named
from cislunar_custody.constants import TU_S, DAY_S
from cislunar_custody.frames import Ephemeris
from cislunar_custody.observability import radec_jacobian, arc_information
from cislunar_custody.sensors import SITES, radec
from cislunar_custody.timeutil import jd_from_iso
from pathlib import Path

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"


def test_radec_jacobian_matches_finite_difference():
    rho = np.array([[250000.0, -120000.0, 90000.0]])
    J = radec_jacobian(rho)[0]
    h = 1e-2
    cols = []
    for e in np.eye(3):
        ap, dp = radec(rho + h * e)
        am, dm = radec(rho - h * e)
        cols.append([(ap - am)[0] / (2 * h), (dp - dm)[0] / (2 * h)])
    assert np.allclose(J, np.array(cols).T, rtol=1e-6, atol=1e-14)


def _setup(n_epochs, span_tu=0.6):
    orb = load_named(CAT, "NRHO_9:2")
    jd0 = jd_from_iso("2027-02-01T00:00")
    t = np.linspace(0.0, span_tu, n_epochs)
    eph = Ephemeris(jd0 + t * TU_S / DAY_S)
    r_site, _ = eph.site_eci(SITES["Hanle"])
    end_eph = Ephemeris(np.array([jd0 + span_tu * TU_S / DAY_S]))
    be = tuple(b[0] for b in end_eph.basis)
    return orb, t, r_site, eph.basis, span_tu, be


def _rank(I):
    D = np.sqrt(np.abs(np.diag(I)))
    D[D == 0] = 1.0                      # unobserved components (e.g. velocity at t0)
    s = np.linalg.svd(I / np.outer(D, D), compute_uv=False)
    return int(np.sum(s > 1e-10 * s[0]))


def test_single_epoch_gives_rank_two():
    orb, t, rs, bm, te, be = _setup(1)
    res = arc_information(orb, 0.0, t, rs, bm, [te], [be])[0]
    assert _rank(res.info) == 2 and not res.observable


def test_arc_is_fully_observable():
    orb, t, rs, bm, te, be = _setup(12)
    res = arc_information(orb, 0.0, t, rs, bm, [te], [be])[0]
    assert _rank(res.info) == 6 and res.observable
    assert res.sigma_pos_km[0] > res.sigma_pos_km[-1] > 0


def test_information_scales_with_inverse_noise_variance():
    orb, t, rs, bm, te, be = _setup(12)
    a = arc_information(orb, 0.0, t, rs, bm, [te], [be], sigma_arcsec=1.0)[0].info
    b = arc_information(orb, 0.0, t, rs, bm, [te], [be], sigma_arcsec=2.0)[0].info
    assert np.allclose(a, 4.0 * b)


def test_noise_model_is_isotropic_on_the_sky_everywhere():
    """Simulator, filters and CRLB share one noise model: sigma on the sky in both axes,
    i.e. R = diag(sigma^2 / cos^2(dec), sigma^2) in (RA, Dec)."""
    from cislunar_custody.constants import MU_EM
    from cislunar_custody.filters.common import AnglesModel, Measurement
    from cislunar_custody.observability import sky_jacobian
    from cislunar_custody.sensors import simulate_radec
    from cislunar_custody.sensors.measurement import ARCSEC
    rho = np.array([[200000.0, 150000.0, 230000.0]])                  # dec ~ 43 deg
    ra0, dec0 = radec(rho)
    rng = np.random.default_rng(3)
    ra, dec = simulate_radec(np.repeat(rho, 20000, 0), np.zeros((20000, 3)), 1.0, rng)
    d = np.stack([(ra - ra0 + np.pi) % (2 * np.pi) - np.pi, dec - dec0], 1)
    m = Measurement(0.0, "x", np.zeros(3), (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]), np.r_[ra0, dec0])
    R = AnglesModel(MU_EM, 1.0).R_for(m)
    assert np.allclose(np.cov(d.T), R, rtol=0.05, atol=0.05 * R[1, 1])
    # the CRLB rows are the same model, whitened: sky_jacobian^T sky_jacobian = J^T R^-1 J sigma^2
    J = radec_jacobian(rho)[0]
    Js = sky_jacobian(rho)[0]
    assert np.allclose(Js.T @ Js, J.T @ np.linalg.inv(R) @ J * ARCSEC ** 2, rtol=1e-6)
