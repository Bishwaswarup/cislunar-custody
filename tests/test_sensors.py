import numpy as np
import pytest

from cislunar_custody.constants import MU_EM, LU_KM
from cislunar_custody.frames import (sun_position, moon_position, gmst, obliquity,
                                     synodic_basis, synodic_to_eci, Ephemeris, AU_KM)
from cislunar_custody.sensors import (SITES, TELESCOPES, Target, apparent_magnitude,
                                      ks_moon_sky_nl, is_shadowed, site_visibility, runs,
                                      gap_stats, radec)
from cislunar_custody.timeutil import jd_from_iso, jd_grid


def _elong_deg(jd):
    s, m = sun_position(jd), moon_position(jd)
    return np.rad2deg(np.arccos(np.dot(s, m) / np.linalg.norm(s) / np.linalg.norm(m)))


def test_gmst_at_j2000():
    assert np.rad2deg(gmst(2451545.0)) == pytest.approx(280.46061837, abs=1e-6)


def test_sun_declination_at_june_solstice():
    s = sun_position(jd_from_iso("2026-06-21T12:00"))
    dec = np.rad2deg(np.arcsin(s[2] / np.linalg.norm(s)))
    assert 23.30 < dec < 23.50


def test_moon_distance_and_latitude_bounds():
    jd = jd_grid(jd_from_iso("2026-01-01T00:00"), 365, 60)
    m = moon_position(jd)
    r = np.linalg.norm(m, axis=1)
    assert r.min() > 355000 and r.max() < 408000
    eps = obliquity(jd)
    n_ecl = np.stack([np.zeros_like(eps), -np.sin(eps), np.cos(eps)], axis=1)
    beta = np.rad2deg(np.arcsin(np.sum(m * n_ecl, axis=1) / r))
    assert np.max(np.abs(beta)) < 5.35


def test_full_and_new_moon_january_2024():
    assert _elong_deg(jd_from_iso("2024-01-25T17:54")) > 175.0   # full moon
    assert _elong_deg(jd_from_iso("2024-01-11T11:57")) < 6.0     # new moon


def test_synodic_mapping():
    jd = jd_grid(jd_from_iso("2027-01-01T00:00"), 30, 360)
    xh, yh, zh = synodic_basis(jd)
    for a, b in ((xh, yh), (yh, zh), (zh, xh)):
        assert np.max(np.abs(np.sum(a * b, axis=1))) < 1e-12
    n = len(jd)
    earth = synodic_to_eci(np.tile([-MU_EM, 0, 0], (n, 1)), jd, MU_EM)
    moon = synodic_to_eci(np.tile([1 - MU_EM, 0, 0], (n, 1)), jd, MU_EM)
    assert np.max(np.linalg.norm(earth, axis=1)) < 1e-9
    assert np.allclose(np.linalg.norm(moon, axis=1), LU_KM)
    # mapped Moon direction = real Moon direction
    m = moon_position(jd)
    assert np.allclose(moon / LU_KM, m / np.linalg.norm(m, axis=1)[:, None], atol=1e-12)


def test_target_at_zenith_has_90deg_elevation():
    jd = np.array([jd_from_iso("2027-03-01T20:00")])
    eph = Ephemeris(jd)
    r_site, up = eph.site_eci(SITES["Hanle"])
    tgt = r_site + 1000.0 * up
    v = site_visibility(eph, tgt, "Hanle", SITES["Hanle"], TELESCOPES["1m"], Target())
    assert v.elev_deg[0] == pytest.approx(90.0, abs=1e-6)


def test_earth_shadow():
    sun = np.array([[AU_KM, 0.0, 0.0]] * 2)
    pts = np.array([[-20000.0, 1000.0, 0.0], [20000.0, 0.0, 0.0]])
    sh = is_shadowed(pts, np.zeros_like(pts), 6378.137, sun)
    assert sh.tolist() == [True, False]


def test_photometry_reference_value():
    """2 m sphere, albedo 0.2, at Moon distance, zero phase: m ~ 18.37."""
    obj = np.array([384400.0, 0, 0])
    sun = np.array([384400.0 - AU_KM, 0, 0])      # behind the observer, 1 AU from obj
    m = apparent_magnitude(obj, np.zeros(3), sun, 1.0, 0.2)
    assert m == pytest.approx(18.37, abs=0.02)


def test_moon_sky_brightness_behaviour():
    near = ks_moon_sky_nl(0.0, 20.0, 40.0, 30.0, 0.15)
    far = ks_moon_sky_nl(0.0, 90.0, 40.0, 30.0, 0.15)
    half = ks_moon_sky_nl(90.0, 20.0, 40.0, 30.0, 0.15)
    below = ks_moon_sky_nl(0.0, 20.0, 95.0, 30.0, 0.15)
    assert near > far > 0 and near > half and below == 0.0


def test_gap_statistics():
    vis = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0], bool)
    lens, starts = runs(~vis)
    assert lens.tolist() == [3, 1, 2] and starts.tolist() == [2, 6, 8]
    g = gap_stats(vis, 0.5)                 # trailing gap is censored
    assert g["n_gaps"] == 2 and g["max_gap_d"] == 1.5
    never = gap_stats(np.zeros(10, bool), 0.5)
    assert never["frac_visible"] == 0 and np.isnan(never["max_gap_d"])


def test_radec_axes():
    ra, dec = radec(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))
    assert ra[0] == pytest.approx(np.pi / 2) and dec[1] == pytest.approx(np.pi / 2)
