from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jplephem")
from cislunar_custody.ephem import (DE440, DEFAULT_KERNEL, EphemerisModel, synodic_to_inertial,  # noqa: E402
                                    inertial_to_synodic, transform_jacobian)
from cislunar_custody.catalogue import load_named  # noqa: E402
from cislunar_custody.constants import MU_EM, LU_KM, TU_S, DAY_S  # noqa: E402
from cislunar_custody.dynamics import propagate  # noqa: E402
from cislunar_custody.frames import moon_position  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso, jd_grid  # noqa: E402

if not DEFAULT_KERNEL.exists():
    pytest.skip("data/de440s.bsp not downloaded", allow_module_level=True)

CAT = Path(__file__).resolve().parents[1] / "data" / "catalogue.npz"
JD = jd_from_iso("2027-03-01T00:00")


@pytest.fixture(scope="module")
def de():
    return DE440()


def test_de440_moon_matches_low_precision_model(de):
    jd = jd_grid(JD, 30, 720)
    m_de = de.moon_position(jd).T
    m_lp = moon_position(jd)
    r = np.linalg.norm(m_de, axis=1)
    assert r.min() > 355000 and r.max() < 408000
    ang = np.degrees(np.arccos(np.sum(m_de * m_lp, 1) / r / np.linalg.norm(m_lp, axis=1)))
    assert ang.max() < 0.5          # analytic model good to a few tenths of a degree


def test_transform_round_trip(de):
    s = load_named(CAT, "NRHO_9:2").states_at(0.4)[0]
    x = synodic_to_inertial(s, JD, de, MU_EM)
    assert np.allclose(inertial_to_synodic(x, JD, de, MU_EM), s, atol=1e-12)
    assert np.linalg.det(transform_jacobian(s, JD, de, MU_EM)) != 0


def test_ephemeris_nrho_stays_close_to_cr3bp_for_a_day(de):
    """Same initial state: the models separate slowly (Sun, eccentricity, Moon inclination)."""
    orb = load_named(CAT, "NRHO_9:2")
    s0 = orb.states_at(0.5 * orb.period)[0]                   # start at apolune
    model = EphemerisModel(de, JD, 2.0)
    tau = 1.0 * DAY_S / TU_S
    x1 = model.propagate_dense(synodic_to_inertial(s0, JD, de, MU_EM), np.array([0.0, tau]))[-1, 0]
    s1 = propagate(s0, tau, MU_EM).y[:, -1]
    x1_cr = synodic_to_inertial(s1, JD + 1.0, de, MU_EM)
    dist_km = np.linalg.norm(x1[:3] - x1_cr[:3]) * LU_KM
    assert dist_km < 2000.0


def test_multiple_shooting_l1_halo_converges(de):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from visibility_study import representative_orbits
    from cislunar_custody.ephem import correct_ephemeris_orbit
    orb = representative_orbits()["L1 halo (Az~30k km)"]
    m = EphemerisModel(de, JD, 23.0)
    tr = correct_ephemeris_orbit(m, de, orb, 0.0, 20.0, MU_EM)
    assert tr.residual * LU_KM < 1e-3                  # continuous to < 1 m
    assert np.median(tr.deviation_km()) < 20000.0      # stays near the CR3BP orbit
