import numpy as np
import pytest

from cislunar_custody.constants import MU_EM, LU_KM, TU_DAYS, SYNODIC_MONTH_DAYS
from cislunar_custody.dynamics import (correct, continue_family, find_by_period,
                                       richardson_halo, lyapunov_seed, dro_seed)
from cislunar_custody.dynamics.periodic import PLANAR_FREE, PLANAR_CONSTRAIN

MU = MU_EM


@pytest.mark.parametrize("L, northern", [(1, True), (2, False)])
def test_richardson_seed_corrects_to_periodic_halo(L, northern):
    s, T_guess = richardson_halo(MU, 8000 / LU_KM, L=L, northern=northern)
    orb = correct(s, MU, free=(0, 4))
    assert orb.meta["iterations"] <= 8
    assert orb.closure_error() < 1e-8
    assert orb.period == pytest.approx(T_guess, rel=0.02)   # 3rd-order guess is close
    assert np.sign(orb.state0[2]) == (1 if northern else -1)


def test_monodromy_has_reciprocal_and_unit_multipliers():
    s, _ = richardson_halo(MU, 8000 / LU_KM, L=1)
    lam = correct(s, MU, free=(0, 4)).floquet()
    assert np.prod(lam).real == pytest.approx(1.0, abs=1e-6)
    assert np.sum(np.abs(lam - 1.0) < 1e-4) == 2              # trivial pair
    big = lam[np.argmax(np.abs(lam))]
    assert np.min(np.abs(lam - 1.0 / big)) < 1e-6 * abs(big)  # reciprocal pair


@pytest.mark.parametrize("L", [1, 2])
def test_lyapunov_orbit(L):
    s, _ = lyapunov_seed(MU, 0.005, L=L)
    orb = correct(s, MU, free=(4,), constrain=PLANAR_CONSTRAIN)
    assert orb.closure_error() < 1e-8
    assert orb.state0[2] == 0.0
    assert orb.stability_index() > 100          # collinear Lyapunov orbits: strongly unstable


def test_dro_is_linearly_stable():
    s, _ = dro_seed(MU, 0.05)
    orb = correct(s, MU, free=(4,), constrain=PLANAR_CONSTRAIN)
    assert orb.closure_error() < 1e-8
    assert orb.stability_index() < 1.0 + 1e-6


def test_continuation_keeps_orbits_periodic():
    s, _ = richardson_halo(MU, 8000 / LU_KM, L=1)
    fam = continue_family(correct(s, MU, free=(0, 4)), 6, ds=2e-3)
    assert len(fam) == 6
    assert all(m.closure_error() < 1e-8 for m in fam)
    assert np.all(np.diff([m.state0[2] for m in fam]) > 0)   # amplitude grows


@pytest.mark.slow
def test_nrho_9_2_matches_literature():
    """9:2 synodic-resonant L2 southern NRHO: perilune ~3,250 km, apolune ~71,000 km."""
    s, _ = richardson_halo(MU, 8000 / LU_KM, L=2, northern=False)
    orb = correct(s, MU, free=(0, 4))
    stop = lambda o: o.moon_distance_range(600)[0] * LU_KM < 2500
    fam = continue_family(orb, 200, ds=2e-3, ds_max=1.5e-2, stop=stop)
    T92 = SYNODIC_MONTH_DAYS * 2 / 9 / TU_DAYS
    nrho = find_by_period(fam, T92)
    rp, ra = nrho.moon_distance_range(4000)
    assert nrho.period * TU_DAYS == pytest.approx(6.5624, abs=1e-3)
    assert 3000 < rp * LU_KM < 3500
    assert 69000 < ra * LU_KM < 73000
    assert 1.0 < nrho.stability_index() < 2.0      # nearly stable
