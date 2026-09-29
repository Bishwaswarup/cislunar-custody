import numpy as np
import pytest

from cislunar_custody.constants import MU_EM
from cislunar_custody.dynamics import (eom, jacobian, jacobi, propagate, stm, lagrange_points)

MU = MU_EM
S_TEST = np.array([0.82338562, 0.0, 0.02227785, 0.0, 0.13418412, 0.0])  # near an L1 halo


def test_lagrange_points_are_equilibria():
    L = lagrange_points(MU)
    for name, r in L.items():
        acc = eom(0.0, np.r_[r, 0, 0, 0], MU)[3:]
        assert np.max(np.abs(acc)) < 1e-12, name
    # published Earth-Moon values (mu ~ 0.01215)
    assert L["L1"][0] == pytest.approx(0.836915, abs=1e-5)
    assert L["L2"][0] == pytest.approx(1.155682, abs=1e-5)
    assert L["L3"][0] == pytest.approx(-1.005063, abs=1e-5)


def test_jacobian_matches_finite_difference():
    s = S_TEST + np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
    A = jacobian(s, MU)
    h = 1e-6
    A_fd = np.column_stack([(eom(0, s + h * e, MU) - eom(0, s - h * e, MU)) / (2 * h)
                            for e in np.eye(6)])
    assert np.max(np.abs(A - A_fd)) < 1e-7


def test_stm_matches_finite_difference():
    """STM from the variational equations vs Richardson-extrapolated central
    differences of the flow (plain central differences carry an O(h^2) error of
    ~6e-6 here because |Phi| ~ 120 over t = 1.5)."""
    t = 1.5
    rt = dict(rtol=2.3e-14, atol=1e-16)
    _, Phi = stm(S_TEST, t, MU, **rt)

    def central(h):
        cols = []
        for e in np.eye(6):
            fp = propagate(S_TEST + h * e, t, MU, **rt).y[:, -1]
            fm = propagate(S_TEST - h * e, t, MU, **rt).y[:, -1]
            cols.append((fp - fm) / (2 * h))
        return np.column_stack(cols)

    h = 2e-5
    Phi_fd = (4 * central(h / 2) - central(h)) / 3   # cancels the h^2 term
    rel = np.max(np.abs(Phi - Phi_fd)) / np.max(np.abs(Phi))
    assert rel < 1e-8, rel


def test_stm_is_symplectic():
    _, Phi = stm(S_TEST, 2.0, MU)
    assert np.linalg.det(Phi) == pytest.approx(1.0, abs=1e-8)


def test_jacobi_constant_conserved():
    sol = propagate(S_TEST, 10.0, MU, t_eval=np.linspace(0, 10, 200))
    C = jacobi(sol.y.T, MU)
    assert np.max(np.abs(C - C[0])) < 1e-10
