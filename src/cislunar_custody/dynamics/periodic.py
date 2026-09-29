"""Symmetric periodic orbits of the CR3BP: differential correction and
pseudo-arclength continuation.

All orbits here are symmetric about the xz-plane: they start on y = 0 with
vx = vz = 0 (perpendicular crossing) and are corrected so the next y = 0 crossing
is also perpendicular. The period is twice that half-period.

Free-variable / constraint conventions (indices into the state vector):
    3-D (halo, NRHO):   free = (0, 2, 4) -> x0, z0, vy0   constrain = (3, 5) -> vx_f, vz_f
    planar (Lyapunov, DRO): free = (0, 4) -> x0, vy0      constrain = (3,)   -> vx_f
Fix one coordinate by dropping it from `free` (e.g. free=(0, 4) fixes z0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cr3bp import eom, propagate, stm, jacobi, distances

HALO_FREE, HALO_CONSTRAIN = (0, 2, 4), (3, 5)
PLANAR_FREE, PLANAR_CONSTRAIN = (0, 4), (3,)


class CorrectionError(RuntimeError):
    """Differential correction or continuation failed to converge."""


@dataclass
class PeriodicOrbit:
    state0: np.ndarray
    period: float
    mu: float
    family: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.state0 = np.asarray(self.state0, float)

    @property
    def jacobi(self) -> float:
        return float(jacobi(self.state0, self.mu))

    def monodromy(self, **kw) -> np.ndarray:
        _, M = stm(self.state0, self.period, self.mu, **kw)
        return M

    def floquet(self, **kw):
        """Floquet multipliers (eigenvalues of the monodromy matrix)."""
        return np.linalg.eigvals(self.monodromy(**kw))

    def stability_index(self, **kw) -> float:
        """max over pairs of nu = |lambda + 1/lambda| / 2  (nu <= 1: linearly stable)."""
        lam = self.floquet(**kw)
        return float(np.max(0.5 * np.abs(lam + 1.0 / lam)))

    def trajectory(self, n=2000, periods=1.0):
        t = np.linspace(0.0, periods * self.period, n)
        sol = propagate(self.state0, (0.0, t[-1]), self.mu, t_eval=t)
        return sol.t, sol.y.T

    def moon_distance_range(self, n=4000):
        """(min, max) distance to the Moon over one period [LU]."""
        _, S = self.trajectory(n)
        _, r2 = distances(S, self.mu)
        return float(r2.min()), float(r2.max())

    def states_at(self, t, phase=0.0):
        """States at times t [TU] using periodicity (dense output over one period).
        phase: orbit time [TU] at t = 0."""
        if "_dense" not in self.meta:
            self.meta["_dense"] = propagate(self.state0, self.period, self.mu, dense_output=True).sol
        tt = np.mod(np.asarray(t, float) + phase, self.period)
        return np.atleast_2d(self.meta["_dense"](tt).T)

    def z_extreme(self, n=2000) -> float:
        """Signed z at the largest |z| (> 0: northern, < 0: southern)."""
        _, S = self.trajectory(n)
        return float(S[np.argmax(np.abs(S[:, 2])), 2])

    def closure_error(self) -> float:
        sf = propagate(self.state0, self.period, self.mu).y[:, -1]
        return float(np.linalg.norm(sf - self.state0))


def _y_crossing_event(direction):
    def ev(t, y, mu):
        return y[1]
    ev.terminal = True
    ev.direction = direction
    return ev


def half_period(s0, mu, *, with_stm=True, t_max=30.0):
    """Propagate to the next y = 0 crossing. Returns (t_half, s_f, Phi_f or None)."""
    if s0[4] == 0.0:
        raise CorrectionError("vy0 = 0: not a perpendicular xz-plane crossing")
    direction = -np.sign(s0[4])
    sol = propagate(s0, (0.0, t_max), mu, with_stm=with_stm,
                    events=_y_crossing_event(direction))
    if sol.t_events[0].size == 0:
        raise CorrectionError("no y = 0 crossing found")
    th = float(sol.t_events[0][0])
    yf = sol.y_events[0][0]
    return th, yf[:6], (yf[6:].reshape(6, 6) if with_stm else None)


def _residual(s, mu, free, constrain, period_target):
    """Constraint vector F and its Jacobian D w.r.t. the free variables.

    The crossing time varies with the initial state, so
    d s_f / dX = Phi[:, free] + f(s_f) * dt/dX,  dt/dX = -Phi[1, free] / ydot_f.
    """
    try:
        th, sf, Phi = half_period(s, mu)
    except RuntimeError as exc:  # includes CorrectionError
        raise CorrectionError(str(exc)) from exc
    fdot = eom(0.0, sf, mu)
    dtdX = -Phi[1, free] / fdot[1]
    F = sf[constrain]
    D = Phi[np.ix_(constrain, free)] + np.outer(fdot[constrain], dtdX)
    if period_target is not None:
        F = np.append(F, 2.0 * th - period_target)
        D = np.vstack([D, 2.0 * dtdX])
    return F, D, th


def _symmetric(s0):
    s = np.array(s0, float)
    s[1] = 0.0
    s[3] = 0.0
    s[5] = 0.0
    return s


def correct(s0, mu, *, free=HALO_FREE, constrain=HALO_CONSTRAIN, period_target=None,
            tol=1e-11, max_iter=50, max_step=0.05, family=""):
    """Single-shooting differential correction to a symmetric periodic orbit.

    Uses the minimum-norm Newton step when there are more free variables than
    constraints. `period_target` (full period, TU) adds a period constraint.
    """
    free, constrain = np.asarray(free), np.asarray(constrain)
    s = _symmetric(s0)
    for it in range(max_iter):
        F, D, th = _residual(s, mu, free, constrain, period_target)
        if np.max(np.abs(F)) < tol:
            return PeriodicOrbit(s, 2.0 * th, mu, family, {"iterations": it})
        dX = np.linalg.lstsq(D, -F, rcond=None)[0]
        n = np.linalg.norm(dX)
        if n > max_step:
            dX *= max_step / n
        s[free] += dX
    raise CorrectionError(f"no convergence in {max_iter} iterations (|F| = {np.max(np.abs(F)):.2e})")


def _null_vector(D):
    return np.linalg.svd(D)[2][-1]


def continue_family(orbit, n_members, *, ds=1e-3, direction=1.0, free=HALO_FREE,
                    constrain=HALO_CONSTRAIN, ds_min=1e-6, ds_max=2e-2, tol=1e-11,
                    stop=None, family=None, verbose=False):
    """Pseudo-arclength continuation of a family of symmetric periodic orbits.

    Newton on G(X) = [F(X); (X - X_prev) . tau - ds] with the family tangent tau
    = null(D). Step size adapts to Newton effort. `stop(orbit) -> bool` ends early.
    Returns the list of orbits, starting with `orbit`.
    """
    free, constrain = np.asarray(free), np.asarray(constrain)
    mu = orbit.mu
    fam = family or orbit.family
    s = _symmetric(orbit.state0)
    _, D, _ = _residual(s, mu, free, constrain, None)
    tau = direction * _null_vector(D)
    X = s[free].copy()
    out = [orbit]
    while len(out) < n_members:
        Y = X + ds * tau
        converged = False
        for it in range(12):
            trial = s.copy()
            trial[free] = Y
            try:
                F, D, th = _residual(trial, mu, free, constrain, None)
            except CorrectionError:
                break
            G = np.append(F, (Y - X) @ tau - ds)
            if np.max(np.abs(G)) < tol:
                converged = True
                break
            Y = Y - np.linalg.solve(np.vstack([D, tau]), G)
        if not converged:
            ds *= 0.5
            if ds < ds_min:
                if verbose:
                    print("continuation stopped: step below ds_min")
                break
            continue
        new_tau = _null_vector(D)
        if new_tau @ tau < 0:
            new_tau = -new_tau
        X, tau, s = Y, new_tau, trial
        orb = PeriodicOrbit(trial.copy(), 2.0 * th, mu, fam)
        out.append(orb)
        if verbose:
            print(f"[{len(out):3d}] T = {orb.period:.5f}  C = {orb.jacobi:.6f}  ds = {ds:.2e}")
        if it <= 3:
            ds = min(1.5 * ds, ds_max)
        if stop is not None and stop(orb):
            break
    return out


def find_by_period(members, period, *, tol=1e-11, family=None):
    """Pick the family member bracketing `period` and correct with a period constraint
    (3-D orbits: free x0, z0, vy0; constraints vx_f, vz_f, T)."""
    T = np.array([m.period for m in members])
    idx = np.where(np.diff(np.sign(T - period)) != 0)[0]
    if idx.size == 0:
        raise CorrectionError("target period not bracketed by the family")
    i = idx[0]
    w = (period - T[i]) / (T[i + 1] - T[i])
    guess = (1 - w) * members[i].state0 + w * members[i + 1].state0
    orb = correct(guess, members[i].mu, free=HALO_FREE, constrain=HALO_CONSTRAIN,
                  period_target=period, tol=tol, family=family or members[i].family)
    return orb
