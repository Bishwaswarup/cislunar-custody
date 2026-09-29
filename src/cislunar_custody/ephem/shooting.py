"""Multiple shooting: transition a CR3BP periodic orbit into a continuous trajectory of
the ephemeris model over a long window (its 'ephemeris counterpart').

Patch points every seg_frac * period are initialised from the CR3BP orbit mapped with
the DE440 Moon (instantaneous frame) and corrected by minimum-norm Newton iterations
on the full-state continuity constraints X_{j+1} = phi(X_j, tau_j -> tau_{j+1}) (no
manoeuvres), solved by Levenberg-Marquardt (plain minimum-norm Newton diverges or
wanders far from the orbit across NRHO perilune passages). All segments and their STMs (gravity-gradient variational equations) are integrated
together, each state carrying its own epoch.
"""
from dataclasses import dataclass

import numpy as np

from ..constants import LU_KM, TU_S, DAY_S
from .transform import synodic_to_inertial


@dataclass
class EphemerisTrajectory:
    model: object
    taus: np.ndarray             # patch epochs [TU since model.jd0]
    X: np.ndarray                # corrected patch states (K+1, 6), inertial LU, LU/TU
    X_guess: np.ndarray          # CR3BP-mapped initial guess
    iterations: int
    residual: float

    def states_at(self, tau):
        """Inertial states at model times tau [TU] (array)."""
        tau = np.atleast_1d(np.asarray(tau, float))
        j = np.clip(np.searchsorted(self.taus, tau, side="right") - 1, 0, len(self.taus) - 2)
        out = np.empty((len(tau), 6))
        for jj in np.unique(j):
            sel = np.where(j == jj)[0]
            for i in sel:                                   # distinct dt per state
                out[i] = self.model.propagate_offsets(self.X[jj], [self.taus[jj]], tau[i] - self.taus[jj])[0]
        return out

    def deviation_km(self):
        """Distance between corrected patch points and the CR3BP-mapped guess [km]."""
        return np.linalg.norm(self.X[:, :3] - self.X_guess[:, :3], axis=1) * LU_KM


def correct_ephemeris_orbit(model, de, orbit, t0_tu, span_days, mu, *, seg_frac=0.1, tol=1e-9,
                            max_iter=60, fd_step=1e-6, lam0=1e-3, newton_km=200.0, verbose=False):
    """Ephemeris counterpart of `orbit` over [model.jd0, model.jd0 + span_days].
    t0_tu: CR3BP orbit time (phase) at model.jd0."""
    dt = seg_frac * orbit.period
    K = int(np.ceil(span_days * DAY_S / TU_S / dt))
    taus = np.arange(K + 1) * dt
    X = np.array([synodic_to_inertial(orbit.states_at(t0_tu + tj)[0], model.jd0 + tj * TU_S / DAY_S, de, mu)
                  for tj in taus])
    X_guess = X.copy()
    off = taus[:-1]

    def evaluate(X):
        Xf, Phi = model.propagate_offsets_stm(X[:-1], off, dt)
        return Xf - X[1:], Phi

    def jac(Phi):
        J = np.zeros((6 * K, 6 * (K + 1)))
        for j in range(K):
            J[6 * j:6 * j + 6, 6 * j:6 * j + 6] = Phi[j]
            J[6 * j:6 * j + 6, 6 * j + 6:6 * j + 12] = -np.eye(6)
        return J

    F, Yf = evaluate(X)
    cost = float(np.sum(F ** 2))
    lam = lam0
    it = 0
    for it in range(1, max_iter + 1):
        res = float(np.max(np.abs(F)))
        if verbose:
            print(f"    LM iter {it - 1}: max continuity error {res * LU_KM:.3e} km  (lambda {lam:.1e})")
        if res < tol:
            break
        J = jac(Yf)
        if res * LU_KM < newton_km:                  # close: minimum-norm Newton (quadratic)
            dX = np.linalg.lstsq(J, -F.ravel(), rcond=None)[0].reshape(K + 1, 6)
            try:
                F_new, Yf_new = evaluate(X + dX)
                c_new = float(np.sum(F_new ** 2))
            except RuntimeError:
                c_new = np.inf
            if c_new < cost:
                X, F, Yf, cost = X + dX, F_new, Yf_new, c_new
                continue
        JtJ = J.T @ J
        g = J.T @ F.ravel()
        D = np.diag(JtJ).copy()
        D[D == 0] = 1.0
        for _ in range(12):
            dX = np.linalg.solve(JtJ + lam * np.diag(D), -g).reshape(K + 1, 6)
            try:
                F_new, Yf_new = evaluate(X + dX)
                c_new = float(np.sum(F_new ** 2))
            except RuntimeError:
                c_new = np.inf
            if c_new < cost:
                X, F, Yf, cost = X + dX, F_new, Yf_new, c_new
                lam = max(lam / 3.0, 1e-12)
                break
            lam *= 4.0
        else:
            break                                    # no progress possible
    res = float(np.max(np.abs(F)))
    return EphemerisTrajectory(model, taus, X, X_guess, it, res)
