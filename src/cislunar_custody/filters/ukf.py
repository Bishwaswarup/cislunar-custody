"""Unscented Kalman filter (scaled sigma points, Van der Merwe weights).

Defaults alpha = 1, beta = 2, kappa = 0 give lambda = 0: 2n equally weighted points plus
a centre point that only carries the beta covariance correction (always PSD). All sigma
points are propagated together in one vectorised integration.
"""
import numpy as np

from ..dynamics.cr3bp import propagate_many
from .common import AnglesModel, process_noise, GaussianBeliefMixin


def _sqrt_psd(P):
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (P + P.T))
        return V * np.sqrt(np.clip(w, 1e-30, None))


class UKF(GaussianBeliefMixin):
    name = "UKF"

    def __init__(self, mu, model: AnglesModel, q_psd_km2_s3=0.0, alpha=1.0, beta=2.0,
                 kappa=0.0, rtol=1e-10, atol=1e-12):
        self.mu, self.model, self.q = mu, model, q_psd_km2_s3
        self.rtol, self.atol = rtol, atol
        n = 6
        lam = alpha ** 2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2 * n + 1, 0.5 / self.c)
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self.c
        self.Wc[0] = lam / self.c + (1.0 - alpha ** 2 + beta)

    def sigma_points(self, x, P):
        L = _sqrt_psd(self.c * P)
        return np.vstack([x, x + L.T, x - L.T])

    def _moments(self, X):
        m = self.Wm @ X
        D = X - m
        return m, (self.Wc[:, None] * D).T @ D, D

    def predict(self, x, P, dt):
        if dt == 0.0:
            return x, P
        X = propagate_many(self.sigma_points(x, P), dt, self.mu, self.rtol, self.atol)
        m, Pp, _ = self._moments(X)
        Pp = Pp + process_noise(dt, self.mu, self.q)
        return m, 0.5 * (Pp + Pp.T)

    def update(self, x, P, m_):
        X = self.sigma_points(x, P)
        Z = self.model.h(X, m_)
        ref = Z[0]
        dZ = self.model.residual(Z, ref)                  # wrap RA around the centre point
        zbar = ref + self.Wm @ dZ
        dZ = self.model.residual(Z, zbar)
        dX = X - (self.Wm @ X)
        S = (self.Wc[:, None] * dZ).T @ dZ + self.model.R_for(m_)
        Pxz = (self.Wc[:, None] * dX).T @ dZ
        K = np.linalg.solve(S, Pxz.T).T
        r = self.model.residual(m_.z, zbar)
        x = x + K @ r
        P = P - K @ S @ K.T
        return x, 0.5 * (P + P.T), r, S
