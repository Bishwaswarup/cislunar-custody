"""Regularised bootstrap particle filter with progressive correction and a UKF handover.

Angles-only likelihoods (1 arcsec ~ 2 km at lunar distance) are extremely peaked
relative to a post-gap cloud of 10^3-10^4 km, so a single Bayes update collapses onto a
few particles. The update is therefore split into tempered stages
p(z|x)^phi_1 ... p(z|x)^phi_k (sum phi = 1), each phi chosen by bisection so the
effective sample size stays at ess_frac * N; after each stage the particles are
resampled (systematic) and jittered with a Gaussian kernel (a fraction of Silverman's
bandwidth) to restore diversity (Oudjane & Musso 2000; Musso et al. 2001).

Over long, dense tracking arcs a finite particle set cannot represent the very thin
posterior and slowly impoverishes (its covariance drops below the CRLB). Once the cloud
is small enough for the dynamics to be locally linear (max position sigma below
`handoff_km` after at least `handoff_min_updates` updates) the filter therefore hands
its mean and covariance to a UKF (`handoff_km=None` disables the handover).
"""
import numpy as np

from ..constants import LU_KM
from ..dynamics.cr3bp import propagate_many
from .common import AnglesModel, process_noise
from .ukf import UKF


class ParticleFilter:
    name = "PF"

    def __init__(self, mu, model: AnglesModel, n=3000, q_psd_km2_s3=0.0, ess_frac=0.5,
                 max_stages=40, jitter_scale=0.5, rng=None, rtol=1e-9, atol=1e-11,
                 handoff_km=50.0, handoff_min_updates=10):
        self.mu, self.model, self.n, self.q = mu, model, n, q_psd_km2_s3
        self.ess_frac, self.max_stages, self.jitter_scale = ess_frac, max_stages, jitter_scale
        self.rng = np.random.default_rng() if rng is None else rng
        self.rtol, self.atol = rtol, atol
        self.handoff_km, self.handoff_min_updates = handoff_km, handoff_min_updates
        self.ukf = UKF(mu, model, q_psd_km2_s3)
        d = 6
        self.h = jitter_scale * (4.0 / (n * (d + 2))) ** (1.0 / (d + 4))

    # ---- belief: {"mode": "pf", "X", "logw", "n_upd"} or {"mode": "ukf", "x", "P"} ----
    def init(self, x0, P0):
        X = x0 + self.rng.standard_normal((self.n, 6)) @ np.linalg.cholesky(P0).T
        return {"mode": "pf", "X": X, "logw": np.zeros(self.n), "n_upd": 0}

    def size(self, b):
        return self.n if b["mode"] == "pf" else 1

    @staticmethod
    def _w(logw):
        w = np.exp(logw - logw.max())
        return w / w.sum()

    def moments(self, b):
        if b["mode"] == "ukf":
            return b["x"], b["P"]
        w = self._w(b["logw"])
        m = w @ b["X"]
        D = b["X"] - m
        return m, (w[:, None] * D).T @ D

    def ess(self, b):
        return 1.0 / np.sum(self._w(b["logw"]) ** 2) if b["mode"] == "pf" else float("nan")

    def predict_belief(self, b, dt):
        if dt == 0.0:
            return b
        if b["mode"] == "ukf":
            x, P = self.ukf.predict(b["x"], b["P"], dt)
            return {"mode": "ukf", "x": x, "P": P}
        X = propagate_many(b["X"], dt, self.mu, self.rtol, self.atol)
        if self.q > 0:
            Q = process_noise(dt, self.mu, self.q)
            X = X + self.rng.standard_normal(X.shape) @ np.linalg.cholesky(Q + 1e-30 * np.eye(6)).T
        return {**b, "X": X}

    def _loglik(self, X, m):
        r = self.model.whiten(self.model.residual(m.z[None, :], self.model.h(X, m)), m)
        return -0.5 * np.sum(r ** 2, axis=1) / self.model.sigma ** 2

    def _ess_of(self, logw):
        w = self._w(logw)
        return 1.0 / np.sum(w ** 2)

    def _resample_jitter(self, X, logw):
        w = self._w(logw)
        u = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.n - 1)
        X = X[idx]
        C = np.cov(X.T) + 1e-30 * np.eye(6)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            ev, V = np.linalg.eigh(C)
            L = V * np.sqrt(np.clip(ev, 0, None))
        X = X + self.h * self.rng.standard_normal(X.shape) @ L.T
        return X, np.zeros(self.n)

    def update_belief(self, b, m):
        if b["mode"] == "ukf":
            x, P, _, _ = self.ukf.update(b["x"], b["P"], m)
            return {"mode": "ukf", "x": x, "P": P}
        X, logw = b["X"], b["logw"]
        target = self.ess_frac * self.n
        remaining, stages = 1.0, 0
        while remaining > 1e-12 and stages < self.max_stages:
            ll = self._loglik(X, m)
            if self._ess_of(logw + remaining * ll) >= target:
                logw = logw + remaining * ll
                break
            lo, hi = 0.0, remaining
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if self._ess_of(logw + mid * ll) >= target:
                    lo = mid
                else:
                    hi = mid
            phi = max(lo, 1e-12)
            X, logw = self._resample_jitter(X, logw + phi * ll)
            remaining -= phi
            stages += 1
        nb = {"mode": "pf", "X": X, "logw": logw, "n_upd": b["n_upd"] + 1}
        if self.handoff_km is not None and nb["n_upd"] >= self.handoff_min_updates:
            x, P = self.moments(nb)
            if np.sqrt(np.linalg.eigvalsh(P[:3, :3]).max()) * LU_KM < self.handoff_km:
                return {"mode": "ukf", "x": x, "P": P}
        return nb
