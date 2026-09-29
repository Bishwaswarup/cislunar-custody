"""Extended Kalman filter with STM covariance propagation and Joseph-form updates."""
import numpy as np

from ..dynamics.cr3bp import propagate
from .common import AnglesModel, process_noise


class EKF:
    name = "EKF"

    def __init__(self, mu, model: AnglesModel, q_psd_km2_s3=0.0, rtol=1e-10, atol=1e-12):
        self.mu, self.model, self.q = mu, model, q_psd_km2_s3
        self.rtol, self.atol = rtol, atol

    def predict(self, x, P, dt):
        if dt == 0.0:
            return x, P
        y = propagate(x, dt, self.mu, with_stm=True, rtol=self.rtol, atol=self.atol).y[:, -1]
        F = y[6:].reshape(6, 6)
        P = F @ P @ F.T + process_noise(dt, self.mu, self.q)
        return y[:6], 0.5 * (P + P.T)

    def update(self, x, P, m):
        H = self.model.H(x, m)
        r = self.model.residual(m.z, self.model.h(x, m)[0])
        S = H @ P @ H.T + self.model.R
        K = np.linalg.solve(S, H @ P).T
        x = x + K @ r
        A = np.eye(6) - K @ H
        P = A @ P @ A.T + K @ self.model.R @ K.T
        return x, 0.5 * (P + P.T), r, S
