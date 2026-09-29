"""Shared pieces: measurement records, the RA/Dec model in the synodic state, process noise."""
from dataclasses import dataclass

import numpy as np

from ..constants import LU_KM, TU_S, VU_KMS
from ..observability.fisher import radec_jacobian
from ..sensors.measurement import ARCSEC


@dataclass
class Measurement:
    t: float                 # TU since scenario start
    site: str
    r_site: np.ndarray       # geocentric inertial site position [km] (3,)
    basis: tuple             # synodic unit vectors (x_hat, y_hat, z_hat) at t, each (3,)
    z: np.ndarray            # [ra, dec] [rad]


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class AnglesModel:
    """RA/Dec of a CR3BP state seen from a ground site (same mapping as the truth)."""

    def __init__(self, mu, sigma_arcsec=1.0):
        self.mu = mu
        self.sigma = sigma_arcsec * ARCSEC
        self.R = np.eye(2) * self.sigma ** 2

    def _rho(self, S, m):
        S = np.atleast_2d(S)
        xh, yh, zh = m.basis
        r = LU_KM * ((S[:, 0:1] + self.mu) * xh + S[:, 1:2] * yh + S[:, 2:3] * zh)
        return r - m.r_site

    def h(self, S, m):
        """Predicted [ra, dec] for states S (k, 6) -> (k, 2)."""
        rho = self._rho(S, m)
        ra = np.mod(np.arctan2(rho[:, 1], rho[:, 0]), 2 * np.pi)
        dec = np.arcsin(rho[:, 2] / np.linalg.norm(rho, axis=1))
        return np.stack([ra, dec], axis=1)

    def H(self, s, m):
        """Measurement Jacobian d[ra, dec]/d(state) (2, 6)."""
        rho = self._rho(s, m)
        C = LU_KM * np.stack(m.basis, axis=-1)                # (3, 3), columns x_hat, y_hat, z_hat
        H3 = radec_jacobian(rho)[0] @ C
        return np.hstack([H3, np.zeros((2, 3))])

    @staticmethod
    def residual(z, zp):
        """z - zp with the RA difference wrapped to (-pi, pi]."""
        d = np.asarray(z, float) - np.asarray(zp, float)
        d[..., 0] = wrap_pi(d[..., 0])
        return d


def process_noise(dt, mu=None, q_psd_km2_s3=0.0):
    """Discrete process noise for continuous white acceleration of PSD q [km^2/s^3],
    in non-dimensional units, over dt [TU]."""
    if q_psd_km2_s3 <= 0.0 or dt == 0.0:
        return np.zeros((6, 6))
    q = q_psd_km2_s3 * TU_S ** 3 / LU_KM ** 2
    I3 = np.eye(3)
    return q * np.block([[dt ** 3 / 3 * I3, dt ** 2 / 2 * I3], [dt ** 2 / 2 * I3, dt * I3]])


def nees(err, P):
    """Normalised estimation error squared e^T P^-1 e."""
    return float(err @ np.linalg.solve(P, err))


def to_physical_sigma(P):
    """(max position sigma [km], max velocity sigma [m/s]) from a nondimensional covariance."""
    sp = np.sqrt(max(np.linalg.eigvalsh(P[:3, :3]).max(), 0.0)) * LU_KM
    sv = np.sqrt(max(np.linalg.eigvalsh(P[3:, 3:]).max(), 0.0)) * VU_KMS * 1e3
    return sp, sv
