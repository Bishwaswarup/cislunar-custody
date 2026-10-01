"""Fisher information / Cramer-Rao bound for angles-only tracking arcs.

For measurements z_k = h(x(t_k)) + v_k (RA, Dec) with noise that is isotropic ON THE SKY,
v_k ~ N(0, R_k), R_k = diag(sigma^2 / cos^2(dec_k), sigma^2) (as simulated by
sensors.simulate_radec), the information about the state at the arc start t0 is

    I(t0) = sum_k Phi(t_k, t0)^T H_k^T R_k^-1 H_k Phi(t_k, t0),

with H_k = d(ra, dec)/d(r_eci) . d(r_eci)/d(r_syn) (angles do not depend on the
instantaneous velocity). It is computed in whitened sky coordinates (ra cos(dec), dec):
the RA row of H_k is multiplied by cos(dec_k) and the weight is sigma^-2 for both rows. The CRLB at t0 is P0 = (I + I_prior)^-1 and at the arc end
P_end = Phi(t_end, t0) P0 Phi(t_end, t0)^T. State units inside: LU and LU/TU.
"""
from dataclasses import dataclass

import numpy as np

from ..constants import LU_KM, VU_KMS
from ..dynamics.cr3bp import propagate
from ..sensors.measurement import ARCSEC


def radec_jacobian(rho):
    """d(ra, dec)/d(rho) for topocentric vectors rho (N, 3) km -> (N, 2, 3) rad/km."""
    rho = np.atleast_2d(rho)
    x, y, z = rho[:, 0], rho[:, 1], rho[:, 2]
    rxy2 = x * x + y * y
    r2 = rxy2 + z * z
    rxy = np.sqrt(rxy2)
    J = np.zeros((len(rho), 2, 3))
    J[:, 0, 0] = -y / rxy2
    J[:, 0, 1] = x / rxy2
    J[:, 1, 0] = -x * z / (r2 * rxy)
    J[:, 1, 1] = -y * z / (r2 * rxy)
    J[:, 1, 2] = rxy / r2
    return J


def sky_jacobian(rho):
    """d(ra cos(dec), dec)/d(rho): radec_jacobian with the RA row scaled by cos(dec), so that both
    rows are on-sky angles and isotropic noise has covariance sigma^2 I. (N, 2, 3) rad/km."""
    rho = np.atleast_2d(rho)
    J = radec_jacobian(rho)
    J[:, 0, :] *= (np.hypot(rho[:, 0], rho[:, 1]) / np.linalg.norm(rho, axis=1))[:, None]
    return J


def eci_from_synodic_jacobian(basis):
    """d(r_eci)/d(r_syn) = LU [x_hat | y_hat | z_hat] -> (N, 3, 3) km/LU."""
    xh, yh, zh = basis
    return LU_KM * np.stack([xh, yh, zh], axis=-1)


def crlb(info, prior_sigma_pos_km=1e5, prior_sigma_vel_kms=1.0):
    """Covariance (I + I_prior)^-1; the weak diagonal prior only regularises."""
    d = np.r_[np.full(3, (prior_sigma_pos_km / LU_KM) ** -2),
              np.full(3, (prior_sigma_vel_kms / VU_KMS) ** -2)]
    return np.linalg.inv(info + np.diag(d))


@dataclass
class ArcResult:
    t_end: float                 # TU after t0
    n_meas: int
    n_epochs: int
    info: np.ndarray             # at t0, (6, 6)
    P_end: np.ndarray            # CRLB propagated to the arc end
    sigma_pos_km: np.ndarray     # sqrt(eig) of the position block at the end, descending
    sigma_vel_ms: np.ndarray
    weak_los_deg: float          # angle between the weakest position direction and the geocentric LOS
    observable: bool             # data, not the prior, bounds every direction


def arc_information(orbit, t0, t_meas, r_site, basis_meas, t_ends, basis_ends, *,
                    sigma_arcsec=1.0, mu=None, prior_sigma_pos_km=1e5, prior_sigma_vel_kms=1.0):
    """CRLB for arcs starting at t0 [TU] and ending at each of t_ends [TU after t0].

    t_meas: absolute measurement times [TU] (M,), r_site: site positions at those times
    (M, 3) km, basis_meas: synodic basis at those times, basis_ends: list of (xh, yh, zh)
    3-vectors at each arc end. A measurement is used by an arc if t_meas - t0 <= t_end.
    """
    mu = orbit.mu if mu is None else mu
    tau = np.asarray(t_meas, float) - t0
    t_ends = np.atleast_1d(np.asarray(t_ends, float))
    grid = np.unique(np.concatenate([[0.0], tau, t_ends]))
    sol = propagate(orbit.states_at(t0)[0], (0.0, grid[-1]), mu, with_stm=True, t_eval=grid)
    Y = sol.y.T
    S, Phi = Y[:, :6], Y[:, 6:].reshape(-1, 6, 6)

    k = np.searchsorted(grid, tau)
    Sk, Phik = S[k], Phi[k]
    xh, yh, zh = basis_meas
    r_eci = LU_KM * ((Sk[:, 0:1] + mu) * xh + Sk[:, 1:2] * yh + Sk[:, 2:3] * zh)
    H3 = sky_jacobian(r_eci - r_site) @ eci_from_synodic_jacobian(basis_meas)     # (M, 2, 3), on-sky
    H = np.concatenate([H3, np.zeros_like(H3)], axis=2)                            # (M, 2, 6)
    A = H @ Phik
    w = (sigma_arcsec * ARCSEC) ** -2
    contrib = np.einsum("mki,mkj->mij", A, A) * w                                  # (M, 6, 6)

    out = []
    for te, be in zip(t_ends, basis_ends):
        use = tau <= te + 1e-12
        info = contrib[use].sum(axis=0) if use.any() else np.zeros((6, 6))
        P0 = crlb(info, prior_sigma_pos_km, prior_sigma_vel_kms)
        Pe_idx = np.searchsorted(grid, te)
        F = Phi[Pe_idx]
        Pe = F @ P0 @ F.T
        ev, V = np.linalg.eigh(Pe[:3, :3])
        sp = np.sqrt(np.clip(ev[::-1], 0, None)) * LU_KM
        sv = np.sqrt(np.clip(np.linalg.eigvalsh(Pe[3:, 3:])[::-1], 0, None)) * VU_KMS * 1e3
        weak = V[:, -1]
        bx, by, bz = be
        weak_eci = weak[0] * bx + weak[1] * by + weak[2] * bz
        se = S[Pe_idx]
        los = (se[0] + mu) * bx + se[1] * by + se[2] * bz
        ang = np.rad2deg(np.arccos(min(1.0, abs(weak_eci @ los) / np.linalg.norm(los))))
        # 'observable' if the data shrink every position direction well below the prior
        P0_meas_pos = np.sqrt(np.linalg.eigvalsh(P0[:3, :3]).max()) * LU_KM
        out.append(ArcResult(float(te), int(use.sum()), int(len(np.unique(tau[use]))), info, Pe,
                             sp, sv, float(ang), bool(P0_meas_pos < 0.1 * prior_sigma_pos_km)))
    return out
