"""Instantaneous (pulsating, rotating) Earth-Moon frame mapping between CR3BP states and
geocentric inertial states, using the DE440 Moon (standard instantaneous-frame mapping):

    r = d C rho,   v = d_dot C rho + d w (z_hat x C rho) + d w C rho',
with rho = (x + mu, y, z), rho' = CR3BP velocity, d = |r_moon|, w = |h| / d^2,
C = [x_hat y_hat z_hat], x_hat along r_moon, z_hat along h = r_moon x v_moon.
Inertial states are returned in LU and LU/TU (divide km by LU, km/s by LU/TU).
"""
import numpy as np

from ..constants import LU_KM, TU_S


def _frame(de, jd):
    rm, vm = de.moon_state(jd)
    d = np.linalg.norm(rm)
    xh = rm / d
    h = np.cross(rm, vm)
    zh = h / np.linalg.norm(h)
    yh = np.cross(zh, xh)
    C = np.stack([xh, yh, zh], axis=1)
    w = np.linalg.norm(h) / d ** 2
    ddot = rm @ vm / d
    return C, d, ddot, w, zh


def synodic_to_inertial(s, jd, de, mu):
    s = np.asarray(s, float)
    C, d, ddot, w, zh = _frame(de, jd)
    rho = np.array([s[0] + mu, s[1], s[2]])
    Crho = C @ rho
    r = d * Crho
    v = ddot * Crho + d * w * np.cross(zh, Crho) + d * w * (C @ s[3:6])
    return np.concatenate([r / LU_KM, v / (LU_KM / TU_S)])


def inertial_to_synodic(x, jd, de, mu):
    x = np.asarray(x, float)
    C, d, ddot, w, zh = _frame(de, jd)
    r, v = x[:3] * LU_KM, x[3:] * (LU_KM / TU_S)
    Crho = r / d
    rho = C.T @ Crho
    rho_p = C.T @ (v - ddot * Crho - d * w * np.cross(zh, Crho)) / (d * w)
    return np.array([rho[0] - mu, rho[1], rho[2], *rho_p])


def transform_jacobian(s, jd, de, mu, h=1e-7):
    """d(inertial)/d(synodic) by central differences (6x6)."""
    J = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        J[:, i] = (synodic_to_inertial(s + e, jd, de, mu) - synodic_to_inertial(s - e, jd, de, mu)) / (2 * h)
    return J
