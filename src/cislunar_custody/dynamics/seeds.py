"""Initial guesses for periodic-orbit families.

richardson_halo: third-order analytical halo approximation (Richardson 1980), in the
form given by Koon, Lo, Marsden & Ross, "Dynamical Systems, the Three-Body Problem and
Space Mission Design" (2011). Both L1 and L2 use the synodic x-orientation, so
X = x_L + gamma * x_R.
"""
from __future__ import annotations

import numpy as np

from .cr3bp import lagrange_points


def _gamma_c(mu, L):
    xl = lagrange_points(mu)[f"L{L}"][0]
    gamma = abs(xl - (1.0 - mu))
    if L == 1:
        c = lambda n: (mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 - gamma) ** (n + 1)) / gamma ** 3
    elif L == 2:
        c = lambda n: ((-1) ** n * mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 + gamma) ** (n + 1)) / gamma ** 3
    else:
        raise ValueError("L must be 1 or 2")
    return xl, gamma, c(2), c(3), c(4)


def _lam_k(c2):
    lam = np.sqrt((2 - c2 + np.sqrt((c2 - 2) ** 2 + 4 * (c2 - 1) * (1 + 2 * c2))) / 2)
    k = (lam ** 2 + 1 + 2 * c2) / (2 * lam)
    return lam, k


def lyapunov_seed(mu, Ax, L=1):
    """Linear planar Lyapunov guess: x = xL - Ax cos(lam t), y = k Ax sin(lam t).
    Ax in LU. Returns (state0, period_guess)."""
    xl, _, c2, _, _ = _gamma_c(mu, L)
    lam, k = _lam_k(c2)
    return np.array([xl - Ax, 0, 0, 0, k * Ax * lam, 0]), 2 * np.pi / lam


def dro_seed(mu, d):
    """Distant retrograde orbit guess: far-side crossing at distance d [LU] from the Moon,
    two-body retrograde speed minus frame rotation. Returns (state0, period_guess)."""
    vy = -(np.sqrt(mu / d) + d)
    return np.array([1 - mu + d, 0, 0, 0, vy, 0]), 2 * np.pi * np.sqrt(d ** 3 / mu)


def richardson_halo(mu, Az, L=1, northern=True):
    """Third-order Richardson halo guess. Az: out-of-plane amplitude [LU].
    Returns (state0, period_guess) with the orbit on y = 0, vx = vz = 0."""
    xl, g, c2, c3, c4 = _gamma_c(mu, L)
    lam, k = _lam_k(c2)
    dm = 1.0 if northern else -1.0
    d1 = 3 * lam ** 2 / k * (k * (6 * lam ** 2 - 1) - 2 * lam)
    d2 = 8 * lam ** 2 / k * (k * (11 * lam ** 2 - 1) - 2 * lam)
    a21 = 3 * c3 * (k ** 2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam / (4 * k * d1) * (3 * k ** 3 * lam - 6 * k * (k - lam) + 4)
    a24 = -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam)
    b21 = -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4)
    b22 = 3 * c3 * lam / d1
    d21 = -c3 / (2 * lam ** 2)
    a31 = (-9 * lam / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2))
           + (9 * lam ** 2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k ** 2)))
    a32 = -1 / d2 * (9 * lam / 4 * (4 * c3 * (k * a24 - b22) + k * c4)
                     + 1.5 * (9 * lam ** 2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4))
    b31 = 3 / (8 * d2) * (8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k ** 2))
                          + (9 * lam ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)))
    b32 = 1 / d2 * (9 * lam * (c3 * (k * b22 + d21 - 2 * a24) - c4)
                    + 3 / 8 * (9 * lam ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4))
    d31 = 3 / (64 * lam ** 2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam ** 2) * (4 * c3 * (a23 - d21) + c4 * (4 + k ** 2))
    den = 2 * lam * (lam * (1 + k ** 2) - 2 * k)
    s1 = (1.5 * c3 * (2 * a21 * (k ** 2 - 2) - a23 * (k ** 2 + 2) - 2 * k * b21)
          - 3 / 8 * c4 * (3 * k ** 4 - 8 * k ** 2 + 8)) / den
    s2 = (1.5 * c3 * (2 * a22 * (k ** 2 - 2) + a24 * (k ** 2 + 2) + 2 * k * b22 + 5 * d21)
          + 3 / 8 * c4 * (12 - k ** 2)) / den
    l1 = -1.5 * c3 * (2 * a21 + a23 + 5 * d21) - 3 / 8 * c4 * (12 - k ** 2) + 2 * lam ** 2 * s1
    l2 = 1.5 * c3 * (a24 - 2 * a22) + 9 / 8 * c4 + 2 * lam ** 2 * s2
    delta = lam ** 2 - c2

    az = Az / g
    ax2 = (-delta - l2 * az ** 2) / l1
    if ax2 <= 0:
        raise ValueError("Az below the halo bifurcation amplitude")
    ax = np.sqrt(ax2)
    om = 1 + s1 * ax ** 2 + s2 * az ** 2
    # tau1 = 0: cos = 1, sin = 0
    x = a21 * ax ** 2 + a22 * az ** 2 - ax + (a23 * ax ** 2 - a24 * az ** 2) + (a31 * ax ** 3 - a32 * ax * az ** 2)
    z = dm * az + dm * d21 * ax * az * (1 - 3) + dm * (d32 * az * ax ** 2 - d31 * az ** 3)
    vy = lam * om * (k * ax + 2 * (b21 * ax ** 2 - b22 * az ** 2) + 3 * (b31 * ax ** 3 - b32 * ax * az ** 2))
    state = np.array([xl + g * x, 0.0, g * z, 0.0, g * vy, 0.0])
    return state, 2 * np.pi / (lam * om)
