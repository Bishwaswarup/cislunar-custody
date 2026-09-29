"""Mapping between the CR3BP synodic frame and geocentric inertial coordinates.

v1 'hybrid' model: the CR3BP trajectory is placed in the instantaneous Earth-Moon
frame of the real (analytic) Moon: x_hat along Earth->Moon, z_hat along the Moon's
orbital angular momentum, lengths scaled by the constant LU. The Moon used for all
geometry (exclusion angle, occultation, shadow, sky brightness) is the MAPPED CR3BP
Moon, LU * x_hat, so target-Moon geometry is exactly the CR3BP one. Truth and filter
share this mapping, so it is self-consistent; the v2 study checks it against DE440.
"""
import numpy as np

from ..constants import LU_KM
from .ephemeris import moon_position, moon_velocity, sun_position
from .earth import gmst, rotate_z


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def synodic_basis(jd):
    """Unit vectors (x_hat, y_hat, z_hat) of the synodic frame, each (N, 3)."""
    jd = np.atleast_1d(np.asarray(jd, float))
    r, v = moon_position(jd), moon_velocity(jd)
    xh = _unit(r)
    zh = _unit(np.cross(r, v))
    yh = np.cross(zh, xh)
    return xh, yh, zh


def synodic_to_eci(r_syn, jd, mu, basis=None):
    """CR3BP positions (N, 3+) [LU] at Julian dates jd (N,) -> geocentric inertial [km]."""
    r_syn = np.atleast_2d(r_syn)
    xh, yh, zh = synodic_basis(jd) if basis is None else basis
    return LU_KM * ((r_syn[:, 0:1] + mu) * xh + r_syn[:, 1:2] * yh + r_syn[:, 2:3] * zh)


class Ephemeris:
    """Everything that depends only on time, precomputed once for a JD grid."""

    def __init__(self, jd):
        self.jd = np.atleast_1d(np.asarray(jd, float))
        self.sun = sun_position(self.jd)
        self.basis = synodic_basis(self.jd)
        self.moon = LU_KM * self.basis[0]      # mapped CR3BP Moon
        self.theta = gmst(self.jd)

    def to_eci(self, r_syn, mu):
        return synodic_to_eci(r_syn, self.jd, mu, self.basis)

    def site_eci(self, site):
        """(position [km], local-up unit vector), each (N, 3)."""
        return rotate_z(site.ecef, self.theta), rotate_z(site.up, self.theta)
