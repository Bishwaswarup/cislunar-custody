"""Topocentric right ascension / declination (angles-only) measurements."""
import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)


def radec(rho):
    """Topocentric line of sight rho (N, 3) -> (ra, dec) [rad], ra in [0, 2 pi)."""
    rho = np.atleast_2d(rho)
    ra = np.mod(np.arctan2(rho[:, 1], rho[:, 0]), 2 * np.pi)
    dec = np.arcsin(rho[:, 2] / np.linalg.norm(rho, axis=1))
    return ra, dec


def simulate_radec(r_tgt, r_site, sigma_arcsec, rng=None):
    """Noisy RA/Dec. The RA noise is sigma / cos(dec) so the on-sky error is isotropic."""
    rng = np.random.default_rng() if rng is None else rng
    ra, dec = radec(r_tgt - r_site)
    s = sigma_arcsec * ARCSEC
    return (np.mod(ra + rng.normal(0, s, ra.shape) / np.cos(dec), 2 * np.pi),
            dec + rng.normal(0, s, dec.shape))
