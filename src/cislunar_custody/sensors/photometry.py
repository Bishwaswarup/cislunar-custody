"""Target brightness and sky background.

Target: Lambertian sphere, m = m_sun - 2.5 log10[(2/3) A (R/rho)^2 F(phi) (AU/d_sun)^2],
        F(phi) = ((pi - phi) cos phi + sin phi) / pi, phi = solar phase angle.
Sky:    Krisciunas & Schaefer (1991, PASP 103, 1033) scattered-moonlight model (V band).
Limit:  background-limited detection: m_lim degrades by 1.25 log10(B_total / B_dark).
"""
import numpy as np

from ..frames.ephemeris import AU_KM

M_SUN_V = -26.74


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _angle(a, b):
    return np.arccos(np.clip(np.sum(_unit(a) * _unit(b), axis=-1), -1.0, 1.0))


def lambert_phase(phi):
    return ((np.pi - phi) * np.cos(phi) + np.sin(phi)) / np.pi


def apparent_magnitude(r_obj, r_obs, r_sun, radius_m, albedo):
    """Visual magnitude of a Lambertian sphere; positions in km (N, 3) or (3,)."""
    to_sun, to_obs = r_sun - r_obj, r_obs - r_obj
    phi = _angle(to_sun, to_obs)
    rho = np.linalg.norm(to_obs, axis=-1)
    dsun = np.linalg.norm(to_sun, axis=-1)
    flux = (2.0 / 3.0) * albedo * (radius_m * 1e-3 / rho) ** 2 * lambert_phase(phi) * (AU_KM / dsun) ** 2
    return M_SUN_V - 2.5 * np.log10(np.maximum(flux, 1e-300))


def airmass(z_deg):
    """K&S airmass X = (1 - 0.96 sin^2 Z)^-1/2 (finite at the horizon)."""
    z = np.deg2rad(np.clip(z_deg, 0.0, 90.0))
    return 1.0 / np.sqrt(1.0 - 0.96 * np.sin(z) ** 2)


def mag_arcsec2_to_nl(V):
    """Surface brightness V mag/arcsec^2 -> nanoLamberts (K&S eq. 1)."""
    return 34.08 * np.exp(20.7233 - 0.92104 * np.asarray(V, float))


def lunar_phase_angle_deg(r_moon, r_sun, r_obs=None):
    """Angle at the Moon between the Sun and the observer (0 = full, 180 = new)."""
    obs = np.zeros(3) if r_obs is None else r_obs
    return np.rad2deg(_angle(r_sun - r_moon, obs - r_moon))


def ks_moon_sky_nl(alpha_deg, rho_deg, z_moon_deg, z_deg, k):
    """Scattered-moonlight sky brightness [nL] (0 when the Moon is below the horizon).
    alpha: lunar phase angle, rho: target-Moon separation, z_moon/z: zenith distances."""
    a = np.abs(np.asarray(alpha_deg, float))
    I = 10 ** (-0.4 * (3.84 + 0.026 * a + 4e-9 * a ** 4))
    rho = np.clip(np.asarray(rho_deg, float), 0.5, 180.0)   # model diverges as rho -> 0
    f = 10 ** 5.36 * (1.06 + np.cos(np.deg2rad(rho)) ** 2) + 10 ** (6.15 - rho / 40.0)
    B = f * I * 10 ** (-0.4 * k * airmass(z_moon_deg)) * (1.0 - 10 ** (-0.4 * k * airmass(z_deg)))
    return np.where(np.asarray(z_moon_deg) < 90.0, B, 0.0)


def limiting_magnitude(m_lim_zenith_dark, B_moon_nl, sky_dark_mag, k, z_deg):
    """Effective limiting magnitude under moonlight at zenith distance z (before target
    extinction). Dark sky brightens with airmass as X 10^(-0.4 k (X - 1)) (K&S eq. 2)."""
    X = airmass(z_deg)
    B_dark_zen = mag_arcsec2_to_nl(sky_dark_mag)
    B_dark = B_dark_zen * X * 10 ** (-0.4 * k * (X - 1.0))
    return m_lim_zenith_dark - 1.25 * np.log10((B_dark + B_moon_nl) / B_dark_zen)
