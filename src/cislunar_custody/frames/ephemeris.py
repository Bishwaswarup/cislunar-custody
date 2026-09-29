"""Low-precision analytic Sun and Moon positions (geocentric, equator/equinox of date).

Sun:  Astronomical Almanac low-precision formulae (~0.01 deg, 1800-2200).
Moon: Montenbruck & Gill, "Satellite Orbits" (2000), Sec. 3.3.2 (~0.1-0.2 deg,
      ~500 km in distance). Mean longitude taken of date (no -1.3972 T term).
Accurate enough for visibility geometry (Moon exclusion, night/day, shadow); the
v2 ephemeris study replaces these with DE440.
"""
import numpy as np

AU_KM = 149597870.7
_AS = np.pi / (180.0 * 3600.0)   # arcsec -> rad
_d = np.deg2rad


def _T(jd):
    return (np.asarray(jd, float) - 2451545.0) / 36525.0


def obliquity(jd):
    """Mean obliquity of the ecliptic [rad]."""
    return _d(23.439291 - 0.0130042 * _T(jd))


def _ecl_to_eq(lon, lat, r, jd):
    eps = obliquity(jd)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    ce, se = np.cos(eps), np.sin(eps)
    return np.stack([x, y * ce - z * se, y * se + z * ce], axis=-1)


def sun_position(jd):
    """Geocentric Sun position [km], shape (..., 3)."""
    n = np.asarray(jd, float) - 2451545.0
    L = _d(280.460 + 0.9856474 * n)
    g = _d(357.528 + 0.9856003 * n)
    lam = L + _d(1.915) * np.sin(g) + _d(0.020) * np.sin(2 * g)
    R = (1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)) * AU_KM
    return _ecl_to_eq(lam, np.zeros_like(lam), R, jd)


def moon_position(jd):
    """Geocentric Moon position [km], shape (..., 3)."""
    T = _T(jd)
    L0 = _d(218.31617 + 481267.88088 * T)
    l = _d(134.96292 + 477198.86753 * T)     # Moon mean anomaly
    lp = _d(357.52543 + 35999.04944 * T)     # Sun mean anomaly
    F = _d(93.27283 + 483202.01873 * T)      # argument of latitude
    D = _d(297.85027 + 445267.11135 * T)     # mean elongation
    s = np.sin
    dlam = (22640 * s(l) + 769 * s(2 * l) - 4586 * s(l - 2 * D) + 2370 * s(2 * D)
            - 668 * s(lp) - 412 * s(2 * F) - 212 * s(2 * l - 2 * D) - 206 * s(l + lp - 2 * D)
            + 192 * s(l + 2 * D) - 165 * s(lp - 2 * D) + 148 * s(l - lp) - 125 * s(D)
            - 110 * s(l + lp) - 55 * s(2 * F - 2 * D)) * _AS
    lam = L0 + dlam
    beta = (18520 * s(F + dlam + (412 * s(2 * F) + 541 * s(lp)) * _AS)
            - 526 * s(F - 2 * D) + 44 * s(l + F - 2 * D) - 31 * s(-l + F - 2 * D)
            - 25 * s(-2 * l + F) - 23 * s(lp + F - 2 * D) + 21 * s(-l + F)
            + 11 * s(-lp + F - 2 * D)) * _AS
    c = np.cos
    r = (385000.0 - 20905 * c(l) - 3699 * c(2 * D - l) - 2956 * c(2 * D) - 570 * c(2 * l)
         + 246 * c(2 * l - 2 * D) - 205 * c(lp - 2 * D) - 171 * c(l + 2 * D)
         - 152 * c(l + lp - 2 * D))
    return _ecl_to_eq(lam, beta, r, jd)


def moon_velocity(jd, h_days=1.0 / 24.0):
    """Geocentric Moon velocity [km/s] by central difference of moon_position."""
    jd = np.asarray(jd, float)
    return (moon_position(jd + h_days) - moon_position(jd - h_days)) / (2 * h_days * 86400.0)
