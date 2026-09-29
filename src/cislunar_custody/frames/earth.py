"""Earth rotation (GMST, IAU 1982) and WGS-84 geodetic sites.
Precession/nutation/polar motion are ignored: the inertial frame is 'of date'."""
import numpy as np

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563


def gmst(jd_ut1):
    """Greenwich mean sidereal angle [rad] (IAU 1982)."""
    Tu = (np.asarray(jd_ut1, float) - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * Tu
           + 0.093104 * Tu ** 2 - 6.2e-6 * Tu ** 3)
    return np.deg2rad(np.mod(sec, 86400.0) / 240.0)


def geodetic_to_ecef(lat_deg, lon_deg, h_km):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    N = WGS84_A_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array([(N + h_km) * np.cos(lat) * np.cos(lon),
                     (N + h_km) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - e2) + h_km) * np.sin(lat)])


def geodetic_up(lat_deg, lon_deg):
    """Local vertical (ellipsoid normal) in ECEF."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def rotate_z(v, theta):
    """Rotate fixed 3-vector v about z by angles theta (N,) -> (N, 3)."""
    theta = np.atleast_1d(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[0] - s * v[1], s * v[0] + c * v[1], np.full_like(theta, v[2])], axis=-1)
