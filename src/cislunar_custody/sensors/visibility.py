"""Visibility of a cislunar target from ground optical sites.

A detection is possible at a time step only if ALL hold (each violation is a 'reason'):
    day        Sun above the astronomical-twilight limit (default -18 deg)
    low_elev   target elevation below the minimum (default 20 deg)
    occulted   line of sight passes through the Moon
    eclipsed   target in Earth's or the Moon's (cylindrical) shadow
    moon_excl  target-Moon separation below the exclusion angle (instrumental/baffling
               limit; swept in the study because cislunar L1/L2 targets sit only
               ~4-10 deg from the Moon as seen from Earth)
    faint      extinction-corrected magnitude fainter than the moonlight-degraded limit
"""
from dataclasses import dataclass

import numpy as np

from ..constants import R_EARTH_KM, R_MOON_KM
from .photometry import (apparent_magnitude, airmass, ks_moon_sky_nl, limiting_magnitude,
                         lunar_phase_angle_deg, _angle, _unit)

REASONS = ("day", "low_elev", "occulted", "eclipsed", "moon_excl", "faint")


def is_shadowed(r, r_body, R_body, r_sun):
    """Cylindrical shadow of a spherical body. r, r_body, r_sun: (N, 3) km."""
    s_hat = _unit(r_sun - r_body)
    rel = r - r_body
    along = np.sum(rel * s_hat, axis=-1)
    perp = np.linalg.norm(rel - along[..., None] * s_hat, axis=-1)
    return (along < 0.0) & (perp < R_body)


@dataclass
class SiteVisibility:
    site: str
    visible: np.ndarray
    blocked: dict              # reason -> bool array (True = violated)
    elev_deg: np.ndarray
    moon_sep_deg: np.ndarray
    mag: np.ndarray            # extinction-corrected apparent magnitude
    m_lim: np.ndarray          # effective limiting magnitude

    def first_reason(self):
        """Integer code per step: 0 visible, i+1 = first violated reason in REASONS."""
        code = np.zeros(self.visible.shape, int)
        for i, r in reversed(list(enumerate(REASONS))):
            code[self.blocked[r]] = i + 1
        return code


def site_visibility(eph, r_tgt, site_key, site, telescope, target, *, moon_excl_deg=10.0,
                    min_elev_deg=20.0, sun_elev_max_deg=-18.0):
    """eph: frames.Ephemeris for the time grid; r_tgt: geocentric target positions (N, 3) km."""
    r_site, up = eph.site_eci(site)
    rho = r_tgt - r_site
    elev = 90.0 - np.rad2deg(_angle(rho, up))
    sun_elev = 90.0 - np.rad2deg(_angle(eph.sun - r_site, up))
    moon_rel = eph.moon - r_site
    moon_elev = 90.0 - np.rad2deg(_angle(moon_rel, up))
    sep = np.rad2deg(_angle(rho, moon_rel))

    rho_n = np.linalg.norm(rho, axis=-1)
    t_close = np.sum(moon_rel * (rho / rho_n[:, None]), axis=-1)
    d_close = np.linalg.norm(moon_rel - t_close[:, None] * rho / rho_n[:, None], axis=-1)
    occulted = (t_close > 0.0) & (t_close < rho_n) & (d_close < R_MOON_KM)

    zeros = np.zeros_like(r_tgt)
    eclipsed = (is_shadowed(r_tgt, zeros, R_EARTH_KM, eph.sun)
                | is_shadowed(r_tgt, eph.moon, R_MOON_KM, eph.sun))

    z = 90.0 - elev
    alpha = lunar_phase_angle_deg(eph.moon, eph.sun)
    B_moon = ks_moon_sky_nl(alpha, sep, 90.0 - moon_elev, z, site.extinction)
    m_lim = limiting_magnitude(telescope.m_lim, B_moon, site.sky_dark_mag, site.extinction, z)
    mag = (apparent_magnitude(r_tgt, r_site, eph.sun, target.radius_m, target.albedo)
           + site.extinction * (airmass(z) - 1.0))

    blocked = {
        "day": sun_elev > sun_elev_max_deg,
        "low_elev": elev < min_elev_deg,
        "occulted": occulted,
        "eclipsed": eclipsed,
        "moon_excl": sep < moon_excl_deg,
        "faint": mag > m_lim,
    }
    visible = ~np.any(np.stack([blocked[r] for r in REASONS]), axis=0)
    return SiteVisibility(site_key, visible, blocked, elev, sep, mag, m_lim)


def network_visibility(eph, r_tgt, site_keys, sites, telescope, target, **kw):
    """Per-site results and the network union (visible from at least one site)."""
    per_site = {k: site_visibility(eph, r_tgt, k, sites[k], telescope, target, **kw)
                for k in site_keys}
    union = np.any(np.stack([v.visible for v in per_site.values()]), axis=0)
    return per_site, union


def runs(mask):
    """Lengths and start indices of consecutive True runs in a boolean array."""
    m = np.concatenate([[0], np.asarray(mask, int), [0]])
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return ends - starts, starts


def gap_stats(visible, dt_days):
    """Observation-gap statistics [days] for a visibility time series. Gaps touching
    the ends of the series are censored and excluded; a target never visible has no
    finite gaps and gets NaN statistics."""
    visible = np.asarray(visible, bool)
    lens, starts = runs(~visible)
    keep = (starts > 0) & (starts + lens < len(visible))
    gaps = lens[keep] * dt_days
    stat = (lambda f: float(f(gaps))) if gaps.size else (lambda f: float("nan"))
    return {
        "frac_visible": float(np.mean(visible)),
        "n_gaps": int(keep.sum()),
        "median_gap_d": stat(np.median),
        "p90_gap_d": stat(lambda g: np.percentile(g, 90)),
        "max_gap_d": stat(np.max),
        "gaps_d": gaps,
    }
