"""Gap propagation and custody-horizon metrics.

Given a post-fit estimate (x0, P0) at the start of an observation gap, the uncertainty
is propagated three ways over a grid of gap lengths:
    lin  linear covariance Phi P0 Phi^T (what an EKF predicts)
    ut   unscented transform (what a UKF predicts)
    mc   Monte Carlo samples of N(x0, P0) through the full CR3BP (the reference)
and projected onto the geocentric sky. With a search pattern of angular radius
r_search (N fields of view), the custody horizon is the longest gap for which 99% of
the true (MC) probability mass is still inside the searched circle:
    ideal   circle centred on the true (MC) mean direction
    actual  circle centred on the method's own predicted direction (the operational case)
    claim   when the method's OWN 99% region outgrows r_search (what the filter believes)
'claim > actual' is false custody: the filter believes the target is still within the
search pattern after it has in fact left it.
FTLE predictor: T_ftle = first gap for which s_max(Phi) >= r_search / theta0, where
theta0 is the initial 99% angular radius and s_max the largest singular value of the STM.
"""
import numpy as np
from scipy.stats import chi2

from ..constants import LU_KM, TU_S, DAY_S, R_EARTH_KM, R_MOON_KM
from ..dynamics.cr3bp import propagate, propagate_many_dense
from ..filters.ukf import UKF, _sqrt_psd
from ..frames import Ephemeris

CHI2_2_99 = chi2.ppf(0.99, 2)
_Z = np.random.default_rng(12345).standard_normal((20000, 2))


def search_radius_deg(fov_deg, n_fields):
    """Radius of the circle with the same area as n_fields square fields of view."""
    return fov_deg * np.sqrt(n_fields / np.pi)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _tangent_basis(u):
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, u)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(u, e1)])


def _eci(S, basis_k, mu):
    xh, yh, zh = basis_k
    S = np.atleast_2d(S)
    return LU_KM * ((S[:, 0:1] + mu) * xh + S[:, 1:2] * yh + S[:, 2:3] * zh)


def _mahalanobis2(p, C):
    """Squared Mahalanobis distances of rows of p under C. A blown-up gap covariance can
    be so elongated that C is numerically rank-1, so eigenvalues are floored relative to
    the largest instead of calling solve()."""
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.maximum(w, max(w[-1], 1e-300) * 1e-12)
    return np.sum((p @ V) ** 2 / w, axis=1)


def strip_pattern(fov_deg, n_fields, axis_ratio):
    """Operational search pattern: n_fields square fields of side fov_deg laid out as a
    rectangle of n_l x n_w fields whose aspect follows the predicted sky ellipse
    (axis_ratio = minor / major 1-sigma axis, <= 1). Uses at most n_fields fields.
    Returns (full length, full width) in degrees."""
    n_w = int(max(1, min(n_fields, round(np.sqrt(n_fields * max(axis_ratio, 0.0))))))
    n_l = max(1, n_fields // n_w)
    return n_l * fov_deg, n_w * fov_deg


def containment_horizon(t, frac, level=0.99):
    """First t at which the contained fraction drops below level (linear interpolation
    in log t). 0 if already below at t[0]; inf if never within the grid."""
    t, frac = np.asarray(t, float), np.asarray(frac, float)
    below = np.where(frac < level)[0]
    if below.size == 0:
        return np.inf
    i = below[0]
    if i == 0:
        return 0.0
    f0, f1 = frac[i - 1], frac[i]
    w = (f0 - level) / (f0 - f1) if f0 > f1 else 1.0
    t0, t1 = max(t[i - 1], 1e-12), t[i]
    return float(np.exp(np.log(t0) + w * (np.log(t1) - np.log(t0))))


def predict_gap(x0, P0, dt_grid, jd0, mu, n_samples=500, rng=None, rtol=1e-9, atol=1e-11, batch=None,
                strips=None):
    """Propagate (x0, P0) over gap lengths dt_grid [TU] (starting with 0) from Julian
    date jd0. Returns a dict of arrays over dt_grid (angles in degrees).
    batch: integrate the Monte Carlo samples in chunks of this size. The samples share one
    adaptive step size, so one sample passing close to the Moon slows the whole set;
    chunking keeps large sample sets fast. None (default) integrates all samples at once.
    strips: optional list of (fov_deg, n_fields). For each, out[f"contain_strip_N{n}"] is the
    share of MC samples inside a strip_pattern centred on the UT-predicted direction and
    aligned with the UT sky ellipse (the operator's search); out["ut_axis_ratio"] is the
    minor/major ratio of that ellipse. The other outputs do not depend on this option."""
    rng = np.random.default_rng() if rng is None else rng
    dt_grid = np.asarray(dt_grid, float)
    M = len(dt_grid)
    sol = propagate(x0, (0.0, dt_grid[-1]), mu, with_stm=True, t_eval=dt_grid, rtol=1e-11, atol=1e-12)
    Xn = sol.y[:6].T
    Phi = sol.y[6:].T.reshape(M, 6, 6)
    P_lin = np.einsum("kij,jl,kml->kim", Phi, P0, Phi)
    smax = np.linalg.svd(Phi, compute_uv=False)[:, 0]

    ukf = UKF(mu, model=None)
    Ys = propagate_many_dense(ukf.sigma_points(x0, P0), dt_grid, mu, rtol, atol)     # (M, 13, 6)
    m_ut = np.einsum("s,ksj->kj", ukf.Wm, Ys)
    D = Ys - m_ut[:, None, :]
    P_ut = np.einsum("s,ksi,ksj->kij", ukf.Wc, D, D)

    S0 = x0 + rng.standard_normal((n_samples, 6)) @ np.linalg.cholesky(P0).T
    floor = (R_EARTH_KM / LU_KM, R_MOON_KM / LU_KM)      # samples that would impact the Moon/Earth
    if batch is None or n_samples <= batch:
        Ymc = propagate_many_dense(S0, dt_grid, mu, rtol, atol, r_floor=floor)        # (M, n, 6)
    else:
        Ymc = np.concatenate([propagate_many_dense(S0[i:i + batch], dt_grid, mu, rtol, atol, r_floor=floor)
                              for i in range(0, n_samples, batch)], axis=1)

    eph = Ephemeris(jd0 + dt_grid * TU_S / DAY_S)
    out = {k: np.zeros(M) for k in ("theta_ideal", "theta_lin_claim", "theta_lin_actual",
                                    "contain_lin", "theta_ut_claim", "theta_ut_actual", "contain_ut")}
    out["dt"] = dt_grid
    out["smax"] = smax
    strips = list(strips or [])
    for fov, n in strips:
        out[f"contain_strip_N{n}"] = np.zeros(M)
    if strips:
        out["ut_axis_ratio"] = np.zeros(M)
    for k in range(M):
        bk = tuple(b[k] for b in eph.basis)
        B = LU_KM * np.stack(bk, axis=-1)
        umc = _unit(_eci(Ymc[k], bk, mu))
        c = _unit(umc.mean(axis=0))
        out["theta_ideal"][k] = np.degrees(np.quantile(np.arccos(np.clip(umc @ c, -1, 1)), 0.99))
        for tag, m, P in (("lin", Xn[k], P_lin[k]), ("ut", m_ut[k], P_ut[k])):
            r = _eci(m, bk, mu)[0]
            u = r / np.linalg.norm(r)
            E = _tangent_basis(u)
            C_sky = E @ (B @ P[:3, :3] @ B.T) @ E.T / (r @ r)
            L = _sqrt_psd(0.5 * (C_sky + C_sky.T))
            out[f"theta_{tag}_claim"][k] = np.degrees(np.quantile(np.linalg.norm(_Z @ L.T, axis=1), 0.99))
            out[f"theta_{tag}_actual"][k] = np.degrees(np.quantile(np.arccos(np.clip(umc @ u, -1, 1)), 0.99))
            out[f"contain_{tag}"][k] = np.mean(_mahalanobis2(umc @ E.T, C_sky) < CHI2_2_99)
            if tag == "ut" and strips:
                w2, V2 = np.linalg.eigh(0.5 * (C_sky + C_sky.T))
                ratio = float(np.sqrt(max(w2[0], 0.0) / max(w2[1], 1e-300)))
                out["ut_axis_ratio"][k] = ratio
                cosu = umc @ u
                q = (umc @ E.T) / np.where(cosu > 0, cosu, np.nan)[:, None]      # gnomonic [rad]
                a, b = q @ V2[:, 1], q @ V2[:, 0]                                # major, minor axes
                for fov, n in strips:
                    L_deg, W_deg = strip_pattern(fov, n, ratio)
                    inside = (np.abs(a) <= np.radians(L_deg) / 2) & (np.abs(b) <= np.radians(W_deg) / 2)
                    out[f"contain_strip_N{n}"][k] = float(np.mean(np.where(cosu > 0, inside, False)))
    return out


def crossing_time(t, y, threshold):
    """First t at which y exceeds threshold (log-interpolated). 0 if already above at
    t[0]; inf if never within the grid (censored)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    above = np.where(y > threshold)[0]
    if above.size == 0:
        return np.inf
    i = above[0]
    if i == 0:
        return 0.0
    y0, y1 = max(y[i - 1], 1e-300), max(y[i], 1e-300)
    t0, t1 = max(t[i - 1], 1e-12), t[i]
    f = (np.log(threshold) - np.log(y0)) / (np.log(y1) - np.log(y0)) if y1 > y0 else 1.0
    return float(np.exp(np.log(t0) + f * (np.log(t1) - np.log(t0))))


def custody_horizons(pred, r_search_deg):
    """Custody horizons [same units as pred['dt']] for one search radius."""
    t = pred["dt"]
    theta0 = max(pred["theta_ideal"][0], 1e-12)
    return {
        "ideal": crossing_time(t, pred["theta_ideal"], r_search_deg),
        "lin_claim": crossing_time(t, pred["theta_lin_claim"], r_search_deg),
        "lin_actual": crossing_time(t, pred["theta_lin_actual"], r_search_deg),
        "ut_claim": crossing_time(t, pred["theta_ut_claim"], r_search_deg),
        "ut_actual": crossing_time(t, pred["theta_ut_actual"], r_search_deg),
        "ftle": crossing_time(t, pred["smax"], r_search_deg / theta0),
    }


def strip_horizon(pred, n_fields, level=0.99):
    """Operator horizon for the elongated, UT-centred strip search of n_fields fields."""
    return containment_horizon(pred["dt"], pred[f"contain_strip_N{n_fields}"], level)
