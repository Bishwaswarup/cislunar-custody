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


MAX_STRIP_DEG = 10.0     # longest strip an observer would lay out along the predicted arc
STRIP_TILT_DEG = 2.0     # tolerance on the strip orientation (UT axis vs true stretch direction)


def strip_pattern(fov_deg, n_fields, axis_ratio, max_len_deg=MAX_STRIP_DEG):
    """Operational search pattern: n_fields square fields of side fov_deg laid out as a
    rectangle of n_l x n_w fields whose aspect follows the predicted sky ellipse
    (axis_ratio = minor / major 1-sigma axis, <= 1), no longer than max_len_deg.
    Uses at most n_fields fields. Returns (full length, full width) in degrees."""
    n_w = int(max(1, min(n_fields, round(np.sqrt(n_fields * max(axis_ratio, 0.0))))))
    if max_len_deg is not None:
        n_w = max(n_w, int(np.ceil(n_fields * fov_deg / max_len_deg - 1e-9)))
    n_w = min(n_w, n_fields)
    n_l = max(1, n_fields // n_w)
    return n_l * fov_deg, n_w * fov_deg


def strip_coordinates(umc, u, e_major):
    """Spherical along-/cross-track angles [rad] of unit vectors umc about the great circle
    through u in direction e_major (no gnomonic distortion; valid far from u)."""
    n_gc = np.cross(u, e_major)
    n_gc /= np.linalg.norm(n_gc)
    along = np.arctan2(umc @ e_major, umc @ u)
    cross = np.arcsin(np.clip(umc @ n_gc, -1.0, 1.0))
    return along, cross


def strip_containment(along, cross, L_deg, W_deg, tilt_deg=STRIP_TILT_DEG):
    """Share of samples inside an L x W strip centred on the prediction, taking the WORST of
    orientations -tilt, 0, +tilt (the strip axis is only known to within tilt_deg)."""
    Lh, Wh = np.radians(L_deg) / 2, np.radians(W_deg) / 2
    worst = 1.0
    for th in ((0.0,) if not tilt_deg else (-tilt_deg, 0.0, tilt_deg)):
        c, s_ = np.cos(np.radians(th)), np.sin(np.radians(th))
        a = c * along + s_ * cross
        b = -s_ * along + c * cross
        worst = min(worst, float(np.mean((np.abs(a) <= Lh) & (np.abs(b) <= Wh))))
    return worst


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
                strips=None, keep=None, strip_max_len_deg=MAX_STRIP_DEG, strip_tilt_deg=STRIP_TILT_DEG,
                consider=None):
    """Propagate (x0, P0) over gap lengths dt_grid [TU] (starting with 0) from Julian
    date jd0. Returns a dict of arrays over dt_grid (angles in degrees).
    batch: integrate the Monte Carlo samples in chunks of this size. The samples share one
    adaptive step size, so one sample passing close to the Moon slows the whole set;
    chunking keeps large sample sets fast. None (default) integrates all samples at once.
    strips: optional list of (fov_deg, n_fields). For each, out[f"contain_strip_N{n}"] is the
    share of MC samples inside a strip_pattern centred on the UT-predicted direction and
    aligned with the UT sky ellipse (the operator's search); out["ut_axis_ratio"] is the
    minor/major ratio of that ellipse. Strips are at most strip_max_len_deg long, distances are
    measured along and across the great circle of the ellipse's major axis, and containment is the
    worst over orientation errors of +-strip_tilt_deg. out["strip_along99_deg"] and
    out["strip_cross99_deg"] give the 99th percentile of |along| and |cross| of the true samples.
    The other outputs do not depend on this option.
    keep: optional collection of dt_grid indices; with strips, out["_debug"][k] stores the sky samples
    and strip geometry at those times (diagnostics only; absent by default).
    consider: optional dict for a TRUTH that differs from the operator's model (milestone 9):
        "P_mc"   covariance of (truth - x0) used for the Monte Carlo samples, (6, 6), or (7, 7) when
                 the 7th component is an error dp in the solar-pressure area-to-mass ratio;
        "accel"  callable t [TU since the gap start] -> (3,) acceleration per unit dp (needed for 7x7);
        "mean"   optional mean of the sampled (truth - x0[, dp]) (default 0), e.g. for an SRP force that
                 the operator does not model at all (a fixed dp, so the truth drifts systematically).
    The linear and UT predictions still use P0 and the unperturbed CR3BP (the operator's view), so
    'claim' and 'actual' measure an operator who does not know about the extra error sources.
    None (default) gives exactly the original behaviour."""
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

    floor = (R_EARTH_KM / LU_KM, R_MOON_KM / LU_KM)      # samples that would impact the Moon/Earth
    if consider is None:
        S0 = x0 + rng.standard_normal((n_samples, 6)) @ np.linalg.cholesky(P0).T
        dp, accel = None, None
    else:
        Pm = np.asarray(consider["P_mc"], float)
        D0 = rng.standard_normal((n_samples, len(Pm))) @ _sqrt_psd(0.5 * (Pm + Pm.T)).T
        D0 = D0 + np.asarray(consider.get("mean", np.zeros(len(Pm))), float)
        S0 = x0 + D0[:, :6]
        dp, accel = (D0[:, 6], consider["accel"]) if len(Pm) == 7 else (None, None)
    step = n_samples if (batch is None or n_samples <= batch) else batch
    Ymc = np.concatenate([propagate_many_dense(S0[i:i + step], dt_grid, mu, rtol, atol, r_floor=floor,
                                               p=None if dp is None else dp[i:i + step], accel=accel)
                          for i in range(0, n_samples, step)], axis=1)                 # (M, n, 6)

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
        out["strip_along99_deg"] = np.zeros(M)
        out["strip_cross99_deg"] = np.zeros(M)
    if keep is not None:
        out["_debug"] = {}
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
                e_maj = V2[:, 1] @ E                                             # major axis, 3-D
                a, b = strip_coordinates(umc, u, e_maj)                          # along, cross [rad]
                out["strip_along99_deg"][k] = float(np.degrees(np.quantile(np.abs(a), 0.99)))
                out["strip_cross99_deg"][k] = float(np.degrees(np.quantile(np.abs(b), 0.99)))
                for fov, n in strips:
                    L_deg, W_deg = strip_pattern(fov, n, ratio, strip_max_len_deg)
                    out[f"contain_strip_N{n}"][k] = strip_containment(a, b, L_deg, W_deg, strip_tilt_deg)
                if keep is not None and k in keep:
                    out["_debug"][k] = dict(umc=umc.copy(), u=u, E=E, V2=V2, w2=w2, ratio=ratio, C_sky=C_sky,
                                            e_maj=e_maj, a=a, b=b)
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
