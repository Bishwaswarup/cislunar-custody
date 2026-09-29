"""Tracking scenarios: truth-consistent angles-only measurements and filter runs.

Time convention: t [TU] since `jd_epoch`, which is also the orbit-phase reference
(orbit.states_at(t)), exactly as in the visibility and observability studies.
"""
import numpy as np

from .constants import MU_EM
from .frames import Ephemeris
from .filters import Measurement, nees
from .sensors import SITES, NETWORKS, TELESCOPES, Target, network_visibility, simulate_radec
from .timeutil import jd_grid, tu_from_jd


def build_measurements(orbit, jd_epoch, start_day, days, *, network="Tri-3+S", telescope="1m",
                       moon_excl_deg=5.0, sigma_arcsec=1.0, step_min=10.0, target=Target(),
                       rng=None, mu=MU_EM):
    """RA/Dec from every site and time step at which the target is visible.
    Returns (measurements sorted by time, t_start, t_end) with times in TU."""
    rng = np.random.default_rng() if rng is None else rng
    jd = jd_grid(jd_epoch + start_day, days, step_min)
    eph = Ephemeris(jd)
    t = tu_from_jd(jd, jd_epoch)
    r_eci = eph.to_eci(orbit.states_at(t), mu)
    per_site, _ = network_visibility(eph, r_eci, NETWORKS[network], SITES, TELESCOPES[telescope],
                                     target, moon_excl_deg=moon_excl_deg)
    meas = []
    for k, v in per_site.items():
        idx = np.where(v.visible)[0]
        if idx.size == 0:
            continue
        r_site = eph.site_eci(SITES[k])[0][idx]
        ra, dec = simulate_radec(r_eci[idx], r_site, sigma_arcsec, rng)
        for j, i in enumerate(idx):
            meas.append(Measurement(float(t[i]), k, r_site[j], tuple(b[i] for b in eph.basis),
                                    np.array([ra[j], dec[j]])))
    meas.sort(key=lambda m: (m.t, m.site))
    return meas, float(t[0]), float(t[-1])


def run_filter(filt, x0, P0, t0, meas, orbit, t_end=None, keep_first_prior=False):
    """Process all measurements with any filter exposing init / predict_belief /
    update_belief / moments / size. One history entry per measurement epoch (after all
    updates at that time) plus a final prediction to t_end. Returns a dict with t, x,
    P, err (vs truth), nees, n_comp and 'diverged'. With keep_first_prior=True the
    predicted belief just before the first update is returned as 'first_prior'."""
    b, tc = filt.init(x0, P0), t0
    T, X, Ps, K = [], [], [], []
    first_prior = None
    diverged = False
    try:
        for m in meas:
            b = filt.predict_belief(b, m.t - tc)
            if keep_first_prior and first_prior is None:
                first_prior = b
            tc = m.t
            b = filt.update_belief(b, m)
            x, P = filt.moments(b)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
                raise FloatingPointError("non-finite state")
            if T and T[-1] == tc:
                X[-1], Ps[-1], K[-1] = x, P, filt.size(b)
            else:
                T.append(tc)
                X.append(x)
                Ps.append(P)
                K.append(filt.size(b))
        if t_end is not None and t_end > tc:
            b = filt.predict_belief(b, t_end - tc)
            x, P = filt.moments(b)
            T.append(t_end)
            X.append(x)
            Ps.append(P)
            K.append(filt.size(b))
    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError):
        diverged = True
    if not T:
        return {"t": np.array([]), "diverged": True, "first_prior": first_prior}
    T, X, Ps = np.array(T), np.array(X), np.array(Ps)
    err = X - orbit.states_at(T)
    return {"t": T, "x": X, "P": Ps, "err": err, "n_comp": np.array(K),
            "nees": np.array([nees(e, p) for e, p in zip(err, Ps)]), "diverged": diverged,
            "first_prior": first_prior}
