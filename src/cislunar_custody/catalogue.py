"""Orbit catalogue: builds and stores the Earth-Moon periodic-orbit families used in
the custody study (L1/L2 halo incl. NRHOs, L1/L2 Lyapunov, DRO) plus named orbits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import MU_EM, LU_KM, TU_DAYS, SYNODIC_MONTH_DAYS
from .dynamics import (PeriodicOrbit, correct, continue_family, find_by_period,
                       richardson_halo, lyapunov_seed, dro_seed)
from .dynamics.periodic import PLANAR_FREE, PLANAR_CONSTRAIN

_rp_below = lambda km: (lambda o: o.moon_distance_range(600)[0] * LU_KM < km)

# name -> builder settings
FAMILY_SPECS = {
    "L1_halo_north": dict(kind="halo", L=1, northern=True, Az_km=8000, n=200, ds_max=5e-3, stop_rp_km=1900),
    "L2_halo_south": dict(kind="halo", L=2, northern=False, Az_km=8000, n=200, ds_max=1.5e-2, stop_rp_km=1900),
    "L1_lyapunov": dict(kind="lyapunov", L=1, Ax=0.005, n=60, ds_max=1e-2),
    "L2_lyapunov": dict(kind="lyapunov", L=2, Ax=0.005, n=60, ds_max=1e-2),
    "DRO": dict(kind="dro", d=0.02, n=120, ds_max=1e-2, stop_x=1.45),
}


def build_family(name, mu=MU_EM, verbose=False):
    spec = FAMILY_SPECS[name]
    if spec["kind"] == "halo":
        s, _ = richardson_halo(mu, spec["Az_km"] / LU_KM, L=spec["L"], northern=spec["northern"])
        o = correct(s, mu, free=(0, 4), family=name)
        stop = _rp_below(spec["stop_rp_km"])
        fam = continue_family(o, spec["n"], ds=2e-3, ds_max=spec["ds_max"], direction=1.0,
                              stop=stop, family=name, verbose=verbose)
        if stop(fam[-1]):  # member that crossed the perilune limit is dropped
            fam = fam[:-1]
        return fam
    if spec["kind"] == "lyapunov":
        s, _ = lyapunov_seed(mu, spec["Ax"], L=spec["L"])
        o = correct(s, mu, free=(4,), constrain=PLANAR_CONSTRAIN, family=name)
        return continue_family(o, spec["n"], ds=2e-3, ds_max=spec["ds_max"], direction=1.0,
                               free=PLANAR_FREE, constrain=PLANAR_CONSTRAIN, family=name, verbose=verbose)
    if spec["kind"] == "dro":
        s, _ = dro_seed(mu, spec["d"])
        o = correct(s, mu, free=(4,), constrain=PLANAR_CONSTRAIN, family=name)
        return continue_family(o, spec["n"], ds=5e-3, ds_max=spec["ds_max"], direction=1.0,
                               free=PLANAR_FREE, constrain=PLANAR_CONSTRAIN, family=name,
                               stop=lambda orb: orb.state0[0] > spec["stop_x"], verbose=verbose)
    raise ValueError(spec["kind"])


def family_table(members):
    """Per-member arrays: states, periods, Jacobi, stability index, Moon distance range."""
    rng = np.array([m.moon_distance_range(1500) for m in members])
    return {
        "states": np.array([m.state0 for m in members]),
        "periods": np.array([m.period for m in members]),
        "jacobi": np.array([m.jacobi for m in members]),
        "stability": np.array([m.stability_index() for m in members]),
        "r_moon_min": rng[:, 0],
        "r_moon_max": rng[:, 1],
    }


def nrho_by_resonance(l2_south_members, n_rev, m_syn, mu=MU_EM):
    """NRHO completing n_rev revolutions per m_syn synodic months (e.g. 9:2)."""
    T = SYNODIC_MONTH_DAYS * m_syn / n_rev / TU_DAYS
    return find_by_period(l2_south_members, T, family=f"NRHO_{n_rev}:{m_syn}")


def save_catalogue(path, families: dict, named: dict, mu=MU_EM):
    path = Path(path)
    arrays = {"mu": np.array(mu)}
    for name, members in families.items():
        for k, v in family_table(members).items():
            arrays[f"{name}__{k}"] = v
    np.savez_compressed(path, **arrays)
    summary = {}
    for name, o in named.items():
        rp, ra = o.moon_distance_range(4000)
        summary[name] = {
            "state0": o.state0.tolist(), "period_TU": o.period, "period_days": o.period * TU_DAYS,
            "jacobi": o.jacobi, "stability_index": o.stability_index(),
            "perilune_km": rp * LU_KM, "apolune_km": ra * LU_KM, "closure_LU": o.closure_error(),
        }
    path.with_name("named_orbits.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_family(path, name, mu=MU_EM):
    d = np.load(path)
    return [PeriodicOrbit(s, T, float(d["mu"]), name)
            for s, T in zip(d[f"{name}__states"], d[f"{name}__periods"])]


def load_named(path, name, mu=MU_EM):
    info = json.loads(Path(path).with_name("named_orbits.json").read_text())[name]
    return PeriodicOrbit(np.array(info["state0"]), info["period_TU"], mu, name)
