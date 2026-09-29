"""Robustness check: are the custody horizons converged in the Monte Carlo sample size?

custody_study.py estimates each 99% quantile from 300 samples. This script re-runs a
subset of its cases with the SAME post-fit covariance and seed (which must reproduce
data/custody_cases.csv exactly) and with a larger, independent sample, and reports how
much the ideal custody horizon T_c changes.

    python scripts/sensitivity_check.py                   # 6 cases per orbit, 2000 samples (~5-10 min)
    python scripts/sensitivity_check.py --per-orbit 3     # quick look
Needs data/custody_cases.csv (run custody_study.py first).
Output: data/sensitivity_samples.csv and a summary on stdout.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
np.seterr(invalid="ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from visibility_study import representative_orbits, EPOCH, STEP_MIN, TARGET  # noqa: E402
from custody_study import DAYS, DT_DAYS, FOV_DEG, N_FIELDS, TEL, NET, EXCL, SIGMA  # noqa: E402
from cislunar_custody.constants import MU_EM, TU_S, DAY_S  # noqa: E402
from cislunar_custody.custody import predict_gap, custody_horizons, search_radius_deg  # noqa: E402
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability import arc_information  # noqa: E402
from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402


def pick(rows, per_orbit):
    """Cases with a finite ideal horizon (N=10) spread over the range, per orbit; orbits
    whose horizons are all beyond 30 d get phase cases with 1-day arcs (largest sigma0)."""
    out = []
    for o in dict.fromkeys(r["orbit"] for r in rows):
        sel = [r for r in rows if r["orbit"] == o and np.isfinite(float(r["tc_ideal_N10"]))]
        if not sel:
            sel = [r for r in rows if r["orbit"] == o and float(r["arc_d"]) == 1.0]
        sel.sort(key=lambda r: float(r["tc_ideal_N10"]) if np.isfinite(float(r["tc_ideal_N10"]))
                 else float(r["sig0_pos_km"]))
        seen, uniq = set(), []
        for r in sel:
            key = (r["arc_d"], r["t_end_day"])
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        idx = np.unique(np.linspace(0, len(uniq) - 1, min(per_orbit, len(uniq))).astype(int))
        out += [uniq[i] for i in idx]
    return out


def main(per_orbit, n_big):
    t_start = time.time()
    rows = list(csv.DictReader(open(ROOT / "data" / "custody_cases.csv")))
    cases = pick(rows, per_orbit)
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    spd = int(round(1440 / STEP_MIN))
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in NETWORKS[NET]}
    dt_tu = DT_DAYS * DAY_S / TU_S
    orbits = representative_orbits()
    vis_cache, out = {}, []
    for c in cases:
        oname, orb = c["orbit"], orbits[c["orbit"]]
        if oname not in vis_cache:
            r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
            vis_cache[oname] = network_visibility(eph, r_eci, NETWORKS[NET], SITES, TELESCOPES[TEL], TARGET,
                                                  moon_excl_deg=EXCL)[0]
        per_site = vis_cache[oname]
        arc_d = float(c["arc_d"])
        i_end = int(round(float(c["t_end_day"]) * spd))
        i0 = i_end - int(arc_d * spd)
        idx, pos = [], []
        for k, v in per_site.items():
            vis = np.where(v.visible[i0:i_end + 1])[0] + i0
            idx.append(vis)
            pos.append(site_pos[k][vis])
        idx, pos = np.concatenate(idx), np.concatenate(pos)
        bm = tuple(b[idx] for b in eph.basis)
        be = tuple(b[i_end] for b in eph.basis)
        res = arc_information(orb, t_tu[i0], t_tu[idx], pos, bm, [t_tu[i_end] - t_tu[i0]], [be],
                              sigma_arcsec=SIGMA)[0]
        x_end = orb.states_at(t_tu[i_end])[0]
        row = {"orbit": oname, "scenario": c["scenario"], "arc_d": arc_d, "t_end_day": float(c["t_end_day"])}
        for tag, n, seed in (("n300", 300, i_end), (f"n{n_big}", n_big, 10_000_000 + i_end)):
            pred = predict_gap(x_end, res.P_end, dt_tu, jd[i_end], MU_EM, n_samples=n,
                               rng=np.random.default_rng(seed), batch=None if n == 300 else 250)
            for N in N_FIELDS:
                tc = custody_horizons(pred, search_radius_deg(FOV_DEG, N))["ideal"]
                row[f"tc_{tag}_N{N}"] = tc * TU_S / DAY_S if np.isfinite(tc) else np.inf
        for N in N_FIELDS:
            row[f"tc_csv_N{N}"] = float(c[f"tc_ideal_N{N}"])
        out.append(row)
        print(f"  {oname:22s} {c['scenario']:8s} arc {arc_d:.0f} d: T_c(N=10) csv {row['tc_csv_N10']:6.2f}  "
              f"300 {row['tc_n300_N10']:6.2f}  {n_big} {row[f'tc_n{n_big}_N10']:6.2f}  "
              f"({time.time() - t_start:.0f} s)", flush=True)

    path = ROOT / "data" / "sensitivity_samples.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nSample-size sensitivity of the ideal T_c ({len(out)} cases): 300 vs {n_big} samples")
    for N in N_FIELDS:
        a = np.array([r[f"tc_n300_N{N}"] for r in out])
        b = np.array([r[f"tc_n{n_big}_N{N}"] for r in out])
        c = np.array([r[f"tc_csv_N{N}"] for r in out])
        fin = np.isfinite(a) & np.isfinite(b)
        rep = np.all((np.isfinite(a) == np.isfinite(c)) & (~np.isfinite(a) | (np.abs(a - c) < 1e-6)))
        d = np.abs(a[fin] - b[fin])
        rel = d / b[fin]
        print(f"  N={N:3d}: reproduces csv: {'yes' if rep else 'NO'};  |dT_c| median {np.median(d):.2f} d, "
              f"max {d.max():.2f} d;  relative median {100 * np.median(rel):.1f}%, max {100 * rel.max():.1f}%;  "
              f"finite/inf flips {int(np.sum(np.isfinite(a) != np.isfinite(b)))}")
    print(f"\ndone in {time.time() - t_start:.0f} s -> {path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-orbit", type=int, default=6)
    ap.add_argument("--samples", type=int, default=2000)
    a = ap.parse_args()
    main(a.per_orbit, a.samples)
