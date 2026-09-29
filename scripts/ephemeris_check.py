"""Milestone 7: do the CR3BP custody conclusions hold in a high-fidelity ephemeris model?

Dynamics: Earth + DE440 Moon and Sun + cannonball SRP (A/m = 0.01 m^2/kg, Cr = 1.3).
Each orbit is first transitioned to its ephemeris counterpart by multiple shooting
(Levenberg-Marquardt) in 45-day windows. A subset of the milestone-6 custody cases is
then re-run: the same CRLB post-fit covariance is mapped to inertial coordinates at the
gap start and propagated in the ephemeris model (Monte Carlo, UT, finite-difference STM).

    python scripts/ephemeris_check.py                  # ~10-15 min
    python scripts/ephemeris_check.py --max-cases 6 --samples 150     # quick look
Needs: pip install jplephem ; data/de440s.bsp (see src/cislunar_custody/ephem/de440.py)
Outputs: data/ephemeris_check.csv, data/ephemeris_counterparts.csv,
         figures/fig16_tc_cr3bp_vs_ephemeris.png, figures/fig17_growth_cr3bp_vs_ephemeris.png
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from visibility_study import representative_orbits, EPOCH, STEP_MIN, TARGET  # noqa: E402
from custody_study import DT_DAYS, FOV_DEG, N_FIELDS, TEL, NET, EXCL, SIGMA, COLS  # noqa: E402
from cislunar_custody.constants import MU_EM, TU_S, DAY_S, LU_KM  # noqa: E402
from cislunar_custody.custody import predict_gap, custody_horizons, search_radius_deg  # noqa: E402
from cislunar_custody.ephem import (DE440, EphemerisModel, correct_ephemeris_orbit,  # noqa: E402
                                    transform_jacobian, predict_gap_ephem)
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability import arc_information  # noqa: E402
from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402
from cislunar_custody.plotstyle import (use_jas_style, save, SINGLE, DOUBLE, ORBIT_STYLE,  # noqa: E402
                                        ORBIT_LABEL)

DAYS, WINDOW_D = 365, 45.0
CURVES = ROOT / "data" / "ephemeris_curves.npz"


def counterparts(de, orb, jd0):
    wins = []
    for w0 in np.arange(0.0, DAYS, WINDOW_D):
        m = EphemerisModel(de, jd0 + w0, WINDOW_D + 3.0)
        tr = correct_ephemeris_orbit(m, de, orb, w0 * DAY_S / TU_S, WINDOW_D, MU_EM, max_iter=30)
        wins.append((w0, tr))
    return wins


def inertial_state(wins, t_day):
    for w0, tr in wins:
        if w0 <= t_day < w0 + WINDOW_D:
            return tr.states_at((t_day - w0) * DAY_S / TU_S)[0]
    raise ValueError(t_day)


def select_cases(max_cases):
    rows = list(csv.DictReader(open(ROOT / "data" / "custody_cases.csv")))
    out = {}
    for o in dict.fromkeys(r["orbit"] for r in rows):
        bl = [r for r in rows if r["orbit"] == o and r["scenario"] == "blackout"
              and float(r["blackout_d"]) >= 8.0 and float(r["arc_d"]) in (1.0, 7.0)]
        ph = [r for r in rows if r["orbit"] == o and r["scenario"] == "phase" and float(r["arc_d"]) == 7.0]
        pick = lambda L, n: [L[i] for i in np.unique(np.linspace(0, len(L) - 1, min(n, len(L))).astype(int))] if L else []
        out[o] = pick(bl, max_cases) + pick(ph, max_cases // 2)
    return out


def main(max_cases, n_samples):
    t_start = time.time()
    de = DE440()
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    spd = int(round(1440 / STEP_MIN))
    dt_tu = DT_DAYS * DAY_S / TU_S
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in NETWORKS[NET]}
    cases = select_cases(max_cases)
    rows, cp_rows, curves = [], [], {}
    for oname, orb in representative_orbits().items():
        t0 = time.time()
        wins = counterparts(de, orb, jd0)
        res = np.array([tr.residual for _, tr in wins]) * LU_KM
        dev = np.concatenate([tr.deviation_km() for _, tr in wins])
        cp_rows.append({"orbit": oname, "windows": len(wins), "max_continuity_km": float(res.max()),
                        "median_continuity_km": float(np.median(res)), "median_dev_km": float(np.median(dev)),
                        "max_dev_km": float(dev.max())})
        print(f"  {oname}: counterpart in {len(wins)} windows, continuity error median {np.median(res):.2e} / "
              f"max {res.max():.2e} km, deviation from CR3BP median {np.median(dev):.0f} km "
              f"({time.time() - t0:.0f} s)", flush=True)
        r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
        per_site, _ = network_visibility(eph, r_eci, NETWORKS[NET], SITES, TELESCOPES[TEL], TARGET,
                                         moon_excl_deg=EXCL)
        curves[oname] = []
        for c in cases.get(oname, []):
            arc_d, i_end = float(c["arc_d"]), int(round(float(c["t_end_day"]) * spd))
            i0 = i_end - int(arc_d * spd)
            idx, pos = [], []
            for k, v in per_site.items():
                vis = np.where(v.visible[i0:i_end + 1])[0] + i0
                idx.append(vis)
                pos.append(site_pos[k][vis])
            idx = np.concatenate(idx)
            pos = np.concatenate(pos)
            bm = tuple(b[idx] for b in eph.basis)
            be = tuple(b[i_end] for b in eph.basis)
            crlb = arc_information(orb, t_tu[i0], t_tu[idx], pos, bm, [t_tu[i_end] - t_tu[i0]], [be],
                                   sigma_arcsec=SIGMA)[0]
            if not crlb.observable:
                continue
            s_end = orb.states_at(t_tu[i_end])[0]
            try:
                p_cr = predict_gap(s_end, crlb.P_end, dt_tu, jd[i_end], MU_EM, n_samples=n_samples,
                                   rng=np.random.default_rng(i_end))
                J = transform_jacobian(s_end, jd[i_end], de, MU_EM)
                x_in = inertial_state(wins, i_end / spd)
                model = EphemerisModel(de, jd[i_end], DT_DAYS[-1] + 1.0)
                p_ep = predict_gap_ephem(model, x_in, J @ crlb.P_end @ J.T, dt_tu, n_samples=n_samples,
                                         rng=np.random.default_rng(i_end))
            except RuntimeError:
                continue
            row = {"orbit": oname, "scenario": c["scenario"], "arc_d": arc_d, "t_end_day": i_end / spd,
                   "blackout_d": float(c["blackout_d"]) if c["blackout_d"] not in ("", "nan") else np.nan,
                   "sig0_pos_km": float(crlb.sigma_pos_km[0])}
            for N in N_FIELDS:
                r = search_radius_deg(FOV_DEG, N)
                tcc, tce = custody_horizons(p_cr, r), custody_horizons(p_ep, r)
                for tag, tc in (("cr3bp", tcc), ("ephem", tce)):
                    for m in ("ideal", "ut_claim", "lin_claim", "ut_actual"):
                        row[f"tc_{tag}_{m}_N{N}"] = tc[m] * TU_S / DAY_S if np.isfinite(tc[m]) else np.inf
            for g in (7.0, 14.0):
                for tag, p in (("cr3bp", p_cr), ("ephem", p_ep)):
                    row[f"contain_lin_{tag}_{g:.0f}d"] = float(np.interp(g, DT_DAYS, p["contain_lin"]))
                    row[f"contain_ut_{tag}_{g:.0f}d"] = float(np.interp(g, DT_DAYS, p["contain_ut"]))
            rows.append(row)
            if arc_d == 7.0:
                curves[oname].append((p_cr["theta_ideal"], p_ep["theta_ideal"]))
        print(f"    {len([r for r in rows if r['orbit'] == oname])} cases ({time.time() - t_start:.0f} s)", flush=True)

    for name, data in (("ephemeris_check.csv", rows), ("ephemeris_counterparts.csv", cp_rows)):
        with open(ROOT / "data" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    out = {}
    for i, (o, cs) in enumerate(curves.items()):
        out[f"name_{i}"] = np.array(o)
        out[f"cr3bp_{i}"] = np.array([c[0] for c in cs]) if cs else np.zeros((0, len(DT_DAYS)))
        out[f"ephem_{i}"] = np.array([c[1] for c in cs]) if cs else np.zeros((0, len(DT_DAYS)))
    np.savez(CURVES, dt_days=DT_DAYS, **out)
    report(rows)
    use_jas_style()
    plot_scatter(rows)
    plot_growth(curves)
    print(f"\ndone in {time.time() - t_start:.0f} s -> ephemeris_check.csv, fig16, fig17")


def _med(v):
    v = np.minimum(np.asarray(v, float), 30.0)
    return ">30" if np.median(v) >= 30 else f"{np.median(v):5.1f}"


def report(rows):
    orbits = list(dict.fromkeys(r["orbit"] for r in rows))
    print("\nCustody horizon T_c (ideal, N=10 fields) [days]: median CR3BP vs ephemeris, by tracking arc")
    print(f"  {'orbit':22s} {'arc':>4s} {'n':>3s} {'CR3BP':>7s} {'ephem':>7s} {'median ratio e/c':>17s}")
    for o in orbits:
        for arc in (1.0, 7.0):
            sel = [r for r in rows if r["orbit"] == o and r["arc_d"] == arc]
            if not sel:
                continue
            c = np.array([r["tc_cr3bp_ideal_N10"] for r in sel])
            e = np.array([r["tc_ephem_ideal_N10"] for r in sel])
            fin = np.isfinite(c) & np.isfinite(e) & (c > 0)
            ratio = f"{np.median(e[fin] / c[fin]):8.2f}" if fin.any() else "     n/a (both >30 d)"
            print(f"  {o:22s} {arc:3.0f}d {len(sel):3d} {_med(c):>7s} {_med(e):>7s} {ratio:>17s}")

    print("\nBlackouts >= 8 d survived (ideal T_c >= length), N = 10 fields: CR3BP vs ephemeris")
    for o in orbits:
        for arc in (1.0, 7.0):
            sel = [r for r in rows if r["orbit"] == o and r["scenario"] == "blackout" and r["arc_d"] == arc]
            if not sel:
                continue
            L = np.array([r["blackout_d"] for r in sel])
            sc = 100 * np.mean(np.array([r["tc_cr3bp_ideal_N10"] for r in sel]) >= L)
            se = 100 * np.mean(np.array([r["tc_ephem_ideal_N10"] for r in sel]) >= L)
            print(f"  {o:22s} {arc:3.0f}d n={len(sel):2d}   CR3BP {sc:5.0f}%   ephemeris {se:5.0f}%")

    print("\nIn the ephemeris model: UT predictor R^2 (log) vs Monte Carlo, and 14-day Gaussian containment")
    for N in N_FIELDS:
        a = np.array([r[f"tc_ephem_ideal_N{N}"] for r in rows])
        b = np.array([r[f"tc_ephem_ut_claim_N{N}"] for r in rows])
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a < 30) & (b < 30)
        if ok.sum() >= 3:
            la, lb = np.log(a[ok]), np.log(b[ok])
            r2 = 1 - np.sum((la - lb) ** 2) / np.sum((la - la.mean()) ** 2)
            print(f"  N={N:3d}: UT R2 = {r2:5.2f} (n={ok.sum()})")
    for o in orbits:
        sel = [r for r in rows if r["orbit"] == o]
        print(f"  {o:22s} lin {100 * np.median([r['contain_lin_ephem_14d'] for r in sel]):5.1f}%   "
              f"UT {100 * np.median([r['contain_ut_ephem_14d'] for r in sel]):5.1f}%   "
              f"(CR3BP: lin {100 * np.median([r['contain_lin_cr3bp_14d'] for r in sel]):5.1f}%, "
              f"UT {100 * np.median([r['contain_ut_cr3bp_14d'] for r in sel]):5.1f}%)")


def plot_scatter(rows, N=10):
    fig, ax = plt.subplots(figsize=(SINGLE, 0.95 * SINGLE))
    from matplotlib.lines import Line2D
    for o, col in COLS.items():
        for arc, mk in ((1.0, "v"), (7.0, "o")):
            sel = [r for r in rows if r["orbit"] == o and r["arc_d"] == arc]
            c = np.minimum([r[f"tc_cr3bp_ideal_N{N}"] for r in sel], 35.0)
            e = np.minimum([r[f"tc_ephem_ideal_N{N}"] for r in sel], 35.0)
            ax.plot(c, e, mk, color=col, ms=3.5, mew=0.7, mfc="none" if arc == 1.0 else col)
    hs = [Line2D([], [], ls="none", marker="o", color=COLS[o], ms=3.5, label=ORBIT_LABEL[o]) for o in COLS]
    hs += [Line2D([], [], ls="none", marker=mk, color="grey", ms=3.5, mfc="none" if a == 1.0 else "grey",
                  label=f"{a:.0f}-day arc") for a, mk in ((1.0, "v"), (7.0, "o"))]
    ax.plot([0, 36], [0, 36], "k--", lw=0.7)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 36)
    ax.set_xlabel("$T_c$, CR3BP [days] (35 = beyond 30 d)")
    ax.set_ylabel("$T_c$, DE440 + SRP [days]")
    ax.legend(handles=hs, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig16_tc_cr3bp_vs_ephemeris", ROOT)


def plot_growth(curves):
    fig, ax = plt.subplots(figsize=(SINGLE, 0.95 * SINGLE))
    for o, cs in curves.items():
        if not cs:
            continue
        st = ORBIT_STYLE[o]
        cr = np.median([c[0] for c in cs], axis=0)
        ep = np.median([c[1] for c in cs], axis=0)
        ax.loglog(DT_DAYS[1:], cr[1:], color=st["color"], ls="--", lw=0.8)
        ax.loglog(DT_DAYS[1:], ep[1:], color=st["color"], ls="-", lw=1.2, marker=st["marker"], ms=2.5,
                  markevery=4, label=ORBIT_LABEL[o])
    for N in N_FIELDS:
        ax.axhline(search_radius_deg(FOV_DEG, N), color="grey", ls=":", lw=0.6)
    ax.plot([], [], "k--", lw=0.8, label="CR3BP")
    ax.plot([], [], "k-", lw=1.2, label="DE440 + SRP")
    ax.set_xlabel("gap length [days]")
    ax.set_ylabel("true 99% sky radius [deg]")
    ax.legend(loc="upper left")
    ax.grid(True, which="both")
    fig.tight_layout()
    save(fig, "fig17_growth_cr3bp_vs_ephemeris", ROOT)


def plots_only():
    rows = []
    for r in csv.DictReader(open(ROOT / "data" / "ephemeris_check.csv")):
        rows.append({k: (v if k in ("orbit", "scenario") else float(v)) for k, v in r.items()})
    d = np.load(CURVES)
    curves, i = {}, 0
    while f"name_{i}" in d:
        curves[str(d[f"name_{i}"])] = list(zip(d[f"cr3bp_{i}"], d[f"ephem_{i}"]))
        i += 1
    report(rows)
    use_jas_style()
    plot_scatter(rows)
    plot_growth(curves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=10, help="blackout cases per orbit (phase cases: half)")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--plots-only", action="store_true", help="redraw fig16-17 from saved data")
    a = ap.parse_args()
    if a.plots_only:
        plots_only()
    else:
        main(a.max_cases, a.samples)
