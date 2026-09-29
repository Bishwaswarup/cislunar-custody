"""Milestone 6: custody horizons for ground-based angles-only tracking.

Two scenarios per orbit (reference sensors: 1 m, Tri-3+S, 5 deg exclusion, 1 arcsec), each
with tracking arcs of 1, 3 and 7 days:
    phase     arcs ending every STEP days through the year; the gap starts at the LAST
              measurement of the arc
    blackout  arcs ending exactly when a >= 2-day observation blackout begins
For each case the post-fit CRLB covariance at the gap start is propagated across gaps of
0.1-30 days (linear, unscented and Monte Carlo) and the custody horizon T_c is found for
search patterns of 1, 10 and 100 one-degree fields of view.

    python scripts/custody_study.py                    # step 15 d, 300 samples (~10-20 min)
    python scripts/custody_study.py --step-days 45 --samples 150     # quick look (~3 min)
Outputs: data/custody_cases.csv, figures/fig12-fig15, summary tables on stdout.
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
from cislunar_custody.constants import MU_EM, TU_S, DAY_S  # noqa: E402
from cislunar_custody.custody import predict_gap, custody_horizons, search_radius_deg, strip_horizon  # noqa: E402
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability import arc_information  # noqa: E402
from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility, runs  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402
from cislunar_custody.plotstyle import (use_jas_style, save, panel_label, SINGLE, DOUBLE,  # noqa: E402
                                        ORBIT_STYLE, ORBIT_LABEL)

DAYS, ARCS_D, MIN_BLACKOUT_D = 365, (1.0, 3.0, 7.0), 2.0
TEL, NET, EXCL, SIGMA = "1m", "Tri-3+S", 5.0, 1.0
FOV_DEG, N_FIELDS = 1.0, (1, 10, 100)
DT_DAYS = np.concatenate([[0.0], np.geomspace(0.1, 30.0, 36)])
METHODS = ("ideal", "lin_claim", "lin_actual", "ut_claim", "ut_actual", "ftle")
COLS = {o: st["color"] for o, st in ORBIT_STYLE.items()}
CURVES = ROOT / "data" / "custody_curves.npz"


def main(step_days, n_samples):
    t_start = time.time()
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    spd = int(round(1440 / STEP_MIN))
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in NETWORKS[NET]}
    dt_tu = DT_DAYS * DAY_S / TU_S
    rows, curves = [], {}
    max_arc_n = int(max(ARCS_D) * spd)
    for oname, orb in representative_orbits().items():
        r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
        per_site, union = network_visibility(eph, r_eci, NETWORKS[NET], SITES, TELESCOPES[TEL], TARGET,
                                             moon_excl_deg=EXCL)
        vis_idx = np.where(union)[0]
        ends = []
        for d in np.arange(max(ARCS_D), DAYS, step_days):          # gap starts at the last measurement
            k = np.searchsorted(vis_idx, int(d * spd), side="right") - 1
            if k >= 0:
                ends.append(("phase", int(vis_idx[k]), np.nan))
        lens, starts = runs(~union)
        for L, s0 in zip(lens, starts):
            if L * STEP_MIN / 1440 >= MIN_BLACKOUT_D and s0 >= max_arc_n and s0 + L < len(union):
                ends.append(("blackout", int(s0) - 1, L * STEP_MIN / 1440))   # last visible step
        ends = list(dict.fromkeys(ends))
        curves[oname] = []
        n_ok = 0
        for scen, i_end, black_d in ends:
            for arc_d in ARCS_D:
                i0 = i_end - int(arc_d * spd)
                idx, pos = [], []
                for k, v in per_site.items():
                    vis = np.where(v.visible[i0:i_end + 1])[0] + i0
                    idx.append(vis)
                    pos.append(site_pos[k][vis])
                idx = np.concatenate(idx)
                if idx.size < 3:
                    continue
                pos = np.concatenate(pos)
                bm = tuple(b[idx] for b in eph.basis)
                be = tuple(b[i_end] for b in eph.basis)
                res = arc_information(orb, t_tu[i0], t_tu[idx], pos, bm, [t_tu[i_end] - t_tu[i0]], [be],
                                      sigma_arcsec=SIGMA)[0]
                if not res.observable:
                    continue
                x_end = orb.states_at(t_tu[i_end])[0]
                try:
                    pred = predict_gap(x_end, res.P_end, dt_tu, jd[i_end], MU_EM, n_samples=n_samples,
                                       rng=np.random.default_rng(i_end),
                                       strips=[(FOV_DEG, N) for N in N_FIELDS])
                except RuntimeError:
                    continue
                n_ok += 1
                row = {"orbit": oname, "scenario": scen, "arc_d": arc_d, "t_end_day": i_end / spd,
                       "phase": float(np.mod(t_tu[i_end], orb.period) / orb.period),
                       "n_meas": int(idx.size), "sig0_pos_km": float(res.sigma_pos_km[0]),
                       "theta0_arcsec": float(pred["theta_ideal"][0] * 3600), "blackout_d": black_d,
                       "ftle_7d_per_day": float(np.log(np.interp(7.0, DT_DAYS, pred["smax"])) / 7.0)}
                for g in (7.0, 14.0):
                    row[f"contain_lin_{g:.0f}d"] = float(np.interp(g, DT_DAYS, pred["contain_lin"]))
                    row[f"contain_ut_{g:.0f}d"] = float(np.interp(g, DT_DAYS, pred["contain_ut"]))
                for N in N_FIELDS:
                    tc = custody_horizons(pred, search_radius_deg(FOV_DEG, N))
                    for m in METHODS:
                        row[f"tc_{m}_N{N}"] = tc[m] * TU_S / DAY_S if np.isfinite(tc[m]) else np.inf
                    ts = strip_horizon(pred, N)
                    row[f"tc_strip_N{N}"] = ts * TU_S / DAY_S if np.isfinite(ts) else np.inf
                row["ut_axis_ratio_14d"] = float(np.interp(14.0, DT_DAYS, pred["ut_axis_ratio"]))
                rows.append(row)
                if scen == "phase" and arc_d == max(ARCS_D):
                    curves[oname].append({k: pred[k] for k in ("theta_ideal", "contain_lin", "contain_ut")})
        print(f"  {oname}: {n_ok} cases ({time.time() - t_start:.0f} s)", flush=True)

    out = ROOT / "data" / "custody_cases.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    save_curves(curves)
    report(rows)
    make_plots(rows, curves)
    print(f"\ndone in {time.time() - t_start:.0f} s -> {out.name}, fig12-fig15")


def save_curves(curves):
    """7-day phase-scenario curves used by fig12/fig14, so figures can be redrawn with --plots-only."""
    out = {}
    for i, (o, cs) in enumerate(curves.items()):
        out[f"name_{i}"] = np.array(o)
        for key in ("theta_ideal", "contain_lin", "contain_ut"):
            out[f"{key}_{i}"] = np.array([c[key] for c in cs]) if cs else np.zeros((0, len(DT_DAYS)))
    np.savez(CURVES, dt_days=DT_DAYS, **out)


def load_rows_and_curves():
    rows = []
    for r in csv.DictReader(open(ROOT / "data" / "custody_cases.csv")):
        rows.append({k: (v if k in ("orbit", "scenario") else float(v)) for k, v in r.items()})
    d = np.load(CURVES)
    curves, i = {}, 0
    while f"name_{i}" in d:
        o = str(d[f"name_{i}"])
        n = d[f"theta_ideal_{i}"].shape[0]
        curves[o] = [{k: d[f"{k}_{i}"][j] for k in ("theta_ideal", "contain_lin", "contain_ut")} for j in range(n)]
        i += 1
    return rows, curves


def make_plots(rows, curves):
    use_jas_style()
    plot_growth(curves)
    plot_predictor(rows)
    plot_containment(curves)
    plot_blackouts(rows)


def _fmt(x):
    return ">30" if not np.isfinite(x) or x >= 30 else f"{x:5.1f}"


def _r2(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a < 30) & (b < 30)
    if ok.sum() < 3:
        return np.nan, ok.sum()
    la, lb = np.log(a[ok]), np.log(b[ok])
    return 1 - np.sum((la - lb) ** 2) / np.sum((la - la.mean()) ** 2), int(ok.sum())


def report(rows):
    orbits = list(dict.fromkeys(r["orbit"] for r in rows))
    ph = [r for r in rows if r["scenario"] == "phase"]
    print(f"\nCustody horizon T_c [days], 'phase' scenario: median (p10-p90) of the ideal horizon; "
          f"search = N x {FOV_DEG:.0f} deg fields")
    print(f"  {'orbit':22s} {'arc':>4s} {'cases':>5s} {'sig0 km':>8s} " + " ".join(f"{'N=' + str(N):>20s}" for N in N_FIELDS))
    for o in orbits:
        for arc in ARCS_D:
            sel = [r for r in ph if r["orbit"] == o and r["arc_d"] == arc]
            if not sel:
                print(f"  {o:22s} {arc:3.0f}d     0")
                continue
            cells = []
            for N in N_FIELDS:
                v = np.minimum(np.array([r[f"tc_ideal_N{N}"] for r in sel], float), 30.0)
                p10, p50, p90 = np.percentile(v, [10, 50, 90])
                cells.append(f"{_fmt(p50)} ({_fmt(p10)}-{_fmt(p90)})")
            s0 = np.median([r["sig0_pos_km"] for r in sel])
            print(f"  {o:22s} {arc:3.0f}d {len(sel):5d} {s0:8.1f} " + " ".join(f"{c:>20s}" for c in cells))

    print("\nOperator horizons (phase scenario, N=10): median of ideal (true-mean circle) / UT-centred "
          "circle / UT-centred strip aligned with the predicted ellipse [days]")
    for o in orbits:
        for arc in ARCS_D:
            sel = [r for r in ph if r["orbit"] == o and r["arc_d"] == arc]
            if not sel or "tc_strip_N10" not in sel[0]:
                continue
            cells = [_fmt(np.median(np.minimum(np.array([r[k] for r in sel], float), 30.0)))
                     for k in ("tc_ideal_N10", "tc_ut_actual_N10", "tc_strip_N10")]
            print(f"  {o:22s} {arc:3.0f}d   " + " / ".join(cells))

    print("\nPredicting T_c (all cases, finite horizons < 30 d): R^2 in log space against the ideal horizon")
    for N in N_FIELDS:
        a = np.array([r[f"tc_ideal_N{N}"] for r in rows], float)
        parts = []
        for m in ("ftle", "lin_claim", "ut_claim"):
            r2, n = _r2(a, np.array([r[f"tc_{m}_N{N}"] for r in rows], float))
            parts.append(f"{m} R2={r2:5.2f} (n={n})")
        print(f"  N={N:3d}:  " + "   ".join(parts))

    print("\nFalse custody (claimed horizon > 1.1 x actual horizon), share of cases, N=10; "
          "and median Gaussian containment of the true 99% mass after 7 / 14 days")
    for o in orbits:
        sel = [r for r in rows if r["orbit"] == o]
        cells = []
        for g in ("lin", "ut"):
            c = np.array([r[f"tc_{g}_claim_N10"] for r in sel], float)
            a = np.array([r[f"tc_{g}_actual_N10"] for r in sel], float)
            fin = np.isfinite(a)
            cells.append(f"{g}: {100 * np.mean(c[fin] > 1.1 * a[fin]) if fin.any() else 0:5.1f}%")
        cont = " ".join(f"{g}{d} {100 * np.median([r[f'contain_{g}_{d}d'] for r in sel]):5.1f}%"
                        for g in ("lin", "ut") for d in (7, 14))
        print(f"  {o:22s} " + "   ".join(cells) + "   | " + cont)

    bl = [r for r in rows if r["scenario"] == "blackout"]
    print("\nBlackouts (>= 2 d): share survived (ideal T_c >= blackout length), by tracking-arc length")
    print(f"  {'orbit':22s} {'arc':>4s} {'n':>3s} {'med len':>8s} {'max len':>8s} "
          + " ".join(f"{'N=' + str(N):>7s}" for N in N_FIELDS) + "   UT-centred N=10")
    for o in orbits:
        for arc in ARCS_D:
            sel = [r for r in bl if r["orbit"] == o and r["arc_d"] == arc]
            if not sel:
                print(f"  {o:22s} {arc:3.0f}d   0")
                continue
            L = np.array([r["blackout_d"] for r in sel])
            cells = [f"{100 * np.mean(np.array([r[f'tc_ideal_N{N}'] for r in sel]) >= L):6.0f}%" for N in N_FIELDS]
            op = 100 * np.mean(np.array([r["tc_ut_actual_N10"] for r in sel]) >= L)
            print(f"  {o:22s} {arc:3.0f}d {len(sel):3d} {np.median(L):7.1f}d {L.max():7.1f}d "
                  + " ".join(cells) + f"   {op:6.0f}%")


def _search_lines(ax, horizontal=True):
    for N in N_FIELDS:
        r = search_radius_deg(FOV_DEG, N)
        ax.axhline(r, color="grey", ls=(0, (4, 2)), lw=0.6)
        ax.text(0.105, r * 1.1, f"search radius, N = {N}", color="#555555")


def plot_growth(curves):
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.46 * DOUBLE))
    for o, cs in curves.items():
        if not cs:
            continue
        st = ORBIT_STYLE[o]
        T = np.array([c["theta_ideal"] for c in cs])
        med = np.median(T, axis=0)
        lo, hi = np.percentile(T, [10, 90], axis=0)
        ax.loglog(DT_DAYS[1:], med[1:], color=st["color"], ls=st["ls"], lw=1.2, label=ORBIT_LABEL[o])
        ax.fill_between(DT_DAYS[1:], lo[1:], hi[1:], color=st["color"], alpha=0.15, lw=0)
    _search_lines(ax)
    ax.set_xlabel("gap length [days]")
    ax.set_ylabel("true 99% sky radius [deg]")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    ax.grid(True, which="both")
    fig.tight_layout()
    save(fig, "fig12_uncertainty_growth", ROOT)


def plot_predictor(rows, N=10):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.46 * DOUBLE))
    for ax, m, lab, letter in zip(axes, ("ftle", "ut_claim"),
                                  ("orbit-only FTLE predictor", "UT-predicted covariance"), "ab"):
        for o in COLS:
            st = ORBIT_STYLE[o]
            sel = [r for r in rows if r["orbit"] == o]
            a = np.array([r[f"tc_ideal_N{N}"] for r in sel], float)
            b = np.array([r[f"tc_{m}_N{N}"] for r in sel], float)
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            ax.loglog(b[ok], a[ok], ls="none", marker=st["marker"], ms=2.8, mfc="none", mew=0.6,
                      color=st["color"], label=ORBIT_LABEL[o])
        allv = np.array([r[f"tc_ideal_N{N}"] for r in rows], float)
        r2, n = _r2(allv, np.array([r[f"tc_{m}_N{N}"] for r in rows], float))
        ax.plot([0.1, 30], [0.1, 30], "k--", lw=0.7)
        ax.set_xlim(0.1, 40)
        ax.set_ylim(0.1, 40)
        ax.set_xlabel(f"predicted $T_c$ [days], {lab}")
        ax.set_ylabel("Monte Carlo $T_c$ [days]")
        ax.text(0.97, 0.04, f"log-$R^2$ = {r2:.2f}, n = {n}", transform=ax.transAxes, ha="right", va="bottom")
        ax.grid(True, which="both")
        panel_label(ax, letter)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig13_tc_predictor", ROOT)


def plot_containment(curves):
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.46 * DOUBLE))
    for o, cs in curves.items():
        if not cs:
            continue
        st = ORBIT_STYLE[o]
        for key, lw, mk in (("contain_lin", 0.8, None), ("contain_ut", 1.3, st["marker"])):
            ax.semilogx(DT_DAYS[1:], 100 * np.median([c[key] for c in cs], axis=0)[1:], color=st["color"],
                        ls=":" if key == "contain_lin" else "-", lw=lw, marker=mk, ms=2.5, markevery=3,
                        label=f"{ORBIT_LABEL[o]} ({'linear' if key == 'contain_lin' else 'UT'})")
    ax.axhline(99, color="k", lw=0.6)
    ax.set_xlabel("gap length [days]")
    ax.set_ylabel("true mass inside 99% sky ellipse [%]")
    ax.set_ylim(0, 102)
    ax.legend(ncol=2, loc="lower left")
    ax.grid(True, which="both")
    fig.tight_layout()
    save(fig, "fig14_gaussian_containment", ROOT)


def plot_blackouts(rows, N=10):
    fig, ax = plt.subplots(figsize=(SINGLE, 1.18 * SINGLE))
    marks = {1.0: "v", 3.0: "s", 7.0: "o"}
    for o in COLS:
        for arc, mk in marks.items():
            sel = [r for r in rows if r["orbit"] == o and r["scenario"] == "blackout" and r["arc_d"] == arc]
            L = np.array([r["blackout_d"] for r in sel])
            T = np.minimum(np.array([r[f"tc_ideal_N{N}"] for r in sel], float), 35.0)
            ax.plot(L, T, mk, color=COLS[o], ms=3.2, mew=0.6, alpha=0.9, mfc="none" if arc < 7 else COLS[o])
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker="o", color=COLS[o], ms=3.5, label=ORBIT_LABEL[o]) for o in COLS]
    hs += [Line2D([], [], ls="none", marker=mk, color="grey", ms=3.5, mfc="none" if arc < 7 else "grey",
                  label=f"{arc:.0f}-day arc") for arc, mk in marks.items()]
    ax.plot([0, 35], [0, 35], "k--", lw=0.7)
    ax.text(0.7, 31.5, "survives")
    ax.text(12.5, 1.5, "lost")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 36)
    ax.set_xlabel("blackout length [days]")
    ax.set_ylabel(f"$T_c$ [days], N = {N} (35 = beyond 30 d)")
    ax.legend(handles=hs, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), handletextpad=0.3,
              columnspacing=0.8)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig15_blackout_survival", ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-days", type=float, default=15.0)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--plots-only", action="store_true", help="redraw fig12-15 from saved data")
    a = ap.parse_args()
    if a.plots_only:
        r_, c_ = load_rows_and_curves()
        report(r_)
        make_plots(r_, c_)
    else:
        main(a.step_days, a.samples)
