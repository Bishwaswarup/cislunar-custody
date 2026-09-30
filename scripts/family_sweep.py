"""Milestone 8: operator custody horizons across whole orbit families.

The paper's headline horizons so far are 'ideal': the search circle is centred on the TRUE
mean direction, which no operator knows. Here every case also gets the two horizons an
operator can actually achieve, both centred on the UT-predicted direction:
    circle  N one-degree fields in an equal-area circle            (tc_utc)
    strip   N fields laid out as a rectangle aligned with the UT-predicted sky ellipse,
            aspect matched to the ellipse, at most N fields          (tc_strip)
and the sweep covers MEMBERS of every family (not one orbit per family), each with a random
orbital phase at the epoch, so that T_c can be related to the orbit's stability index.

    python scripts/family_sweep.py                          # 8 members/family, arcs 3 & 7 d (~20-30 min)
    python scripts/family_sweep.py --members 4 --step-days 90 --samples 150      # quick look (~3 min)
    python scripts/family_sweep.py --plots-only             # redraw figures from data/family_sweep.csv
Reference sensors as in custody_study.py (1 m, Tri-3+S, 5 deg exclusion, 1 arcsec).
Outputs: data/family_sweep.csv, figures/fig18_tc_vs_stability.png, figures/fig19_operator_vs_ideal.png
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from visibility_study import EPOCH, STEP_MIN, TARGET, CAT  # noqa: E402
from custody_study import DT_DAYS, FOV_DEG, N_FIELDS, TEL, NET, EXCL, SIGMA  # noqa: E402
from cislunar_custody.catalogue import load_family  # noqa: E402
from cislunar_custody.constants import MU_EM, TU_S, DAY_S, LU_KM, R_MOON_KM  # noqa: E402
from cislunar_custody.custody import predict_gap, custody_horizons, search_radius_deg, strip_horizon  # noqa: E402
from cislunar_custody.dynamics.periodic import PeriodicOrbit  # noqa: E402
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability import arc_information  # noqa: E402
from cislunar_custody.plotstyle import use_jas_style, save, panel_label, SINGLE, DOUBLE  # noqa: E402
from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402

DAYS = 365
FAMILIES = {  # catalogue name: (label, colour, marker)
    "L1_halo_north": ("$L_1$ halo (N)", "#0072B2", "s"),
    "L2_halo_south": ("$L_2$ halo (S) / NRHO", "#000000", "o"),
    "L1_lyapunov": ("$L_1$ Lyapunov", "#CC79A7", "v"),
    "L2_lyapunov": ("$L_2$ Lyapunov", "#D55E00", "^"),
    "DRO": ("DRO", "#009E73", "D"),
}
OUT = ROOT / "data" / "family_sweep.csv"
MIN_VISIBLE = 0.01          # skip members visible < 1 % of the year (no custody problem to study)
MOON_MARGIN_KM = 200.0      # skip members that pass within 200 km of the lunar surface


def shifted(orb, phase_tu):
    """Same periodic orbit, but at orbit time phase_tu at the epoch."""
    return PeriodicOrbit(orb.states_at(0.0, phase_tu)[0], orb.period, orb.mu, orb.family)


def pick_members(name, n):
    d = np.load(CAT)
    rmin = d[f"{name}__r_moon_min"] * LU_KM
    ok = np.where(rmin > R_MOON_KM + MOON_MARGIN_KM)[0]
    idx = ok[np.unique(np.linspace(0, len(ok) - 1, min(n, len(ok))).astype(int))]
    fam = load_family(CAT, name)
    info = {i: {"period_d": float(d[f"{name}__periods"][i] * TU_S / DAY_S),
                "jacobi": float(d[f"{name}__jacobi"][i]),
                "stability": float(d[f"{name}__stability"][i]),
                "r_moon_min_km": float(rmin[i]),
                "r_moon_max_km": float(d[f"{name}__r_moon_max"][i] * LU_KM)} for i in idx}
    return [(int(i), fam[i], info[i]) for i in idx]


def main(n_members, step_days, arcs, n_samples, seed):
    t_start = time.time()
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    spd = int(round(1440 / STEP_MIN))
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in NETWORKS[NET]}
    dt_tu = DT_DAYS * DAY_S / TU_S
    rng_phase = np.random.default_rng(seed)
    rows = []
    for fname in FAMILIES:
        for idx, orb0, info in pick_members(fname, n_members):
            phase = float(rng_phase.uniform(0.0, orb0.period))
            orb = shifted(orb0, phase)
            r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
            per_site, union = network_visibility(eph, r_eci, NETWORKS[NET], SITES, TELESCOPES[TEL], TARGET,
                                                 moon_excl_deg=EXCL)
            vis_frac = float(np.mean(union))
            base = {"family": fname, "member": idx, **info, "phase_frac": phase / orb0.period,
                    "visible_frac": vis_frac}
            if vis_frac < MIN_VISIBLE:
                print(f"  {fname:14s} #{idx:3d}  nu {info['stability']:8.1f}  visible {100 * vis_frac:4.1f}%  -> skipped")
                continue
            vis_idx = np.where(union)[0]
            ends = []
            for dd in np.arange(max(arcs), DAYS, step_days):
                k = np.searchsorted(vis_idx, int(dd * spd), side="right") - 1
                if k >= 0:
                    ends.append(int(vis_idx[k]))
            n_ok = 0
            for i_end in dict.fromkeys(ends):
                for arc_d in arcs:
                    i0 = i_end - int(arc_d * spd)
                    if i0 < 0:
                        continue
                    idxm, pos = [], []
                    for key, v in per_site.items():
                        vis = np.where(v.visible[i0:i_end + 1])[0] + i0
                        idxm.append(vis)
                        pos.append(site_pos[key][vis])
                    idxm = np.concatenate(idxm)
                    if idxm.size < 3:
                        continue
                    pos = np.concatenate(pos)
                    bm = tuple(b[idxm] for b in eph.basis)
                    be = tuple(b[i_end] for b in eph.basis)
                    res = arc_information(orb, t_tu[i0], t_tu[idxm], pos, bm, [t_tu[i_end] - t_tu[i0]], [be],
                                          sigma_arcsec=SIGMA)[0]
                    if not res.observable:
                        continue
                    try:
                        pred = predict_gap(orb.states_at(t_tu[i_end])[0], res.P_end, dt_tu, jd[i_end], MU_EM,
                                           n_samples=n_samples, rng=np.random.default_rng(i_end),
                                           strips=[(FOV_DEG, N) for N in N_FIELDS])
                    except RuntimeError:
                        continue
                    row = {**base, "arc_d": arc_d, "t_end_day": i_end / spd, "n_meas": int(idxm.size),
                           "sig0_pos_km": float(res.sigma_pos_km[0])}
                    for N in N_FIELDS:
                        tc = custody_horizons(pred, search_radius_deg(FOV_DEG, N))
                        vals = {"ideal": tc["ideal"], "utc": tc["ut_actual"], "linc": tc["lin_actual"],
                                "strip": strip_horizon(pred, N)}
                        for k, v in vals.items():
                            row[f"tc_{k}_N{N}"] = v * TU_S / DAY_S if np.isfinite(v) else np.inf
                    rows.append(row)
                    n_ok += 1
            print(f"  {fname:14s} #{idx:3d}  nu {info['stability']:8.1f}  T {info['period_d']:5.1f} d  "
                  f"visible {100 * vis_frac:4.1f}%  cases {n_ok:3d}  ({time.time() - t_start:.0f} s)", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    report(rows)
    plots(rows)
    print(f"\ndone in {time.time() - t_start:.0f} s -> {OUT.name}, fig18, fig19")


def _med(v):
    v = np.minimum(np.asarray(v, float), 30.0)
    return float(np.median(v)) if v.size else np.nan


def _fmt(x):
    return "  n/a" if not np.isfinite(x) else (">30 " if x >= 30 else f"{x:5.1f}")


def member_table(rows, arc, N=10):
    out = []
    for (fam, mem) in dict.fromkeys((r["family"], r["member"]) for r in rows):
        sel = [r for r in rows if r["family"] == fam and r["member"] == mem and r["arc_d"] == arc]
        if not sel:
            continue
        out.append({"family": fam, "member": mem, "stability": sel[0]["stability"], "period_d": sel[0]["period_d"],
                    "n": len(sel), **{k: _med([r[f"tc_{k}_N{N}"] for r in sel]) for k in ("ideal", "utc", "strip", "linc")}})
    return out


def report(rows):
    arcs = sorted({r["arc_d"] for r in rows})
    for arc in arcs:
        print(f"\nMedian custody horizon, {arc:.0f}-day arcs, N=10 [days]: ideal (true-mean circle) | operator "
              f"UT-centred circle | operator UT-centred strip | linear-centred circle")
        print(f"  {'family':14s} {'#':>4s} {'nu':>8s} {'T [d]':>6s} {'n':>3s}   ideal   UTcirc   strip   lincirc")
        for m in member_table(rows, arc):
            print(f"  {m['family']:14s} {m['member']:4d} {m['stability']:8.1f} {m['period_d']:6.1f} {m['n']:3d}   "
                  f"{_fmt(m['ideal'])}   {_fmt(m['utc'])}   {_fmt(m['strip'])}   {_fmt(m['linc'])}")
    print("\nOperator vs ideal (all cases with a finite ideal horizon < 30 d): median ratio operator / ideal")
    for N in N_FIELDS:
        a = np.array([r[f"tc_ideal_N{N}"] for r in rows])
        ok = np.isfinite(a) & (a > 0) & (a < 30)
        cells = []
        for k in ("utc", "strip", "linc"):
            b = np.minimum(np.array([r[f"tc_{k}_N{N}"] for r in rows]), 30.0)
            cells.append(f"{k} {np.median(b[ok] / a[ok]):.2f}")
        print(f"  N={N:3d} (n={ok.sum()}): " + "   ".join(cells))
    print("\nDoes the stability index organise the horizon? Rank correlation of member median T_c "
          "(strip, N=10) with log10(nu), members with nu > 1.01")

    def _corr(label, m, key="strip"):
        m = [x for x in m if np.isfinite(x[key])]
        if len(m) < 4:
            print(f"    {label:28s} too few members ({len(m)})")
            return
        lx = np.log10([x["stability"] for x in m])
        y = [x[key] for x in m]
        rho, p = spearmanr(lx, y)
        tau, pt = kendalltau(lx, y)
        print(f"    {label:28s} rho = {rho:5.2f} (p = {p:.2g}), tau = {tau:5.2f} (p = {pt:.2g}), n = {len(m)}")

    for arc in arcs:
        m = [x for x in member_table(rows, arc) if x["stability"] > 1.01 and np.isfinite(x["strip"])]
        ncap = sum(x["strip"] >= 30 for x in m)
        print(f"  {arc:.0f}-day arcs: {ncap} of {len(m)} members capped at 30 d (strip)")
        _corr("all members", m)
        _corr("nu > 10 only", [x for x in m if x["stability"] > 10])
        for fam in FAMILIES:
            _corr(f"family {fam}", [x for x in m if x["family"] == fam])
        print("    UT-centred circle horizon:")
        _corr("all members", m, "utc")
        _corr("nu > 10 only", [x for x in m if x["stability"] > 10], "utc")


def plots(rows):
    use_jas_style()
    arc = max(r["arc_d"] for r in rows)
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.45 * DOUBLE))
    handles = []
    for fam, (lab, col, mk) in FAMILIES.items():
        mt = [m for m in member_table([r for r in rows if r["family"] == fam], arc)]
        if not mt:
            continue
        x = np.array([m["stability"] for m in mt])
        for m in mt:
            sel = [r for r in rows if r["family"] == fam and r["member"] == m["member"] and r["arc_d"] == arc]
            v = np.minimum([r["tc_strip_N10"] for r in sel], 35.0)
            lo, hi = np.percentile(v, [10, 90])
            ax.plot([m["stability"]] * 2, [lo, hi], color=col, lw=0.6, alpha=0.7)
        y = np.array([m["strip"] for m in mt])
        cap = np.array([m["strip"] >= 30 for m in mt])
        ofc = "none" if fam in ("L1_lyapunov", "L2_lyapunov") else col
        ax.semilogx(x[~cap], y[~cap], ls="none", marker=mk, ms=4, color=col, mfc=ofc)
        if cap.any():
            ax.semilogx(x[cap], np.full(cap.sum(), 33.0), ls="none", marker=mk, ms=4, mfc="none", mew=0.8, color=col)
            for xi in x[cap]:
                ax.annotate("", xy=(xi, 35.5), xytext=(xi, 33.8),
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
        fill_ok = (~cap).any() and ofc != "none"
        handles.append(Line2D([], [], ls="none", marker=mk, ms=4, color=col, mfc=col if fill_ok else "none", label=lab))
    ax.axhline(30, color="grey", ls=":", lw=0.6)
    ax.set_xlabel(r"stability index $\nu$ (1 = linearly stable)")
    ax.set_ylabel(f"operator $T_c$ [days], N = 10, {arc:.0f}-day arcs\n(arrow: capped, $\\geq$ 30 d)")
    ax.set_ylim(0, 37)
    handles.append(Line2D([], [], ls="none", marker=r"$\uparrow$", color="k", ms=6, label=r"$\geq$ 30 d (capped)"))
    ax.legend(handles=handles, ncol=3, loc="lower left")
    ax.grid(True, which="both")
    fig.tight_layout()
    save(fig, "fig18_tc_vs_stability", ROOT)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.45 * DOUBLE))
    a = np.minimum([r["tc_ideal_N10"] for r in rows], 35.0)
    for ax, key, lab, letter in zip(axes, ("tc_utc_N10", "tc_strip_N10"),
                                    ("UT-centred circle", "UT-centred strip"), "ab"):
        for fam, (flab, col, mk) in FAMILIES.items():
            sel = [i for i, r in enumerate(rows) if r["family"] == fam]
            b = np.minimum([rows[i][key] for i in sel], 35.0)
            ax.plot(a[sel], b, ls="none", marker=mk, ms=2.5, mfc="none", mew=0.5, color=col, label=flab)
        ax.plot([0, 36], [0, 36], "k--", lw=0.7)
        ax.set_xlim(0, 36)
        ax.set_ylim(0, 36)
        ax.set_xlabel("ideal $T_c$ [days] (true-mean circle)")
        ax.set_ylabel(f"operator $T_c$ [days], {lab}")
        ax.grid(True)
        panel_label(ax, letter)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig19_operator_vs_ideal", ROOT)


def load_rows():
    rows = []
    for r in csv.DictReader(open(OUT)):
        rows.append({k: (v if k == "family" else float(v)) for k, v in r.items()})
        rows[-1]["member"] = int(rows[-1]["member"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=8, help="members per family")
    ap.add_argument("--step-days", type=float, default=45.0, help="spacing of arc ends through the year")
    ap.add_argument("--arcs", type=float, nargs="+", default=[3.0, 7.0], help="tracking-arc lengths [d]")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2027, help="seed of the random orbital phases")
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    if a.plots_only:
        r_ = load_rows()
        report(r_)
        plots(r_)
    else:
        main(a.members, a.step_days, tuple(a.arcs), a.samples, a.seed)
