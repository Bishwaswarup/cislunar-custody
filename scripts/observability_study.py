"""Milestone 3: angles-only observability of cislunar orbits from the ground network.

For arcs starting every 2 days over one year, uses the ACTUAL visible measurements
(reference case: 1 m telescopes, Tri-3+S, 5 deg Moon exclusion, 1 arcsec, 10-min
cadence per site) to compute the Cramer-Rao bound at the end of 3-day and 7-day arcs.

    python scripts/observability_study.py
Outputs: data/observability_arcs.csv, figures/fig06_crlb_vs_date.png,
         figures/fig07_weak_direction.png, and a summary table on stdout.
"""
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
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability import arc_information  # noqa: E402
from cislunar_custody.sensors import (SITES, NETWORKS, TELESCOPES, network_visibility,  # noqa: E402
                                      lunar_phase_angle_deg)
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402
from cislunar_custody.plotstyle import use_jas_style, save, panel_label, DOUBLE, ORBIT_STYLE, ORBIT_LABEL  # noqa: E402

DAYS, ARC_STEP_D, ARCS_D = 365, 2.0, (3.0, 7.0)
TEL, NET, EXCL, SIGMA = "1m", "Tri-3+S", 5.0, 1.0


def main():
    t_start = time.time()
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS + max(ARCS_D), STEP_MIN)
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    steps_per_day = int(round(1440 / STEP_MIN))
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in NETWORKS[NET]}
    illum = 0.5 * (1 + np.cos(np.deg2rad(lunar_phase_angle_deg(eph.moon, eph.sun))))

    rows = []
    for oname, orb in representative_orbits().items():
        r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
        per_site, _ = network_visibility(eph, r_eci, NETWORKS[NET], SITES, TELESCOPES[TEL], TARGET,
                                         moon_excl_deg=EXCL)
        for d0 in np.arange(0.0, DAYS, ARC_STEP_D):
            i0 = int(d0 * steps_per_day)
            i1 = i0 + int(max(ARCS_D) * steps_per_day)
            idx, pos = [], []
            for k, v in per_site.items():
                vis = np.where(v.visible[i0:i1 + 1])[0] + i0
                idx.append(vis)
                pos.append(site_pos[k][vis])
            idx = np.concatenate(idx)
            pos = np.concatenate(pos) if idx.size else np.zeros((0, 3))
            ends = [i0 + int(a * steps_per_day) for a in ARCS_D]
            base = dict(orbit=oname, start_day=d0, moon_illum=float(illum[i0]))
            if idx.size < 3:
                for a in ARCS_D:
                    rows.append({**base, "arc_d": a, "n_meas": int(idx.size), "observable": False,
                                 "sig_pos_max_km": np.nan, "sig_pos_min_km": np.nan,
                                 "sig_vel_max_ms": np.nan, "weak_los_deg": np.nan})
                continue
            bm = tuple(b[idx] for b in eph.basis)
            be = [tuple(b[e] for b in eph.basis) for e in ends]
            t0 = t_tu[i0]
            res = arc_information(orb, t0, t_tu[idx], pos, bm, [t_tu[e] - t0 for e in ends], be,
                                  sigma_arcsec=SIGMA)
            for a, r in zip(ARCS_D, res):
                ok = r.observable
                rows.append({**base, "arc_d": a, "n_meas": r.n_meas, "observable": ok,
                             "sig_pos_max_km": r.sigma_pos_km[0] if ok else np.nan,
                             "sig_pos_min_km": r.sigma_pos_km[-1] if ok else np.nan,
                             "sig_vel_max_ms": r.sigma_vel_ms[0] if ok else np.nan,
                             "weak_los_deg": r.weak_los_deg if ok else np.nan})
        print(f"  {oname} done ({time.time() - t_start:.0f} s)", flush=True)

    out = ROOT / "data" / "observability_arcs.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nCRLB at arc end ({TELESCOPES[TEL].name}, {NET}, {EXCL:.0f} deg exclusion, {SIGMA} arcsec)")
    print(f"  {'orbit':22s} {'arc':>4s} {'observable':>10s} {'sig_pos max: p10 / median / p90 [km]':>40s} "
          f"{'vel med [m/s]':>13s} {'weak-LOS med [deg]':>18s}")
    for oname in dict.fromkeys(r["orbit"] for r in rows):
        for a in ARCS_D:
            sel = [r for r in rows if r["orbit"] == oname and r["arc_d"] == a]
            sp = np.array([r["sig_pos_max_km"] for r in sel], float)
            sv = np.array([r["sig_vel_max_ms"] for r in sel], float)
            wl = np.array([r["weak_los_deg"] for r in sel], float)
            frac = np.mean([r["observable"] for r in sel])
            if np.all(np.isnan(sp)):
                print(f"  {oname:22s} {a:4.0f} {100 * frac:9.0f}%   (no observable arcs)")
                continue
            p10, p50, p90 = np.nanpercentile(sp, [10, 50, 90])
            print(f"  {oname:22s} {a:4.0f} {100 * frac:9.0f}% {p10:12.2f} / {p50:9.2f} / {p90:9.2f}"
                  f" {np.nanmedian(sv):13.3f} {np.nanmedian(wl):18.1f}")

    plot_vs_date(rows)
    plot_weak(rows)
    print(f"\ndone in {time.time() - t_start:.1f} s -> {out.name}, fig06, fig07")


def plot_vs_date(rows):
    use_jas_style()
    orbits = list(dict.fromkeys(r["orbit"] for r in rows))
    fig, axes = plt.subplots(len(orbits), 1, figsize=(DOUBLE, 0.68 * DOUBLE), sharex=True)
    arc_style = {3.0: dict(color="#D55E00", ls="--", marker="v"), 7.0: dict(color="#0072B2", ls="-", marker="o")}
    for ax, oname, letter in zip(axes, orbits, "abcd"):
        ax2 = ax.twinx()
        for a in ARCS_D:
            sel = [r for r in rows if r["orbit"] == oname and r["arc_d"] == a]
            d = np.array([r["start_day"] for r in sel])
            s = np.array([r["sig_pos_max_km"] for r in sel], float)
            ax.semilogy(d, s, ms=2.0, lw=0.7, label=f"{a:.0f}-day arc", **arc_style[a])
            if a == ARCS_D[0]:
                ax2.fill_between(d, [r["moon_illum"] for r in sel], color="grey", alpha=0.18, step="mid", lw=0)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0, 1])
        ax2.set_ylabel("Moon illum.")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_ylabel(r"$\sigma_{\mathrm{pos,max}}$ [km]")
        panel_label(ax, letter, ORBIT_LABEL[oname])
        ax.grid(True, which="both")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2)
    axes[-1].set_xlabel(f"arc start [days since {EPOCH[:10]} 00:00 UTC]")
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig06_crlb_vs_date", ROOT)


def plot_weak(rows):
    use_jas_style()
    orbits = list(dict.fromkeys(r["orbit"] for r in rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 0.36 * DOUBLE))
    data = []
    for oname in orbits:
        st = ORBIT_STYLE[oname]
        sel = [r for r in rows if r["orbit"] == oname and r["arc_d"] == 7.0 and r["observable"]]
        w = np.array([r["weak_los_deg"] for r in sel], float)
        if w.size:
            a1.hist(w, bins=np.linspace(0, 90, 46), histtype="step", color=st["color"], ls=st["ls"],
                    label=ORBIT_LABEL[oname])
        data.append(np.array([r["sig_pos_max_km"] for r in sel], float))
    a1.set_xlabel("angle between weakest direction and line of sight [deg]")
    a1.set_ylabel("number of 7-day arcs")
    a1.legend()
    a1.grid(True)
    a2.boxplot([d if d.size else [np.nan] for d in data])
    a2.set_xticks(range(1, len(orbits) + 1), [ORBIT_LABEL[o] for o in orbits])
    a2.set_yscale("log")
    a2.set_ylabel(r"$\sigma_{\mathrm{pos,max}}$ after 7-day arc [km]")
    a2.grid(True, which="both")
    panel_label(a1, "a")
    panel_label(a2, "b")
    fig.tight_layout()
    save(fig, "fig07_weak_direction", ROOT)


if __name__ == "__main__":
    main()
