"""Milestone 5: reacquisition after an observation gap with EKF, UKF, GM-UKF and PF.

Same 7-day arcs as milestone 4, 'reacquire' mode: the prior (1000 km, 10 m/s) sits at
the arc start and is propagated across the gap before the first measurement.

    python scripts/nonlinear_filter_study.py                  # 100 runs, 3000 particles
    python scripts/nonlinear_filter_study.py --runs 10        # quick look
Outputs: data/nonlinear_filter_summary.csv, figures/fig10_prior_cloud.png,
         figures/fig11_reacquisition_success.png
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from visibility_study import representative_orbits, EPOCH  # noqa: E402
from filter_study import pick_arcs, ARC_D, SIG0_POS_KM, SIG0_VEL_MS, SIGMA, Q_PSD  # noqa: E402
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS, TU_DAYS  # noqa: E402
from cislunar_custody.filters import (AnglesModel, EKF, UKF, GMUKF, ParticleFilter,  # noqa: E402
                                      to_physical_sigma)
from cislunar_custody.scenario import build_measurements, run_filter  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso  # noqa: E402
from cislunar_custody.plotstyle import (use_jas_style, save, panel_label, DOUBLE, ORBIT_LABEL,  # noqa: E402
                                        FILTER_STYLE)

NAMES = ("EKF", "UKF", "GM-UKF", "PF→UKF")
COLS = {k: FILTER_STYLE[k]["color"] for k in NAMES}


def make_filters(model, n_particles, seed):
    return {
        "EKF": EKF(MU_EM, model, q_psd_km2_s3=Q_PSD),
        "UKF": UKF(MU_EM, model, q_psd_km2_s3=Q_PSD),
        "GM-UKF": GMUKF(MU_EM, model, q_psd_km2_s3=Q_PSD),
        "PF→UKF": ParticleFilter(MU_EM, model, n=n_particles, q_psd_km2_s3=Q_PSD,
                             rng=np.random.default_rng(seed)),
    }


def plots_only():
    """Redraw fig11 from data/nonlinear_filter_summary.csv (fig10 needs a full run)."""
    summary = []
    for r in csv.DictReader(open(ROOT / "data" / "nonlinear_filter_summary.csv")):
        summary.append({k: (v if k in ("orbit", "filter") else float(v)) for k, v in r.items() if v != ""})
    plot_success(summary)


def main(n_runs, n_particles):
    t_start = time.time()
    jd0 = jd_from_iso(EPOCH)
    P0 = np.diag([(SIG0_POS_KM / LU_KM) ** 2] * 3 + [(SIG0_VEL_MS * 1e-3 / VU_KMS) ** 2] * 3)
    L0 = np.linalg.cholesky(P0)
    model = AnglesModel(MU_EM, SIGMA)
    arcs = pick_arcs()
    summary, clouds = [], {}
    c99 = chi2.ppf(0.99, 6)
    for oname, orb in representative_orbits().items():
        d0, crlb_km = arcs[oname]
        res = {k: [] for k in NAMES}
        wall = {k: 0.0 for k in NAMES}
        for i in range(n_runs):
            rng = np.random.default_rng(1000 + i)
            meas, t0, t1 = build_measurements(orb, jd0, d0, ARC_D, sigma_arcsec=SIGMA, rng=rng)
            x0 = orb.states_at(t0)[0] + L0 @ rng.standard_normal(6)
            for k, f in make_filters(model, n_particles, 5000 + i).items():
                tt = time.time()
                out = run_filter(f, x0, P0, t0, meas, orb, t_end=t1, keep_first_prior=(i == 0))
                wall[k] += time.time() - tt
                res[k].append(out)
                if i == 0:
                    clouds[(oname, k)] = (f, out["first_prior"], orb.states_at(meas[0].t)[0])
        gap0 = (meas[0].t - t0) * TU_DAYS
        for k in NAMES:
            ok = [r for r in res[k] if not r["diverged"]]
            base = {"orbit": oname, "filter": k, "gap_before_first_meas_d": gap0,
                    "runs": n_runs, "diverged": n_runs - len(ok), "sec_per_run": wall[k] / n_runs}
            if not ok:
                summary.append(base)
                continue
            fn = np.array([r["nees"][-1] for r in ok])
            ep = np.array([np.linalg.norm(r["err"][-1, :3]) * LU_KM for r in ok])
            sp = np.array([to_physical_sigma(r["P"][-1])[0] for r in ok])
            n = min(len(r["t"]) for r in ok)
            anees = np.mean([r["nees"][:n] for r in ok], axis=0)
            summary.append({**base,
                "runs_consistent": float(np.sum(fn < c99)) / n_runs,
                "median_pos_err_km": float(np.median(ep)),
                "max_pos_err_km": float(np.max(ep)),
                "median_sig_pos_km": float(np.median(sp)),
                "crlb_km": crlb_km,
                "anees_2nd_half": float(anees[n // 2:].mean()),
                "mean_components": float(np.mean([r["n_comp"].mean() for r in ok])),
                "max_components": int(max(r["n_comp"].max() for r in ok))})
        print(f"  {oname} done ({time.time() - t_start:.0f} s)", flush=True)

    out = ROOT / "data" / "nonlinear_filter_summary.csv"
    keys = list(dict.fromkeys(k for s in summary for k in s))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    print(f"\nReacquisition, {n_runs} runs, prior {SIG0_POS_KM:.0f} km / {SIG0_VEL_MS:.0f} m/s at arc start, "
          f"PF N = {n_particles}")
    print(f"  {'orbit':22s} {'filter':7s} {'gap0':>5s} {'consistent':>10s} {'med err':>9s} {'max err':>9s} "
          f"{'med 1sig':>9s} {'CRLB':>7s} {'ANEES':>8s} {'comps':>6s} {'s/run':>6s}")
    for s in summary:
        if "median_pos_err_km" not in s:
            print(f"  {s['orbit']:22s} {s['filter']:7s} all runs diverged")
            continue
        print(f"  {s['orbit']:22s} {s['filter']:7s} {s['gap_before_first_meas_d']:4.1f}d "
              f"{100 * s['runs_consistent']:9.0f}% {s['median_pos_err_km']:7.2f}km {s['max_pos_err_km']:7.1f}km "
              f"{s['median_sig_pos_km']:7.2f}km {s['crlb_km']:5.2f}km {s['anees_2nd_half']:8.2f} "
              f"{s['mean_components']:6.1f} {s['sec_per_run']:6.1f}")
    print("  consistent = final NEES < chi2_99%(6); comps = mean GM components / particles (PF phase)")

    containment(clouds)
    plot_clouds(clouds, "L1 halo (Az~30k km)")
    plot_success(summary)
    print(f"\ndone in {time.time() - t_start:.0f} s -> {out.name}, fig10, fig11")


def containment(clouds):
    """Share of PF prior particles inside each Gaussian filter's 99% ellipsoid at the
    first measurement (1.0 = the Gaussian captures the true predicted spread), and the
    extent of the particle cloud (1-99 percentile span along its two principal axes).
    Also written to data/nonlinear_prior_containment.csv."""
    print("\nPrior at first measurement (run 0): PF particles inside each filter's 99% ellipsoid")
    c99 = chi2.ppf(0.99, 6)
    rows = []
    for oname in dict.fromkeys(o for o, _ in clouds):
        pf, bpf, _ = clouds[(oname, "PF→UKF")]
        if bpf is None:
            continue
        X = bpf["X"]
        Xp = (X[:, :3] - X[:, :3].mean(0)) * LU_KM
        w, V = np.linalg.eigh(np.cov(Xp.T))
        proj = Xp @ V[:, ::-1]
        span = np.percentile(proj, 99, axis=0) - np.percentile(proj, 1, axis=0)
        row = {"orbit": oname, "span1_km": float(span[0]), "span2_km": float(span[1]), "span3_km": float(span[2])}
        cells = []
        for k in ("EKF", "UKF", "GM-UKF"):
            f, b, _ = clouds[(oname, k)]
            if b is None:
                continue
            m, P = f.moments(b)
            d = X - m
            frac = float(np.mean(np.einsum('ij,ij->i', d, np.linalg.solve(P, d.T).T) < c99))
            row[f"inside99_{k}"] = frac
            cells.append(f"{k} {100 * frac:5.1f}%")
        rows.append(row)
        print(f"  {oname:22s} " + "   ".join(cells) + f"   cloud span {span[0]:.0f} x {span[1]:.0f} km")
    if rows:
        with open(ROOT / "data" / "nonlinear_prior_containment.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def _ellipse(ax, mean2, cov2, nsig, **kw):
    w, V = np.linalg.eigh(cov2)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    ax.add_patch(Ellipse(mean2, 2 * nsig * np.sqrt(w[-1]), 2 * nsig * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def plot_clouds(clouds, oname):
    use_jas_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 0.40 * DOUBLE))
    truth = clouds[(oname, "PF→UKF")][2]
    for ax, (i, j, lab), letter in zip(axes, [(0, 1, ("x", "y")), (0, 2, ("x", "z")), (1, 2, ("y", "z"))], "abc"):
        f, b, _ = clouds[(oname, "PF→UKF")]
        X = (b["X"] - truth) * LU_KM / 1e3
        sub = X[:: max(1, len(X) // 2000)]
        ax.scatter(sub[:, i], sub[:, j], s=0.6, color="#999999", alpha=0.5, lw=0, label="PF particles",
                   rasterized=True)
        for k in ("EKF", "UKF"):
            f, bb, _ = clouds[(oname, k)]
            m, P = f.moments(bb)
            _ellipse(ax, (m[[i, j]] - truth[[i, j]]) * LU_KM / 1e3, P[np.ix_([i, j], [i, j])] * (LU_KM / 1e3) ** 2,
                     3, color=COLS[k], ls=FILTER_STYLE[k]["ls"], lw=1.1, label=f"{k} 3σ")
        f, bb, _ = clouds[(oname, "GM-UKF")]
        W, M, Ps = bb
        for w, m, P in zip(W, M, Ps):
            _ellipse(ax, (m[[i, j]] - truth[[i, j]]) * LU_KM / 1e3, P[np.ix_([i, j], [i, j])] * (LU_KM / 1e3) ** 2,
                     1, color=COLS["GM-UKF"], lw=0.6, alpha=min(1.0, 0.35 + 3 * w))
        ax.plot([], [], color=COLS["GM-UKF"], lw=0.6, label=f"GM-UKF components 1σ ({len(W)})")
        ax.plot(0, 0, "k+", ms=7, mew=1.2, label="truth")
        ax.set_xlabel(f"Δ{lab[0]} [10$^3$ km]")
        ax.set_ylabel(f"Δ{lab[1]} [10$^3$ km]")
        ax.grid(True)
        ax.autoscale_view()
        panel_label(ax, letter)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h = [Line2D([], [], ls="none", marker="o", ms=2.5, color="#999999") if lab == "PF particles" else hh
         for hh, lab in zip(h, l)]
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig10_prior_cloud", ROOT)


def plot_success(summary):
    use_jas_style()
    orbits = list(dict.fromkeys(s["orbit"] for s in summary))
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.34 * DOUBLE))
    wbar = 0.2
    for j, k in enumerate(NAMES):
        sel = [next((s for s in summary if s["orbit"] == o and s["filter"] == k), {}) for o in orbits]
        p = np.array([s_.get("runs_consistent", 0.0) for s_ in sel])
        n = np.array([s_.get("runs", 1) for s_ in sel])
        lo, hi = _wilson(p * n, n)
        x = np.arange(len(orbits)) + (j - 1.5) * wbar
        ax.bar(x, 100 * p, wbar, facecolor=COLS[k], edgecolor="k", lw=0.4, hatch=FILTER_STYLE[k]["hatch"], label=k)
        ax.errorbar(x, 100 * p, yerr=[100 * np.clip(p - lo, 0, None), 100 * np.clip(hi - p, 0, None)], fmt="none",
                    ecolor="k", elinewidth=0.6,
                    capsize=1.5)
    ax.set_xticks(range(len(orbits)), [ORBIT_LABEL[o] for o in orbits])
    ax.set_ylabel("runs consistent at arc end [%]")
    ax.set_ylim(0, 112)
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig11_reacquisition_success", ROOT)


def _wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--particles", type=int, default=3000)
    ap.add_argument("--plots-only", action="store_true", help="redraw fig11 from saved data")
    a = ap.parse_args()
    if a.plots_only:
        plots_only()
    else:
        main(a.runs, a.particles)
