"""Milestone 5: reacquisition after an observation gap with EKF, UKF, GM-UKF and PF.

Same 7-day arcs as milestone 4, 'reacquire' mode: the prior (1000 km, 10 m/s) sits at
the arc start and is propagated across the gap before the first measurement.

    python scripts/nonlinear_filter_study.py                  # 10 runs, 3000 particles
    python scripts/nonlinear_filter_study.py --runs 20 --particles 5000
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

NAMES = ("EKF", "UKF", "GM-UKF", "PF→UKF")
COLS = {"EKF": "#d1495b", "UKF": "#2a6fdb", "GM-UKF": "#2a9d8f", "PF→UKF": "#6a4c93"}


def make_filters(model, n_particles, seed):
    return {
        "EKF": EKF(MU_EM, model, q_psd_km2_s3=Q_PSD),
        "UKF": UKF(MU_EM, model, q_psd_km2_s3=Q_PSD),
        "GM-UKF": GMUKF(MU_EM, model, q_psd_km2_s3=Q_PSD),
        "PF→UKF": ParticleFilter(MU_EM, model, n=n_particles, q_psd_km2_s3=Q_PSD,
                             rng=np.random.default_rng(seed)),
    }


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
    first measurement (1.0 = the Gaussian captures the true predicted spread)."""
    print("\nPrior at first measurement (run 0): PF particles inside each filter's 99% ellipsoid")
    c99 = chi2.ppf(0.99, 6)
    for oname in dict.fromkeys(o for o, _ in clouds):
        pf, bpf, _ = clouds[(oname, "PF→UKF")]
        if bpf is None:
            continue
        X = bpf["X"]
        cells = []
        for k in ("EKF", "UKF", "GM-UKF"):
            f, b, _ = clouds[(oname, k)]
            if b is None:
                continue
            m, P = f.moments(b)
            d = X - m
            cells.append(f"{k} {100 * np.mean(np.einsum('ij,ij->i', d, np.linalg.solve(P, d.T).T) < c99):5.1f}%")
        print(f"  {oname:22s} " + "   ".join(cells))


def _ellipse(ax, mean2, cov2, nsig, **kw):
    w, V = np.linalg.eigh(cov2)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    ax.add_patch(Ellipse(mean2, 2 * nsig * np.sqrt(w[-1]), 2 * nsig * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def plot_clouds(clouds, oname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    truth = clouds[(oname, "PF→UKF")][2]
    for ax, (i, j, lab) in zip(axes, [(0, 1, ("x", "y")), (0, 2, ("x", "z")), (1, 2, ("y", "z"))]):
        f, b, _ = clouds[(oname, "PF→UKF")]
        X = (b["X"] - truth) * LU_KM
        sub = X[:: max(1, len(X) // 2000)]
        ax.scatter(sub[:, i], sub[:, j], s=1, color=COLS["PF→UKF"], alpha=0.35, label="PF particles")
        for k in ("EKF", "UKF"):
            f, bb, _ = clouds[(oname, k)]
            m, P = f.moments(bb)
            _ellipse(ax, (m[[i, j]] - truth[[i, j]]) * LU_KM, P[np.ix_([i, j], [i, j])] * LU_KM ** 2, 3,
                     color=COLS[k], lw=1.6, label=f"{k} 3σ")
        f, bb, _ = clouds[(oname, "GM-UKF")]
        W, M, Ps = bb
        for w, m, P in zip(W, M, Ps):
            _ellipse(ax, (m[[i, j]] - truth[[i, j]]) * LU_KM, P[np.ix_([i, j], [i, j])] * LU_KM ** 2, 1,
                     color=COLS["GM-UKF"], lw=0.8, alpha=min(1.0, 0.25 + 3 * w))
        ax.plot([], [], color=COLS["GM-UKF"], label=f"GM-UKF comps 1σ ({len(W)})")
        ax.plot(0, 0, "k+", ms=12, mew=2, label="truth")
        ax.set_xlabel(f"Δ{lab[0]} synodic [km]")
        ax.set_ylabel(f"Δ{lab[1]} synodic [km]")
        ax.grid(alpha=0.3)
        ax.autoscale_view()
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{oname}: predicted uncertainty at the first measurement after the initial gap (run 0)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig10_prior_cloud.png", dpi=180)


def plot_success(summary):
    orbits = list(dict.fromkeys(s["orbit"] for s in summary))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    wbar = 0.2
    for j, k in enumerate(NAMES):
        v = [100 * next((s.get("runs_consistent", 0.0) for s in summary if s["orbit"] == o and s["filter"] == k), 0)
             for o in orbits]
        ax.bar(np.arange(len(orbits)) + (j - 1.5) * wbar, v, wbar, color=COLS[k], label=k)
    ax.set_xticks(range(len(orbits)), orbits)
    ax.set_ylabel("runs consistent at arc end [%]")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Reacquisition after the initial gap: share of runs ending consistent (NEES < χ²₉₉%)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig11_reacquisition_success.png", dpi=180)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--particles", type=int, default=3000)
    a = ap.parse_args()
    main(a.runs, a.particles)
