"""Milestone 4: EKF vs UKF on 7-day angles-only tracking arcs (Monte Carlo).

For each orbit, picks the 7-day arc whose CRLB is closest to that orbit's median
(from data/observability_arcs.csv), then runs N Monte Carlo trials (new measurement
noise and initial error each trial) from a loose prior (1000 km, 10 m/s, 1-sigma), in
two modes:
    track      prior placed at the FIRST measurement (clean filter comparison)
    reacquire  prior placed at the ARC START, so it is first propagated across any
               initial observation gap (preview of the custody experiment)

    python scripts/filter_study.py            # 20 runs per orbit
    python scripts/filter_study.py --runs 50
Outputs: data/filter_summary.csv, figures/fig08_filter_errors.png, figures/fig09_nees.png
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
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from visibility_study import representative_orbits, EPOCH  # noqa: E402
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS, TU_DAYS  # noqa: E402
from cislunar_custody.filters import AnglesModel, EKF, UKF, to_physical_sigma  # noqa: E402
from cislunar_custody.scenario import build_measurements, run_filter  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso  # noqa: E402

ARC_D, SIG0_POS_KM, SIG0_VEL_MS, SIGMA, Q_PSD = 7.0, 1000.0, 10.0, 1.0, 1e-18
FILTERS = {"EKF": EKF, "UKF": UKF}
MODES = ("track", "reacquire")
COLS = {"EKF": "#d1495b", "UKF": "#2a6fdb"}


def pick_arcs():
    rows = list(csv.DictReader(open(ROOT / "data" / "observability_arcs.csv")))
    arcs = {}
    for o in dict.fromkeys(r["orbit"] for r in rows):
        sel = [r for r in rows if r["orbit"] == o and float(r["arc_d"]) == ARC_D and r["observable"] == "True"]
        s = np.array([float(r["sig_pos_max_km"]) for r in sel])
        i = int(np.argmin(np.abs(np.log(s) - np.log(np.median(s)))))
        arcs[o] = (float(sel[i]["start_day"]), float(sel[i]["sig_pos_max_km"]))
    return arcs


def main(n_runs):
    t_start = time.time()
    jd0 = jd_from_iso(EPOCH)
    P0 = np.diag([(SIG0_POS_KM / LU_KM) ** 2] * 3 + [(SIG0_VEL_MS * 1e-3 / VU_KMS) ** 2] * 3)
    L0 = np.linalg.cholesky(P0)
    model = AnglesModel(MU_EM, SIGMA)
    filters = {k: F(MU_EM, model, q_psd_km2_s3=Q_PSD) for k, F in FILTERS.items()}
    arcs = pick_arcs()
    results, summary = {}, []
    for oname, orb in representative_orbits().items():
        d0, crlb_km = arcs[oname]
        runs = {(md, k): [] for md in MODES for k in filters}
        starts = {}
        for i in range(n_runs):
            rng = np.random.default_rng(1000 + i)
            meas, t0, t1 = build_measurements(orb, jd0, d0, ARC_D, sigma_arcsec=SIGMA, rng=rng)
            dx = L0 @ rng.standard_normal(6)
            starts = {"track": meas[0].t, "reacquire": t0}
            for md in MODES:
                ts = starts[md]
                x0 = orb.states_at(ts)[0] + dx
                for k, f in filters.items():
                    runs[(md, k)].append(run_filter(f, x0, P0, ts, meas, orb, t_end=t1))
        gap0 = (starts["track"] - starts["reacquire"]) * TU_DAYS
        results[oname] = (runs, starts, d0)
        for (md, k), rr in runs.items():
            ok = [r for r in rr if not r["diverged"]]
            base = {"orbit": oname, "mode": md, "filter": k, "start_day": d0, "n_meas": len(meas),
                    "gap_before_first_meas_d": gap0, "diverged": n_runs - len(ok)}
            if not ok:
                summary.append(base)
                continue
            ep = np.array([np.linalg.norm(r["err"][-1, :3]) * LU_KM for r in ok])
            ev = np.array([np.linalg.norm(r["err"][-1, 3:]) * VU_KMS * 1e3 for r in ok])
            sp = np.array([to_physical_sigma(r["P"][-1])[0] for r in ok])
            n_epochs = min(len(r["t"]) for r in ok)
            N = np.array([r["nees"][:n_epochs] for r in ok])
            anees = N.mean(axis=0)
            lo, hi = chi2.ppf([0.025, 0.975], 6 * len(ok)) / len(ok)
            half = slice(n_epochs // 2, n_epochs)
            final_nees = np.array([r["nees"][-1] for r in ok])
            summary.append({**base,
                "final_pos_err_rms_km": float(np.sqrt(np.mean(ep ** 2))),
                "final_vel_err_rms_ms": float(np.sqrt(np.mean(ev ** 2))),
                "final_sig_pos_max_km": float(np.median(sp)),
                "crlb_sig_pos_max_km": crlb_km,
                "anees_2nd_half": float(anees[half].mean()),
                "frac_anees_in_95": float(np.mean((anees[half] > lo) & (anees[half] < hi))),
                "frac_runs_consistent": float(np.mean(final_nees < chi2.ppf(0.99, 6))),
            })
        print(f"  {oname} done ({time.time() - t_start:.0f} s)", flush=True)

    out = ROOT / "data" / "filter_summary.csv"
    keys = list(dict.fromkeys(k for s in summary for k in s))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    print(f"\n7-day arcs, {n_runs} runs, prior {SIG0_POS_KM:.0f} km / {SIG0_VEL_MS:.0f} m/s, {SIGMA}\" noise")
    print(f"  {'orbit':22s} {'mode':9s} {'filt':4s} {'gap0':>5s} {'pos err':>9s} {'vel err':>8s} "
          f"{'1sig':>8s} {'CRLB':>8s} {'ANEES':>8s} {'in95%':>6s} {'runs ok':>7s}")
    for s_ in summary:
        if "final_pos_err_rms_km" not in s_:
            print(f"  {s_['orbit']:22s} {s_['mode']:9s} {s_['filter']:4s} all runs diverged")
            continue
        print(f"  {s_['orbit']:22s} {s_['mode']:9s} {s_['filter']:4s} {s_['gap_before_first_meas_d']:4.1f}d "
              f"{s_['final_pos_err_rms_km']:7.2f}km {s_['final_vel_err_rms_ms']:6.3f}ms "
              f"{s_['final_sig_pos_max_km']:6.2f}km {s_['crlb_sig_pos_max_km']:6.2f}km "
              f"{s_['anees_2nd_half']:8.2f} {100 * s_['frac_anees_in_95']:5.0f}% "
              f"{100 * s_['frac_runs_consistent']:6.0f}%")
    print("  gap0 = time from arc start to first measurement (only matters for 'reacquire')")
    print("  ANEES ideal = 6; in95% = epochs in the 95% band (2nd half); runs ok = final NEES < chi2_99%(6)")

    plot_errors(results)
    plot_nees(results, n_runs)
    print(f"\ndone in {time.time() - t_start:.1f} s -> {out.name}, fig08, fig09")


def plot_errors(results):
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.4 * len(results)), sharex=True)
    for ax, (oname, (runs, starts, d0)) in zip(axes, results.items()):
        t0 = starts["reacquire"]
        for (md, k), rr in runs.items():
            ok = [r for r in rr if not r["diverged"]]
            if not ok or md != "track":
                continue
            n = min(len(r["t"]) for r in ok)
            td = (ok[0]["t"][:n] - t0) * TU_DAYS
            e = np.median([np.linalg.norm(r["err"][:n, :3], axis=1) * LU_KM for r in ok], axis=0)
            s3 = np.median([[3 * to_physical_sigma(P)[0] for P in r["P"][:n]] for r in ok], axis=0)
            ax.semilogy(td, e, color=COLS[k], lw=1.0, label=f"{k} |error| (median)")
            ax.semilogy(td, s3, color=COLS[k], lw=1.0, ls="--", label=f"{k} 3σ max")
        ax.set_title(f"{oname}  (arc starts day {d0:.0f})", fontsize=9, loc="left")
        ax.set_ylabel("position [km]")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("time since arc start [days]")
    fig.suptitle(f"EKF vs UKF ('track' mode): position error and 3σ over 7-day arcs (prior {SIG0_POS_KM:.0f} km, {SIGMA}\")")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig08_filter_errors.png", dpi=180)


def plot_nees(results, n_runs):
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.2 * len(results)), sharex=True)
    for ax, (oname, (runs, starts, d0)) in zip(axes, results.items()):
        t0 = starts["reacquire"]
        for (md, k), rr in runs.items():
            ok = [r for r in rr if not r["diverged"]]
            if not ok or md != "track":
                continue
            n = min(len(r["t"]) for r in ok)
            td = (ok[0]["t"][:n] - t0) * TU_DAYS
            ax.semilogy(td, np.mean([r["nees"][:n] for r in ok], axis=0), color=COLS[k], lw=0.9, label=k)
            lo, hi = chi2.ppf([0.025, 0.975], 6 * len(ok)) / len(ok)
        ax.axhspan(lo, hi, color="grey", alpha=0.2, label="95% band")
        ax.axhline(6, color="k", lw=0.6)
        ax.set_title(oname, fontsize=9, loc="left")
        ax.set_ylabel("ANEES")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("time since arc start [days]")
    fig.suptitle(f"Filter consistency: average NEES over {n_runs} runs (ideal 6)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig09_nees.png", dpi=180)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    main(ap.parse_args().runs)
