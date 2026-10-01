"""Milestone 4: EKF vs UKF on 7-day angles-only tracking arcs (Monte Carlo).

Runs N Monte Carlo trials (new measurement noise and initial error each trial) from a loose
prior (1000 km, 10 m/s, 1-sigma) on 7-day arcs, in two modes with their own arcs:
    track      the arc whose CRLB is closest to the orbit's median (data/observability_arcs.csv);
               prior placed at the FIRST measurement (clean filter comparison)
    reacquire  the arc that starts at the onset of the observation gap closest to REACQ_GAP_D days
               (pick_reacq_arcs); prior placed at the ARC START, so it is first propagated across
               that gap. The gap is chosen explicitly so that the experiment does not depend on
               which arc happens to have the median CRLB.

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
from visibility_study import representative_orbits, EPOCH, STEP_MIN, TARGET  # noqa: E402
from cislunar_custody.constants import MU_EM, LU_KM, VU_KMS, TU_DAYS  # noqa: E402
from cislunar_custody.filters import AnglesModel, EKF, UKF, to_physical_sigma  # noqa: E402
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability import arc_information  # noqa: E402
from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility, runs as true_runs  # noqa: E402
from cislunar_custody.timeutil import jd_grid, tu_from_jd  # noqa: E402
from cislunar_custody.scenario import build_measurements, run_filter  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso  # noqa: E402
from cislunar_custody.plotstyle import use_jas_style, save, panel_label, DOUBLE, ORBIT_LABEL, FILTER_STYLE  # noqa: E402

ARC_D, SIG0_POS_KM, SIG0_VEL_MS, SIGMA, Q_PSD = 7.0, 1000.0, 10.0, 1.0, 1e-18
REACQ_GAP_D = 3.0                       # target gap before the first measurement (reacquisition)
TEL, NET, EXCL, DAYS = "1m", "Tri-3+S", 5.0, 365     # reference sensors, as build_measurements
FILTERS = {"EKF": EKF, "UKF": UKF}
MODES = ("track", "reacquire")
COLS = {k: FILTER_STYLE[k]["color"] for k in FILTERS}


def pick_arcs():
    rows = list(csv.DictReader(open(ROOT / "data" / "observability_arcs.csv")))
    arcs = {}
    for o in dict.fromkeys(r["orbit"] for r in rows):
        sel = [r for r in rows if r["orbit"] == o and float(r["arc_d"]) == ARC_D and r["observable"] == "True"]
        s = np.array([float(r["sig_pos_max_km"]) for r in sel])
        i = int(np.argmin(np.abs(np.log(s) - np.log(np.median(s)))))
        arcs[o] = (float(sel[i]["start_day"]), float(sel[i]["sig_pos_max_km"]))
    return arcs


def pick_reacq_arcs(gap_d=REACQ_GAP_D):
    """For each orbit: (start_day, gap_days, crlb_km) of the 7-day arc that starts at the onset of the
    observation gap whose length is closest to gap_d (reference sensors; earliest on ties). Only gaps
    that follow a visible step, leave at least 2 days of the arc after the gap and end inside the year
    are used. crlb_km is the arc's CRLB (largest position axis at the arc end)."""
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    eph = Ephemeris(jd)
    t = tu_from_jd(jd, jd0)
    spd = int(round(1440 / STEP_MIN))
    n_arc = int(ARC_D * spd)
    keys = list(NETWORKS[NET])
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in keys}
    out = {}
    for oname, orb in representative_orbits().items():
        r = eph.to_eci(orb.states_at(t), MU_EM)
        per_site, union = network_visibility(eph, r, keys, SITES, TELESCOPES[TEL], TARGET, moon_excl_deg=EXCL)
        lens, starts = true_runs(~union)
        L = lens / spd
        ok = (starts > 0) & (starts + n_arc < len(union)) & (L <= ARC_D - 2.0)
        if not ok.any():
            raise RuntimeError(f"{oname}: no observation gap usable for reacquisition")
        cand = np.flatnonzero(ok)
        i = cand[np.argmin(np.abs(L[cand] - gap_d))]
        s, e = int(starts[i]), int(starts[i]) + n_arc
        idx = [np.where(per_site[k].visible[s:e + 1])[0] + s for k in keys]
        pos = np.concatenate([site_pos[k][v] for k, v in zip(keys, idx)])
        idx = np.concatenate(idx)
        res = arc_information(orb, t[s], t[idx], pos, tuple(b[idx] for b in eph.basis), [t[e] - t[s]],
                              [tuple(b[e] for b in eph.basis)], sigma_arcsec=SIGMA)[0]
        out[oname] = (s / spd, float(L[i]), float(res.sigma_pos_km[0]))
    return out


def main(n_runs):
    t_start = time.time()
    jd0 = jd_from_iso(EPOCH)
    P0 = np.diag([(SIG0_POS_KM / LU_KM) ** 2] * 3 + [(SIG0_VEL_MS * 1e-3 / VU_KMS) ** 2] * 3)
    L0 = np.linalg.cholesky(P0)
    model = AnglesModel(MU_EM, SIGMA)
    filters = {k: F(MU_EM, model, q_psd_km2_s3=Q_PSD) for k, F in FILTERS.items()}
    arcs = pick_arcs()
    reacq = pick_reacq_arcs()
    results, summary = {}, []
    for oname, orb in representative_orbits().items():
        arc = {"track": arcs[oname], "reacquire": (reacq[oname][0], reacq[oname][2])}
        runs = {(md, k): [] for md in MODES for k in filters}
        info = {}
        for i in range(n_runs):
            for md in MODES:
                rng = np.random.default_rng(1000 + i)          # same noise / initial-error stream per mode
                meas, t0, t1 = build_measurements(orb, jd0, arc[md][0], ARC_D, sigma_arcsec=SIGMA, rng=rng)
                dx = L0 @ rng.standard_normal(6)
                ts = meas[0].t if md == "track" else t0
                info[md] = {"t0": t0, "gap0": (meas[0].t - t0) * TU_DAYS, "n_meas": len(meas)}
                x0 = orb.states_at(ts)[0] + dx
                for k, f in filters.items():
                    runs[(md, k)].append(run_filter(f, x0, P0, ts, meas, orb, t_end=t1))
        results[oname] = (runs, {"reacquire": info["track"]["t0"]}, arc["track"][0])
        for (md, k), rr in runs.items():
            ok = [r for r in rr if not r["diverged"]]
            d0, crlb_km = arc[md]
            base = {"orbit": oname, "mode": md, "filter": k, "start_day": d0, "n_meas": info[md]["n_meas"],
                    "gap_before_first_meas_d": info[md]["gap0"], "diverged": n_runs - len(ok)}
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
    print("  gap0 = time from arc start to first measurement (only matters for 'reacquire'); the reacquire "
          f"arc starts at the gap closest to {REACQ_GAP_D:g} d; CRLB is that of each mode's own arc")
    print("  ANEES ideal = 6; in95% = epochs in the 95% band (2nd half); runs ok = final NEES < chi2_99%(6)")

    plot_errors(results)
    plot_nees(results, n_runs)
    print(f"\ndone in {time.time() - t_start:.1f} s -> {out.name}, fig08, fig09")


def plot_errors(results):
    use_jas_style()
    fig, axes = plt.subplots(len(results), 1, figsize=(DOUBLE, 0.68 * DOUBLE), sharex=True)
    for ax, (oname, (runs, starts, d0)), letter in zip(axes, results.items(), "abcd"):
        t0 = starts["reacquire"]
        for (md, k), rr in runs.items():
            ok = [r for r in rr if not r["diverged"]]
            if not ok or md != "track":
                continue
            n = min(len(r["t"]) for r in ok)
            td = (ok[0]["t"][:n] - t0) * TU_DAYS
            e = np.median([np.linalg.norm(r["err"][:n, :3], axis=1) * LU_KM for r in ok], axis=0)
            s3 = np.median([[3 * to_physical_sigma(P)[0] for P in r["P"][:n]] for r in ok], axis=0)
            ax.semilogy(td, e, color=COLS[k], lw=0.9, ls="-", label=f"{k} |error| (median)")
            ax.semilogy(td, s3, color=COLS[k], lw=0.9, ls="--" if k == "EKF" else ":", label=f"{k} 3σ max")
        panel_label(ax, letter, f"{ORBIT_LABEL[oname]} (arc starts day {d0:.0f})")
        ax.set_ylabel("position [km]")
        ax.grid(True, which="both")
    axes[0].legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 0.98))
    axes[-1].set_xlabel("time since arc start [days]")
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig08_filter_errors", ROOT)


def plot_nees(results, n_runs):
    use_jas_style()
    fig, axes = plt.subplots(len(results), 1, figsize=(DOUBLE, 0.62 * DOUBLE), sharex=True)
    for ax, (oname, (runs, starts, d0)), letter in zip(axes, results.items(), "abcd"):
        t0 = starts["reacquire"]
        for (md, k), rr in runs.items():
            ok = [r for r in rr if not r["diverged"]]
            if not ok or md != "track":
                continue
            n = min(len(r["t"]) for r in ok)
            td = (ok[0]["t"][:n] - t0) * TU_DAYS
            ax.semilogy(td, np.mean([r["nees"][:n] for r in ok], axis=0), color=COLS[k], lw=0.8,
                        ls=FILTER_STYLE[k]["ls"], label=k)
            lo, hi = chi2.ppf([0.025, 0.975], 6 * len(ok)) / len(ok)
        ax.axhspan(lo, hi, color="grey", alpha=0.2, label="95% band")
        ax.axhline(6, color="k", lw=0.6)
        panel_label(ax, letter, ORBIT_LABEL[oname])
        ax.set_ylabel("ANEES")
        ax.grid(True, which="both")
    axes[0].legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 0.98))
    axes[-1].set_xlabel("time since arc start [days]")
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig09_nees", ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    main(ap.parse_args().runs)
