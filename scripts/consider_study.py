"""Milestone 9: do the custody horizons survive realistic error sources?

The paper's starting covariance is the Cramer-Rao bound (CRLB) of the tracking arc, which assumes
white measurement noise only (1 arcsec per axis). Here the truth also carries errors that the operator
does not model (observability/consider.py), z_k = h(x_k) + b_s + b_c + v_k:
    bias<b>       constant angle bias per site and axis, sigma_b = b arcsec (nominal 0.1")
    common<b>     common-mode bias shared by all sites (catalogue zonal errors, one reduction pipeline)
    clock10ms     constant clock offset per site (topocentric rates are ~0.4"/s, so this is ~4 mas)
    srp_am<A/m>   SRP modelled, area-to-mass ratio A/m uncertain by 30 % (random dp)
    nosrp_am<A/m> SRP NOT modelled by the operator: the truth carries the full A/m (fixed dp)
    nominal       0.1" per-site bias, 1 ms, SRP A/m 0.01 +- 30 %
    conservative  0.5" per-site + 0.1" common bias, 10 ms, SRP A/m 0.05 +- 30 %
and the same with 0.3" white noise instead of 1" (base_0.3, nominal_0.3, conservative_0.3).
Each configuration is compared with its reference ('ref': base or base_0.3) on the same arcs.
The operator still plans with the CRLB and the plain CR3BP, exactly as in custody_study.py, so
    ideal      true 99 % sky radius (true-mean circle)                   -> how big the truth is
    operator   circle centred on the operator's UT prediction            -> the paper's headline
    claim      when the operator's own (CRLB) 99 % region outgrows the search -> false custody
Config 'base' has no extra errors and must reproduce data/custody_cases.csv.

    python scripts/consider_study.py                       # phase arcs every 30 d, 300 samples, all cores
    python scripts/consider_study.py --step-days 90 --samples 100 --configs bias0.1 nominal   # quick look
    python scripts/consider_study.py --plots-only          # redraw from data/consider_cases.csv
Outputs: data/consider_cases.csv, figures/fig20_consider_horizon.png, summary on stdout.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from visibility_study import representative_orbits, EPOCH, STEP_MIN, TARGET  # noqa: E402
from custody_study import DT_DAYS, FOV_DEG, N_FIELDS, TEL, NET, EXCL, SIGMA, DAYS  # noqa: E402
from cislunar_custody.constants import MU_EM, TU_S, DAY_S  # noqa: E402
from cislunar_custody.custody import predict_gap, custody_horizons, search_radius_deg  # noqa: E402
from cislunar_custody.dynamics.periodic import PeriodicOrbit  # noqa: E402
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.observability.consider import SRPModel, arc_consider, AM_REF  # noqa: E402
from cislunar_custody.plotstyle import use_jas_style, save, panel_label, DOUBLE, ORBIT_STYLE, ORBIT_LABEL  # noqa: E402
from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility, radec  # noqa: E402
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402

ARCS_D = (1.0, 3.0, 7.0)

def _cfg(noise=1.0, bias=0.0, common=0.0, clock=0.0, srp_sigma=0.0, srp_offset=0.0, ref="base"):
    return dict(noise=noise, bias=bias, common=common, clock=clock, srp_sigma=srp_sigma, srp_offset=srp_offset,
                ref=ref)


SRP_FRAC = 0.30          # relative A/m uncertainty when SRP is modelled
CONFIGS = {
    "base": _cfg(),
    "bias0.05": _cfg(bias=0.05), "bias0.1": _cfg(bias=0.1), "bias0.2": _cfg(bias=0.2), "bias0.5": _cfg(bias=0.5),
    "common0.1": _cfg(common=0.1), "common0.5": _cfg(common=0.5),
    "clock10ms": _cfg(clock=1e-2),
    **{f"srp_am{a:g}": _cfg(srp_sigma=SRP_FRAC * a) for a in (0.005, 0.01, 0.02, 0.05)},
    **{f"nosrp_am{a:g}": _cfg(srp_offset=a) for a in (0.01, 0.05)},
    "nominal": _cfg(bias=0.1, clock=1e-3, srp_sigma=SRP_FRAC * AM_REF),
    "conservative": _cfg(bias=0.5, common=0.1, clock=1e-2, srp_sigma=SRP_FRAC * 0.05),
    "base_0.3": _cfg(noise=0.3, ref="base_0.3"),
    "nominal_0.3": _cfg(noise=0.3, bias=0.1, clock=1e-3, srp_sigma=SRP_FRAC * AM_REF, ref="base_0.3"),
    "conservative_0.3": _cfg(noise=0.3, bias=0.5, common=0.1, clock=1e-2, srp_sigma=SRP_FRAC * 0.05,
                             ref="base_0.3"),
}
OUT = ROOT / "data" / "consider_cases.csv"
KEYS = ("ideal", "ut_actual", "ut_claim")
_ORB = {}


def _orbit(spec):
    key = spec["name"]
    if key not in _ORB:
        _ORB[key] = PeriodicOrbit(spec["state0"], spec["period"], spec["mu"], spec["family"])
    return _ORB[key]


def run_case(c):
    """All configs for one arc. Runs in a worker process."""
    orb = _orbit(c["orbit"])
    t0, te, i_end = c["t0"], c["t_end_tu"], c["i_end"]
    srp = SRPModel(c["jd0"], t0 - 0.05, te + (DT_DAYS[-1] + 1.0) * DAY_S / TU_S)
    arcs = {}
    for noise in sorted({cf["noise"] for cf in c["configs"].values()}, reverse=True):
        arcs[noise] = arc_consider(orb, t0, c["t_meas"], c["r_site"], c["basis"], te - t0, site_id=c["site_id"],
                                   rate=c["rate"], srp=srp, sigma_arcsec=noise)
    if not arcs[max(arcs)].observable:                  # same arcs as custody_study (1-arcsec test)
        return []
    x_end = orb.states_at(te)[0]
    gap_srp = srp.shifted(te)
    dt_tu = DT_DAYS * DAY_S / TU_S
    rows = []
    for name, cf in c["configs"].items():
        ca = arcs[cf["noise"]]
        P, aug, mean = ca.covariance(bias_arcsec=cf["bias"], timing_s=cf["clock"], srp_sigma=cf["srp_sigma"],
                                     common_bias_arcsec=cf["common"], srp_offset=cf["srp_offset"])
        if name in ("base", "base_0.3"):
            cons = None
        elif cf["srp_sigma"] > 0 or cf["srp_offset"] != 0:
            cons = {"P_mc": aug, "mean": mean, "accel": gap_srp}
        else:
            cons = {"P_mc": P}
        try:
            pred = predict_gap(x_end, ca.P_crlb_end, dt_tu, c["jd_end"], MU_EM, n_samples=c["n_samples"],
                               rng=np.random.default_rng(i_end), consider=cons)
        except RuntimeError:
            continue
        row = {"orbit": c["orbit"]["name"], "arc_d": c["arc_d"], "t_end_day": c["t_end_day"], "config": name,
               "ref": cf["ref"], "noise_arcsec": cf["noise"], "bias_arcsec": cf["bias"],
               "common_bias_arcsec": cf["common"], "clock_s": cf["clock"], "srp_sigma": cf["srp_sigma"],
               "srp_offset": cf["srp_offset"], "sig0_crlb_km": ca.sigma_pos_crlb_km, "sig0_km": ca.sigma_pos_km(P),
               "theta0_arcsec": float(pred["theta_ideal"][0] * 3600)}
        for N in N_FIELDS:
            tc = custody_horizons(pred, search_radius_deg(FOV_DEG, N))
            for k in KEYS:
                row[f"tc_{k}_N{N}"] = tc[k] * TU_S / DAY_S if np.isfinite(tc[k]) else np.inf
        rows.append(row)
    return rows


def build_cases(step_days, n_samples, configs, orbits):
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    spd = int(round(1440 / STEP_MIN))
    keys = list(NETWORKS[NET])
    site_pos = {k: eph.site_eci(SITES[k])[0] for k in keys}
    cases = []
    for oname, orb in representative_orbits().items():
        if orbits and not any(o.lower() in oname.lower() for o in orbits):
            continue
        r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
        per_site, union = network_visibility(eph, r_eci, keys, SITES, TELESCOPES[TEL], TARGET, moon_excl_deg=EXCL)
        # topocentric angular rate of the truth at every grid step (central difference, 10-min grid)
        rates = {}
        for k in keys:
            ra, dec = radec(r_eci - site_pos[k])
            ra = np.unwrap(ra)
            dt_s = STEP_MIN * 60.0
            rates[k] = np.stack([np.gradient(ra, dt_s), np.gradient(dec, dt_s)], axis=1)
        vis_idx = np.where(union)[0]
        spec = {"name": oname, "state0": orb.state0, "period": orb.period, "mu": orb.mu, "family": orb.family}
        for d in np.arange(max(ARCS_D), DAYS, step_days):          # same arc ends as custody_study (phase)
            kk = np.searchsorted(vis_idx, int(d * spd), side="right") - 1
            if kk < 0:
                continue
            i_end = int(vis_idx[kk])
            for arc_d in ARCS_D:
                i0 = i_end - int(arc_d * spd)
                idx, sid, rt = [], [], []
                for j, k in enumerate(keys):
                    vis = np.where(per_site[k].visible[i0:i_end + 1])[0] + i0
                    idx.append(vis)
                    sid.append(np.full(vis.size, j))
                    rt.append(rates[k][vis])
                pos = np.concatenate([site_pos[k][v] for k, v in zip(keys, idx)])
                idx = np.concatenate(idx)
                if idx.size < 3:
                    continue
                cases.append({"orbit": spec, "t0": t_tu[i0], "t_end_tu": t_tu[i_end], "jd0": jd0,
                              "i_end": i_end, "arc_d": arc_d,
                              "t_end_day": i_end / spd, "jd_end": jd[i_end], "t_meas": t_tu[idx], "r_site": pos,
                              "basis": tuple(b[idx] for b in eph.basis), "site_id": np.concatenate(sid),
                              "rate": np.concatenate(rt), "configs": configs, "n_samples": n_samples})
    return cases


def main(step_days, n_samples, jobs, configs, orbits):
    t_start = time.time()
    names = list(configs or CONFIGS)
    names = list(dict.fromkeys([CONFIGS[k]["ref"] for k in names] + names))     # references first
    configs = {k: CONFIGS[k] for k in names}
    cases = build_cases(step_days, n_samples, configs, orbits)
    print(f"{len(cases)} arcs x {len(configs)} configs, {jobs} worker(s)", flush=True)
    rows = []
    if jobs <= 1:
        for i, c in enumerate(cases):
            rows += run_case(c)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(cases)} arcs ({time.time() - t_start:.0f} s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for i, r in enumerate(ex.map(run_case, cases, chunksize=1)):
                rows += r
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(cases)} arcs ({time.time() - t_start:.0f} s)", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows)
    plots(rows)
    print(f"\ndone in {time.time() - t_start:.0f} s -> {OUT.name}, fig20")


def _cap(x):
    return np.minimum(np.asarray(x, float), 30.0)


def _fmt(x):
    return ">30 " if x >= 30 else f"{x:4.1f}"


def _pairs(rows, cfg, key="tc_ut_actual_N10"):
    """(reference, config) values for the same arcs; the reference is the config's own baseline."""
    ref = next((r["ref"] for r in rows if r["config"] == cfg), "base")
    base = {(r["orbit"], r["arc_d"], r["t_end_day"]): r for r in rows if r["config"] == ref}
    out = [(base[k][key], r[key], r) for r in rows if r["config"] == cfg
           for k in [(r["orbit"], r["arc_d"], r["t_end_day"])] if k in base]
    return out


def report(rows):
    cfgs = list(dict.fromkeys(r["config"] for r in rows))
    orbits = list(dict.fromkeys(r["orbit"] for r in rows))

    cc = ROOT / "data" / "custody_cases.csv"
    if cc.exists():
        ref = {}
        for r in csv.DictReader(open(cc)):
            if r["scenario"] == "phase":
                ref[(r["orbit"], float(r["arc_d"]), round(float(r["t_end_day"]), 6))] = r
        d, n = 0.0, 0
        for r in rows:
            k = (r["orbit"], r["arc_d"], round(r["t_end_day"], 6))
            if r["config"] == "base" and k in ref:
                for N in N_FIELDS:
                    for key in ("ideal", "ut_actual", "ut_claim"):
                        a, b = float(ref[k][f"tc_{key}_N{N}"]), r[f"tc_{key}_N{N}"]
                        if np.isfinite(a) or np.isfinite(b):
                            d = max(d, abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.inf)
                n += 1
        print(f"\nBaseline check: {n} base arcs matched in custody_cases.csv, max |dT_c| = {d:.3g} d (should be < 1e-3 with --samples 300)")

    print("\nStarting uncertainty: median largest 1-sigma position axis at the gap start [km] (CRLB -> with errors)")
    print(f"  {'orbit':22s} {'arc':>4s} " + " ".join(f"{c:>16s}" for c in cfgs))
    for o in orbits:
        for arc in ARCS_D:
            sel = [r for r in rows if r["orbit"] == o and r["arc_d"] == arc]
            if not sel:
                continue
            cells = [np.median([r["sig0_km"] for r in sel if r["config"] == c]) for c in cfgs]
            print(f"  {o:22s} {arc:3.0f}d " + " ".join(f"{x:16.2f}" for x in cells))

    print("\nOperator horizon (UT-centred circle), N=10: median over arcs [d]")
    print(f"  {'orbit':22s} {'arc':>4s} {'n':>3s} " + " ".join(f"{c:>16s}" for c in cfgs))
    for o in orbits:
        for arc in ARCS_D:
            sel = [r for r in rows if r["orbit"] == o and r["arc_d"] == arc]
            if not sel:
                continue
            n = sum(r["config"] == "base" for r in sel)
            cells = [np.median(_cap([r["tc_ut_actual_N10"] for r in sel if r["config"] == c])) for c in cfgs]
            print(f"  {o:22s} {arc:3.0f}d {n:3d} " + " ".join(f"{_fmt(x):>16s}" for x in cells))

    print("\nChange against the configuration's own CRLB reference (base, or base_0.3 for the 0.3-arcsec runs), "
          "all arcs with a reference operator horizon < 30 d")
    print(f"  {'config':16s} {'N':>4s} {'n':>4s} {'median ratio':>13s} {'p10 ratio':>10s} {'drop>20%':>9s} "
          f"{'ideal ratio':>12s} {'false custody (claim>1.1 actual)':>34s}")
    for c in cfgs:
        for N in N_FIELDS:
            pr = _pairs(rows, c, f"tc_ut_actual_N{N}")
            ok = [(a, b, r) for a, b, r in pr if np.isfinite(a) and a < 30]
            if not ok:
                continue
            ratio = np.array([min(b, 30.0) / a for a, b, _ in ok])
            pi = _pairs(rows, c, f"tc_ideal_N{N}")
            ri = np.array([min(b, 30.0) / a for a, b, _ in pi if np.isfinite(a) and a < 30])
            fc = np.mean([r[f"tc_ut_claim_N{N}"] > 1.1 * r[f"tc_ut_actual_N{N}"] for _, _, r in ok])
            print(f"  {c:16s} {N:4d} {len(ok):4d} {np.median(ratio):13.3f} {np.percentile(ratio, 10):10.3f} "
                  f"{100 * np.mean(ratio < 0.8):8.1f}% {np.median(ri):12.3f} {100 * fc:33.1f}%")

    print("\nNear-stable orbits: arcs whose operator horizon (N=10) falls from > 30 d to <= 30 d")
    for c in cfgs:
        if c in ("base", "base_0.3"):
            continue
        cells = []
        for o in orbits:
            pr = _pairs(rows, c)
            pr = [(a, b) for a, b, r in pr if r["orbit"] == o and not (np.isfinite(a) and a < 30)]
            if pr:
                cells.append(f"{o.split()[0]} {sum(np.isfinite(b) and b < 30 for a, b in pr)}/{len(pr)}")
        print(f"  {c:16s} " + "   ".join(cells))


FIG_CONFIGS = ("bias0.05", "bias0.1", "bias0.2", "bias0.5", "common0.1", "common0.5", "clock10ms",
               "srp_am0.005", "srp_am0.01", "srp_am0.02", "srp_am0.05", "nosrp_am0.01", "nosrp_am0.05",
               "nominal", "conservative", "nominal_0.3", "conservative_0.3")


LABELS = {"bias0.05": 'site bias 0.05"', "bias0.1": 'site bias 0.1"', "bias0.2": 'site bias 0.2"',
          "bias0.5": 'site bias 0.5"', "common0.1": 'common bias 0.1"', "common0.5": 'common bias 0.5"',
          "clock10ms": "clock 10 ms", "srp_am0.005": "SRP, A/m 0.005", "srp_am0.01": "SRP, A/m 0.01",
          "srp_am0.02": "SRP, A/m 0.02", "srp_am0.05": "SRP, A/m 0.05", "nosrp_am0.01": "no SRP model, 0.01",
          "nosrp_am0.05": "no SRP model, 0.05", "nominal": "nominal", "conservative": "conservative",
          "nominal_0.3": 'nominal, 0.3" noise', "conservative_0.3": 'conservative, 0.3" noise'}


def plots(rows):
    use_jas_style()
    have = {r["config"] for r in rows}
    cfgs = [c for c in FIG_CONFIGS if c in have]
    orbits = [o for o in ORBIT_STYLE if any(r["orbit"] == o for r in rows)]
    x = np.arange(len(cfgs))
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 0.72 * DOUBLE), sharex=True)
    off = np.linspace(-0.27, 0.27, max(len(orbits), 1))
    for j, o in enumerate(orbits):
        st = ORBIT_STYLE[o]
        med, lo, hi = [], [], []
        for c in cfgs:
            v = np.array([b / a for a, b, r in _pairs(rows, c, "sig0_km") if r["orbit"] == o])
            med.append(np.median(v)), lo.append(np.percentile(v, 10)), hi.append(np.percentile(v, 90))
        axes[0].errorbar(x + off[j], med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], ls="none",
                         marker=st["marker"], color=st["color"], ms=3.5, elinewidth=0.7, capsize=0,
                         label=ORBIT_LABEL[o])
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20, 50]))
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylim(0.9, None)
    axes[0].set_ylabel("starting uncertainty\n/ CRLB reference")
    unstable = [o for o in orbits if "halo" in o or "Lyapunov" in o]
    for j, o in enumerate(unstable):
        st = ORBIT_STYLE[o]
        med, lo, hi = [], [], []
        for c in cfgs:
            v = np.array([min(b, 30.0) / a for a, b, r in _pairs(rows, c)
                          if r["orbit"] == o and np.isfinite(a) and a < 30])
            med.append(np.median(v)), lo.append(np.percentile(v, 10)), hi.append(np.percentile(v, 90))
        axes[1].errorbar(x + (j - 0.5) * 0.3, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], ls="none",
                         marker=st["marker"], color=st["color"], ms=3.5, elinewidth=0.7, capsize=0,
                         label=ORBIT_LABEL[o])
    axes[1].axhline(0.8, color="grey", ls=(0, (4, 2)), lw=0.6)
    axes[1].axhline(1.0, color="k", lw=0.5)
    axes[1].set_ylabel("operator $T_c$ / reference\n(N = 10)")
    axes[1].set_ylim(min(axes[1].get_ylim()[0], 0.75), None)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS.get(c, c) for c in cfgs], rotation=45, ha="right")
    for ax, letter in zip(axes, "ab"):
        ax.grid(True, axis="y")
        panel_label(ax, letter)
    axes[0].legend(loc="upper left", ncol=4)
    axes[1].legend(loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "fig20_consider_horizon", ROOT)


def load_rows():
    rows = []
    for r in csv.DictReader(open(OUT)):
        rows.append({k: (v if k in ("orbit", "config", "ref") else float(v)) for k, v in r.items()})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-days", type=float, default=30.0, help="spacing of arc ends (custody_study uses 15)")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), help="subset of error configurations")
    ap.add_argument("--orbits", nargs="+", help="subset of orbits, e.g. L1 L2")
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    if a.plots_only:
        r_ = load_rows()
        report(r_)
        plots(r_)
    else:
        main(a.step_days, a.samples, a.jobs, a.configs, a.orbits)
