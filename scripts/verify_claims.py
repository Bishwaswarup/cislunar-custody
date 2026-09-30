"""Check every quantitative claim in the manuscript against the saved study outputs.

    python scripts/verify_claims.py                 # uses data/*.csv and paper/main.tex
    python scripts/verify_claims.py --only FAIL     # show only claims that need attention

For every claim it prints the value written in the paper, the value recomputed from
data/, and a verdict:
    PASS   the paper value matches the data (to the rounding used in the paper)
    FAIL   the data do not support the number as written -> change the paper
    WEAK   the number is right, but the evidence is statistically thin (small n, wide
           confidence interval, or a difference that is not significant) -> soften wording
    TEXT   the paper no longer contains the checked sentence (verifier or paper changed)
    SKIP   the input file is missing (run the corresponding study first)
    MANUAL cannot be checked from saved data (see the note)
A copy of the report is written to data/claims_report.txt.

Run order: build_catalogue -> visibility_study -> observability_study -> filter_study ->
nonlinear_filter_study -> custody_study -> ephemeris_check -> sensitivity_check -> this.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
np.seterr(invalid="ignore")
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"

ORB = {"NRHO": "NRHO 9:2", "L1": "L1 halo (Az~30k km)", "L2": "L2 Lyapunov (mid)", "DRO": "DRO (~70k km)"}
RESULTS = []
UT_MIN = (66, 72)      # paper: minima of the median UT containment curves (Sec 8)


# ----------------------------------------------------------------------------- helpers
def load(name):
    p = DATA / name
    if not p.exists():
        return None
    rows = []
    for r in csv.DictReader(open(p)):
        out = {}
        for k, v in r.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        rows.append(out)
    return rows


def col(rows, key, **where):
    return np.array([r[key] for r in rows if all(r[k] == v for k, v in where.items())], float)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, c - h), min(1.0, c + h)


def r2log(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a < 30) & (b < 30)
    if ok.sum() < 3:
        return np.nan, int(ok.sum()), (np.nan, np.nan)
    la, lb = np.log(a[ok]), np.log(b[ok])
    f = lambda x, y: 1 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, ok.sum(), ok.sum())
        if np.ptp(la[i]) > 0:
            boots.append(f(la[i], lb[i]))
    return f(la, lb), int(ok.sum()), tuple(np.percentile(boots, [2.5, 97.5]))


def distinct(rows):
    """Drop repeated arcs: a phase-scenario arc often ends exactly where a blackout starts, so
    the same arc can appear in both scenarios. Pooled statistics use each arc once."""
    seen, out = set(), []
    for r in rows:
        k = (r["orbit"], r["arc_d"], round(r["t_end_day"], 6))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def cap30(x):
    return np.minimum(np.asarray(x, float), 30.0)


def fmt_tc(x):
    return ">30" if x >= 30 else f"{x:.1f}"


class Claim:
    def __init__(self, cid, where, text, paper, tex=None):
        self.cid, self.where, self.text, self.paper, self.tex = cid, where, text, paper, tex


def record(claim, verdict, computed, note=""):
    RESULTS.append((claim, verdict, computed, note))


def close(paper, value, tol):
    return abs(float(paper) - float(value)) <= tol + 1e-9


def check_num(c, value, tol, weak=None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return record(c, "FAIL", "n/a", "not computable from data")
    ok = close(c.paper, value, tol)
    record(c, ("WEAK" if weak else "PASS") if ok else "FAIL", f"{value:.4g}", weak or "")


def check_range(c, lo, hi, tol, weak=None):
    plo, phi = c.paper
    ok = close(plo, lo, tol) and close(phi, hi, tol)
    record(c, ("WEAK" if weak else "PASS") if ok else "FAIL", f"{lo:.4g}-{hi:.4g}", weak or "")


def check_bool(c, ok, computed, note=""):
    record(c, "PASS" if ok else "FAIL", computed, note)


def skip(c, what):
    record(c, "SKIP", "-", f"missing {what}")


# ----------------------------------------------------------------------------- claims
def claims_models():
    p = DATA / "named_orbits.json"
    cs = [Claim("M1", "Sec 3.1", "9:2 NRHO period [d]", 6.5624, "6.5624"),
          Claim("M2", "Sec 3.1", "NRHO perilune radius [km]", 3249, "3249"),
          Claim("M3", "Sec 3.1", "NRHO apolune radius [km]", 71222, "71\\,222"),
          Claim("M4", "Sec 3.1", "NRHO stability index", 1.32, "1.32")]
    if not p.exists():
        return [skip(c, p.name) for c in cs]
    n = json.load(open(p))["NRHO_9:2"]
    check_num(cs[0], n["period_days"], 5e-5)
    check_num(cs[1], n["perilune_km"], 0.5)
    check_num(cs[2], n["apolune_km"], 0.5)
    check_num(cs[3], n["stability_index"], 0.005)
    cat = DATA / "catalogue.npz"
    fam = {"L1_halo_north": 88, "L2_halo_south": 140, "L1_lyapunov": 60, "L2_lyapunov": 60, "DRO": 94}
    if cat.exists():
        d = np.load(cat)
        for i, (k, v) in enumerate(fam.items()):
            c = Claim(f"M5{'abcde'[i]}", "Sec 3.1", f"family size {k}", v, f"({v})" if "lyapunov" not in k else "60 each")
            check_num(c, float(len(d[f"{k}__periods"])), 0.0)
    try:
        from cislunar_custody.sensors.photometry import apparent_magnitude
        from cislunar_custody.constants import LU_KM
        r_obj = np.array([[LU_KM, 0.0, 0.0]])
        m = apparent_magnitude(r_obj, np.zeros((1, 3)), np.array([[-1.496e8, 0.0, 0.0]]), 1.0, 0.2)
        check_num(Claim("M6", "Sec 3.3", "target magnitude at zero phase, lunar distance", 18.4, "18.4"),
                  float(np.atleast_1d(m)[0]), 0.05)
    except Exception as e:  # noqa: BLE001
        record(Claim("M6", "Sec 3.3", "target magnitude at zero phase", 18.4, "18.4"), "MANUAL", "-", str(e))


def claims_visibility():
    v = load("visibility_summary.csv")
    if v is None:
        return skip(Claim("V", "Sec 5", "visibility claims", None), "visibility_summary.csv")
    g = lambda o, e, t="1m", n="Tri-3+S", k="frac_visible": next(
        r[k] for r in v if r["orbit"] == ORB[o] and r["moon_excl_deg"] == e and r["telescope"] == t and r["network"] == n)
    table = {  # Table 1 (1 m, Tri-3+S): visible %, max gap d
        ("NRHO", 2): (37.5, 12.8), ("NRHO", 5): (31.4, 13.2), ("NRHO", 10): (4.5, 19.5),
        ("L1", 2): (42.7, 11.2), ("L1", 5): (32.6, 11.8), ("L1", 10): (0.0, None),
        ("L2", 2): (32.7, 13.5), ("L2", 5): (20.6, 14.5), ("L2", 10): (0.0, None),
        ("DRO", 2): (38.0, 11.8), ("DRO", 5): (32.5, 11.9), ("DRO", 10): (21.2, 16.5)}
    for (o, e), (fv, mg) in table.items():
        check_num(Claim(f"V1.{o}.{e}", "Table 1", f"{o} visible % at {e} deg", fv), 100 * g(o, e), 0.05)
        if mg is not None:
            check_num(Claim(f"V2.{o}.{e}", "Table 1", f"{o} max gap [d] at {e} deg", mg), g(o, e, k="max_gap_d"), 0.05)
    never = all(r["frac_visible"] == 0 for r in v if r["orbit"] in (ORB["L1"], ORB["L2"]) and r["moon_excl_deg"] == 10)
    check_bool(Claim("V3", "Sec 5", "L1/L2 never observable at 10 deg (all telescopes and networks)", True,
                     "never observable"), never, str(never))
    lo, hi = 100 * min(g("L1", 5), g("L2", 5)), 100 * max(g("L1", 5), g("L2", 5))
    check_range(Claim("V4", "Sec 5", "L1/L2 observable share at 5 deg [%]", (21, 33), "21--33\\,\\%"), lo, hi, 0.5)
    d = [100 * (g(o, 5, "2m") - g(o, 5, "1m")) for o in ORB]
    check_range(Claim("V5", "Sec 5", "1 m -> 2 m change in visible share, Tri-3+S, 5 deg [pp]", (0.5, 1.7),
                      "0.5--1.7 percentage"), min(d), max(d), 0.05)
    mg = [r["max_gap_d"] for r in v if r["moon_excl_deg"] == 5 and r["frac_visible"] > 0 and r["telescope"] != "0.36m"]
    check_range(Claim("V6", "Sec 5 / abstract", "longest gap at 5 deg, 1-2 m, all networks and orbits [d]",
                      (8.5, 15.7), "8.5--15.7"), min(mg), max(mg), 0.05)
    try:
        from cislunar_custody.frames import Ephemeris
        from cislunar_custody.constants import MU_EM
        from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd
        sys.path.insert(0, str(ROOT / "scripts"))
        from visibility_study import representative_orbits, EPOCH, STEP_MIN, geocentric_moon_separation_deg
        jd0 = jd_from_iso(EPOCH)
        jd = jd_grid(jd0, 365, STEP_MIN)
        eph = Ephemeris(jd)
        t = tu_from_jd(jd, jd0)
        seps = {k: geocentric_moon_separation_deg(eph, eph.to_eci(o.states_at(t), MU_EM))
                for k, o in zip(ORB, representative_orbits().values())}
        paper = {"L1": (3.9, 6.1), "L2": (0.0, 7.0), "NRHO": (0.5, 10.0), "DRO": (0.0, 13.9)}
        for o, pr in paper.items():
            check_range(Claim(f"V7.{o}", "Sec 5", f"{o} Moon separation range [deg]", pr), seps[o].min(), seps[o].max(), 0.05)
        check_num(Claim("V8", "Sec 5", "NRHO median Moon separation [deg]", 8.3, "median 8.3"), np.median(seps["NRHO"]), 0.05)
        allmax = max(s.max() for s in seps.values())
        check_bool(Claim("V9", "abstract", "all targets within 14 deg of the Moon", True, "within 14"),
                   allmax < 14.0, f"max {allmax:.2f} deg")
    except Exception as e:  # noqa: BLE001
        record(Claim("V7", "Sec 5", "Moon separation ranges", None), "MANUAL", "-", f"recompute failed: {e}")


def claims_observability():
    a = load("observability_arcs.csv")
    if a is None:
        return skip(Claim("O", "Sec 6", "CRLB claims", None), "observability_arcs.csv")
    paper = {"NRHO": 1.12, "L1": 1.74, "DRO": 6.87, "L2": 17.1}
    vel, frac, weak = [], [], {}
    for o, pv in paper.items():
        s = col(a, "sig_pos_max_km", orbit=ORB[o], arc_d=7.0)
        check_num(Claim(f"O1.{o}", "Sec 6 / abstract", f"{o} median CRLB, 7-d arcs [km]", pv), np.nanmedian(s),
                  0.005 if pv < 10 else 0.05)
        vel.append(np.nanmedian(col(a, "sig_vel_max_ms", orbit=ORB[o], arc_d=7.0)) * 1e3)
        obs = [r["observable"] in (True, "True", 1.0) for r in a if r["orbit"] == ORB[o] and r["arc_d"] == 7.0]
        frac.append(100 * np.mean(obs))
        weak[o] = np.nanmedian(col(a, "weak_los_deg", orbit=ORB[o], arc_d=7.0))
    check_range(Claim("O2", "Sec 6", "median velocity CRLB, 7-d arcs [mm/s]", (6, 101), "6--101"), min(vel), max(vel), 0.5)
    check_range(Claim("O3", "Sec 6", "share of 7-d arcs that are observable [%]", (75, 84), "75\\,\\% and 84"),
                min(frac), max(frac), 0.5)
    check_range(Claim("O4", "Sec 6", "median weak-direction angle to LOS, DRO & L2 [deg]", (6.6, 7.8), "6.6--7.8"),
                min(weak["DRO"], weak["L2"]), max(weak["DRO"], weak["L2"]), 0.05)
    check_range(Claim("O5", "Sec 6", "median weak-direction angle, NRHO & L1 [deg]", (12.7, 18.3), "12.7--18.3"),
                min(weak["NRHO"], weak["L1"]), max(weak["NRHO"], weak["L1"]), 0.05)


def claims_filters():
    f = load("filter_summary.csv")
    if f is None:
        return skip(Claim("F", "Sec 6", "tracking claims", None), "filter_summary.csv")
    from scipy.stats import chi2
    tab = {"NRHO": (1.13, 1.10, 4.51, 1.08, 4.40), "L1": (1.74, 2.33, 9.12, 2.10, 7.66),
           "L2": (17.60, 15.25, 6.35, 15.15, 6.31), "DRO": (6.95, 6.88, 6.23, 6.93, 6.25)}
    for o, (crlb, ee, ea, ue, ua) in tab.items():
        r = {x["filter"]: x for x in f if x["orbit"] == ORB[o] and x["mode"] == "track"}
        check_num(Claim(f"F1.{o}.crlb", "Table 2", f"{o} CRLB of chosen arc", crlb), r["EKF"]["crlb_sig_pos_max_km"], 0.005)
        check_num(Claim(f"F1.{o}.ekf", "Table 2", f"{o} EKF RMS error", ee), r["EKF"]["final_pos_err_rms_km"], 0.005)
        check_num(Claim(f"F1.{o}.ekfa", "Table 2", f"{o} EKF ANEES", ea), r["EKF"]["anees_2nd_half"], 0.005)
        check_num(Claim(f"F1.{o}.ukf", "Table 2", f"{o} UKF RMS error", ue), r["UKF"]["final_pos_err_rms_km"], 0.005)
        check_num(Claim(f"F1.{o}.ukfa", "Table 2", f"{o} UKF ANEES", ua), r["UKF"]["anees_2nd_half"], 0.005)
    n_runs = 20
    lo, hi = chi2.ppf([0.025, 0.975], 6 * n_runs) / n_runs
    bad = [(o, k) for o in ORB for x in f if x["orbit"] == ORB[o] and x["mode"] == "track"
           for k in [x["filter"]] if not (lo < x["anees_2nd_half"] < hi)]
    record(Claim("F2", "Sec 6", "tracking ANEES outside the 95% band of 20 runs "
                                f"[{lo:.2f}, {hi:.2f}]: only NRHO (both, conservative) and L1 (both)",
                 "NRHO,L1", "lies outside the 95\\,\\% band"),
           "PASS" if sorted({o for o, _ in bad}) == ["L1", "NRHO"] else "FAIL", ", ".join(f"{o}/{k}" for o, k in bad))


def claims_reacquisition():
    s = load("nonlinear_filter_summary.csv")
    if s is None:
        return skip(Claim("R", "Sec 6", "reacquisition claims", None), "nonlinear_filter_summary.csv")
    r = {x["filter"]: x for x in s if x["orbit"] == ORB["L1"]}
    n = int(r["EKF"]["runs"])
    k = {f: int(round(x["runs_consistent"] * x["runs"])) for f, x in r.items()}
    paper = {"EKF": 25, "UKF": 55, "GM-UKF": 99, "PF→UKF": 91}
    for f, pv in paper.items():
        lo, hi = wilson(k[f], n)
        ci = f"n={n}, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%"
        weak = f"only {n} runs; {ci}" if n < 50 else ""
        c = Claim(f"R1.{f}", "Table 2 / abstract", f"L1 reacquisition: {f} runs consistent [%]", pv)
        check_num(c, 100 * k[f] / n, 0.5, weak=weak or None)
        if not weak:
            RESULTS[-1] = (c, RESULTS[-1][1], RESULTS[-1][2], ci)
    for a_, b_ in (("GM-UKF", "UKF"), ("GM-UKF", "EKF"), ("UKF", "EKF"), ("GM-UKF", "PF→UKF")):
        p = fisher_exact([[k[a_], n - k[a_]], [k[b_], n - k[b_]]])[1]
        c = Claim(f"R2.{a_}>{b_}", "Sec 6 / Discussion", f"{a_} reacquires more often than {b_} (Fisher exact test)",
                  "p<=0.02", "p \\le 0.02" if a_ == "GM-UKF" else None)
        record(c, "PASS" if (p <= 0.02 and k[a_] > k[b_]) else "WEAK", f"{k[a_]}/{n} vs {k[b_]}/{n}, p={p:.3g}",
               "" if p < 0.05 else "difference not significant: do not claim it, or add runs")
    med = {"EKF": 12.6, "UKF": 3.82, "GM-UKF": 1.50, "PF→UKF": 1.5}   # PF: resampling is platform-sensitive
    mx = {"EKF": 561.8, "UKF": 241.9, "GM-UKF": 4.2, "PF→UKF": 18.5}
    an = {"EKF": 7.5e3, "UKF": 1.0e3, "GM-UKF": 7.0, "PF→UKF": 21.2}
    for fl in paper:
        check_num(Claim(f"R3.{fl}.med", "Table 2", f"{fl} median final error [km]", med[fl]),
                  r[fl]["median_pos_err_km"], 0.05 if fl == "PF→UKF" or med[fl] >= 10 else 0.005)
        check_num(Claim(f"R3.{fl}.max", "Table 2", f"{fl} max final error [km]", mx[fl]), r[fl]["max_pos_err_km"], 0.05)
        tol = 0.05 * an[fl] if an[fl] > 100 else 0.05
        check_num(Claim(f"R3.{fl}.anees", "Table 2", f"{fl} ANEES", an[fl]), r[fl]["anees_2nd_half"], tol)
    check_num(Claim("R4", "Sec 6", "GM-UKF max components (L1)", 27, "up to 27"), r["GM-UKF"]["max_components"], 0)
    oth = [(o, f_) for o in ("NRHO", "L2", "DRO") for f_ in ("EKF", "UKF", "GM-UKF")]
    v = [100 * next(x for x in s if x["orbit"] == ORB[o] and x["filter"] == f_)["runs_consistent"] for o, f_ in oth]
    check_range(Claim("R9", "Sec 6", "other orbits: EKF/UKF/GM-UKF runs consistent [%]", (99, 100), "99--100"),
                min(v), max(v), 0.5)
    v = [100 * next(x for x in s if x["orbit"] == ORB[o] and x["filter"] == "PF→UKF")["runs_consistent"]
         for o in ("NRHO", "L2", "DRO")]
    check_range(Claim("R10", "Sec 6", "other orbits: PF→UKF runs consistent [%]", (96, 98), "(96--98"),
                min(v), max(v), 0.5)
    check_num(Claim("R5", "Sec 6", "GM-UKF mean components (L1)", 3.1, "averaging 3.1"), r["GM-UKF"]["mean_components"], 0.05)
    t = {f: x["sec_per_run"] for f, x in r.items()}
    ratio = t["GM-UKF"] / t["UKF"]
    check_bool(Claim("R6", "Discussion", "GM-UKF costs about twice a UKF (1.5-2.5x)", True, "about twice"),
               1.5 <= ratio <= 2.5, f"{ratio:.2f}x", "wall-clock, machine dependent")
    pc = load("nonlinear_prior_containment.csv")
    if pc is None:
        record(Claim("R7", "Sec 6", "prior containment 96-98% and cloud span", None), "SKIP", "-",
               "missing nonlinear_prior_containment.csv (re-run nonlinear_filter_study.py)")
    else:
        x = next(p for p in pc if p["orbit"] == ORB["L1"])
        v = [100 * x[f"inside99_{f}"] for f in ("EKF", "UKF", "GM-UKF")]
        check_range(Claim("R7", "Sec 6", "PF particles inside Gaussian 99% ellipsoids at first measurement [%]",
                          (96, 98), "96--98"), min(v), max(v), 0.5)
        check_range(Claim("R8", "Sec 6", "L1 prior cloud 1-99% span [10^3 km]", (31, 10), "31\\,000 \\times 10\\,000"),
                    x["span1_km"] / 1e3, x["span2_km"] / 1e3, 0.5)


def claims_custody():
    c = load("custody_cases.csv")
    if c is None:
        return skip(Claim("K", "Sec 7", "custody claims", None), "custody_cases.csv")
    ph = [r for r in c if r["scenario"] == "phase"]
    bl = [r for r in c if r["scenario"] == "blackout"]
    check_num(Claim("K0", "Sec 4", "number of custody cases", 525, "525 cases"), float(len(c)), 0)
    cu = distinct(c)
    check_num(Claim("K0b", "Sec 4", "number of distinct arcs among the custody cases", 394, "394 are distinct"),
              float(len(cu)), 0)
    tab3 = {  # Table 3: (orbit, arc): sigma0, {N: (p50, p10, p90)}
        ("L1", 1): (20.0, {1: (8.8, 5.5, 9.9), 10: (11.0, 7.4, 12.4), 100: (12.8, 9.5, 13.8)}),
        ("L1", 3): (5.1, {1: (11.7, 8.0, 12.4), 10: (13.2, 10.1, 14.1), 100: (14.9, 13.5, 17.4)}),
        ("L1", 7): (0.7, {1: (14.3, 13.1, 15.3), 10: (16.5, 14.7, 18.4), 100: (18.5, 16.3, 20.0)}),
        ("L2", 1): (30.2, {1: (9.5, 7.7, 11.2), 10: (12.5, 9.9, 15.3), 100: (16.2, 12.9, 18.2)}),
        ("L2", 3): (15.5, {1: (10.3, 9.1, 13.4), 10: (14.9, 11.2, 16.5), 100: (18.3, 15.6, 19.0)}),
        ("L2", 7): (7.0, {1: (13.9, 11.9, 15.8), 10: (17.4, 15.5, 18.3), 100: (19.2, 18.4, 21.4)}),
        ("NRHO", 1): (23.8, {1: (30, 5.3, 30), 10: (30, 18.4, 30), 100: (30, 30, 30)}),
        ("NRHO", 3): (3.9, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("NRHO", 7): (0.9, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("DRO", 1): (31.8, {1: (30, 7.8, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("DRO", 3): (11.7, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("DRO", 7): (1.4, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
    }
    for (o, arc), (s0, cells) in tab3.items():
        sel = [r for r in ph if r["orbit"] == ORB[o] and r["arc_d"] == arc]
        check_num(Claim(f"K1.{o}.{arc}.s0", "Table 3", f"{o} {arc}-d median sigma0 [km]", s0),
                  np.median([r["sig0_pos_km"] for r in sel]), 0.05)
        for N, (p50, p10, p90) in cells.items():
            v = cap30([r[f"tc_ideal_N{N}"] for r in sel])
            q10, q50, q90 = np.percentile(v, [10, 50, 90])
            ok = all(close(p, q, 0.05) or (p >= 30 and q >= 30) for p, q in ((p50, q50), (p10, q10), (p90, q90)))
            record(Claim(f"K1.{o}.{arc}.N{N}", "Table 3", f"{o} {arc}-d T_c(N={N}) median (p10-p90)",
                         f"{fmt_tc(p50)} ({fmt_tc(p10)}-{fmt_tc(p90)})"),
                   "PASS" if ok else "FAIL", f"{fmt_tc(q50)} ({fmt_tc(q10)}-{fmt_tc(q90)})", f"n={len(sel)}")
    # growth / text claims
    cur = DATA / "custody_curves.npz"
    if cur.exists():
        d = np.load(cur)
        th = {str(d[f"name_{i}"]): d[f"theta_ideal_{i}"] for i in range(4)}
        mx = max(np.median(th[ORB[o]], axis=0)[-1] for o in ("NRHO", "DRO"))
        check_bool(Claim("K2", "Sec 7", "NRHO & DRO median 99% sky radius < 0.01 deg at 30 d (7-d arcs)", True,
                         "0.01"), mx < 0.01, f"{mx:.4f} deg")
        ci = {str(d[f"name_{i}"]): d[f"contain_ut_{i}"] for i in range(4)}
        cl = {str(d[f"name_{i}"]): d[f"contain_lin_{i}"] for i in range(4)}
        dt = d["dt_days"]
        first = lambda y: float(dt[np.argmax(y < 0.95)]) if np.any(y < 0.95) else np.inf
        umed = {o: np.median(ci[ORB[o]], axis=0) for o in ("L1", "L2")}
        lmed = {o: np.median(cl[ORB[o]], axis=0) for o in ("L1", "L2")}
        mins = [100 * umed[o].min() for o in ("L1", "L2")]
        check_range(Claim("K3", "Sec 8", "minimum median UT containment, L1 & L2 (7-d arcs) [%]", UT_MIN,
                          f"minima of {UT_MIN[0]}--{UT_MIN[1]}"), min(mins), max(mins), 0.5)
        tu = [first(umed[o]) for o in ("L1", "L2")]
        check_range(Claim("K3b", "Sec 8", "gap at which median UT containment first drops below 95% [d]", (18.6, 19.5),
                          "18.6--19.5 days"), min(tu), max(tu), 0.5)
        check_num(Claim("K13a", "Sec 8 / abstract", "L1: gap where median linear containment first < 95% [d]", 9.5,
                        "of 9.5\\,d"), first(lmed["L1"]), 0.05)
        check_num(Claim("K13b", "Sec 8", "L2: gap where median linear containment first < 95% [d]", 18.6,
                        "at 18.6\\,d"), first(lmed["L2"]), 0.05)
        l14 = 100 * np.median([r["contain_lin_14d"] for r in cu if r["orbit"] == ORB["L1"]])
        u14 = 100 * np.median([r["contain_ut_14d"] for r in cu if r["orbit"] == ORB["L1"]])
        check_num(Claim("K4", "Sec 8", "L1 linear 99% ellipse containment at 14 d, median over all L1 arcs [%]",
                        5.4, "only 5.4"), l14, 0.05)
        check_num(Claim("K5", "Sec 8", "L1 UT containment at 14 d, median over all L1 arcs [%]", 98,
                        "contains 98"), u14, 0.5)
    else:
        record(Claim("K2", "Sec 7", "growth-curve claims", None), "SKIP", "-", "missing custody_curves.npz")
    med = lambda o, a, N: np.median(cap30([r[f"tc_ideal_N{N}"] for r in ph if r["orbit"] == ORB[o] and r["arc_d"] == a]))
    gains = [med(o, a, 100) - med(o, a, 10) for o in ("L1", "L2") for a in (1, 3, 7)]
    check_range(Claim("K6", "Sec 7", "gain from 10 -> 100 fields, unstable orbits [d]", (1.7, 3.6), "1.7--3.6"),
                min(gains), max(gains), 0.05)
    t7 = [med(o, 7, N) for o in ("L1", "L2") for N in (1, 10, 100)]
    check_range(Claim("K7", "Sec 7", "7-d arcs: median T_c for N=1..100, unstable [d]", (13.9, 19.2), "13.9--19.2"),
                min(t7), max(t7), 0.05)
    t10 = [med(o, a, 10) for o in ("L1", "L2") for a in (1, 3, 7)]
    check_range(Claim("K8", "abstract / Sec 10", "median T_c(N=10), unstable orbits, all arcs [d]", (11.0, 17.4),
                      "11.0--17.4"), min(t10), max(t10), 0.05)
    # Table 4 blackouts
    tab4 = {("NRHO", 1): (9, 10.5, 67, 78, 78, 78), ("NRHO", 3): (11, 10.5, 100, 100, 100, 100),
            ("NRHO", 7): (11, 10.5, 100, 100, 100, 100),
            ("L1", 1): (27, 3.1, 67, 89, 93, 89), ("L1", 3): (27, 3.1, 93, 96, 96, 96), ("L1", 7): (28, 3.1, 100, 100, 100, 100),
            ("L2", 1): (34, 4.1, 65, 74, 94, 74), ("L2", 3): (34, 4.1, 79, 97, 97, 97), ("L2", 7): (34, 4.1, 91, 100, 100, 100),
            ("DRO", 1): (13, 11.3, 46, 85, 85, 85), ("DRO", 3): (13, 11.3, 100, 100, 100, 100),
            ("DRO", 7): (13, 11.3, 100, 100, 100, 100)}
    surv = {}
    for (o, a), (n, ml, s1, s10, s100, sut) in tab4.items():
        sel = [r for r in bl if r["orbit"] == ORB[o] and r["arc_d"] == a]
        L = np.array([r["blackout_d"] for r in sel])
        got = [len(sel), np.median(L)] + [100 * np.mean(np.array([r[f"tc_ideal_N{N}"] for r in sel]) >= L)
                                            for N in (1, 10, 100)]
        got.append(100 * np.mean(np.array([r["tc_ut_actual_N10"] for r in sel]) >= L))
        surv[(o, a)] = got[3]
        paper = [n, ml, s1, s10, s100, sut]
        ok = all(close(p, g, t) for p, g, t in zip(paper, got, (0, 0.05, 0.5, 0.5, 0.5, 0.5)))
        kN = int(round(got[3] * len(sel) / 100))
        lo, hi = wilson(kN, len(sel))
        record(Claim(f"K9.{o}.{a}", "Table 4", f"{o} {a}-d blackouts: n, median length, survived N=1/10/100/UT",
                     " / ".join(f"{x:g}" for x in paper)),
               "PASS" if ok else "FAIL", " / ".join(f"{x:.3g}" for x in got),
               f"N=10 survival 95% CI {100 * lo:.0f}-{100 * hi:.0f}%")
    for a, pr in ((1, (74, 89)), (3, (96, 100)), (7, (100, 100))):
        v = [surv[(o, a)] for o in ORB]
        check_range(Claim(f"K10.{a}", "abstract", f"blackouts survived (N=10) after {a}-d arcs, all orbits [%]", pr),
                    min(v), max(v), 0.5)
    # false custody
    for g in ("lin", "ut"):
        cl_ = np.array([r[f"tc_{g}_claim_N10"] for r in cu])
        ac_ = np.array([r[f"tc_{g}_actual_N10"] for r in cu])
        fin = np.isfinite(ac_)
        share = 100 * np.mean(cl_[fin] > 1.1 * ac_[fin])
        if g == "ut":
            k = int(np.sum(cl_[fin] > 1.1 * ac_[fin]))
            check_num(Claim("K11", "Sec 8", "distinct arcs with UT false custody at N=10 (claim > 1.1 x actual)", 2,
                            "in 2 of the 254"), float(k), 0)
            RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], f"{share:.1f}% of {fin.sum()} arcs")
            check_num(Claim("K11b", "Sec 8", "distinct arcs with a finite UT-centred horizon at N=10", 254,
                            "in 2 of the 254"), float(fin.sum()), 0)
        else:
            record(Claim("K12", "info", "linear false-custody share at N=10 (not in paper)", "-"), "PASS",
                   f"{share:.1f}% of {fin.sum()} cases", "information only")
    # predictors
    paper_r2 = {"ftle": {1: None, 10: None, 100: None}, "ut_claim": {1: 0.96, 10: 1.00, 100: 0.93},
                "lin_claim": {1: 0.96, 10: 0.99, 100: 0.97}}
    ftle, ns = [], []
    for N in (1, 10, 100):
        a_ = np.array([r[f"tc_ideal_N{N}"] for r in cu])
        for m in ("ftle", "ut_claim", "lin_claim"):
            r2, n, (blo, bhi) = r2log(a_, np.array([r[f"tc_{m}_N{N}"] for r in cu]))
            ns.append(n)
            if m == "ftle":
                ftle.append(r2)
                continue
            weak = f"n={n}; bootstrap 95% CI {blo:.2f}-{bhi:.2f}" if (bhi - blo) > 0.1 or n < 30 else None
            check_num(Claim(f"P1.{m}.N{N}", "Sec 8 / abstract", f"log-R2 of {m} predictor, N={N}", paper_r2[m][N]), r2,
                      0.005, weak=weak)
            if not weak:
                RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], f"n={n}; CI {blo:.2f}-{bhi:.2f}")
    check_range(Claim("P2", "Sec 8", "log-R2 of orbit-only FTLE predictor, N=1..100", (-0.23, 0.04), "-0.23"),
                min(ftle), max(ftle), 0.005)
    check_range(Claim("P3", "Sec 8", "number of distinct arcs used for the R2 values", (251, 269), "251--269"),
                min(ns), max(ns), 0)


def claims_ephemeris():
    cp = load("ephemeris_counterparts.csv")
    e = load("ephemeris_check.csv")
    if cp is None or e is None:
        return skip(Claim("E", "Sec 9", "ephemeris claims", None), "ephemeris_check.csv / ephemeris_counterparts.csv")
    g = {r["orbit"]: r for r in cp}
    others = [g[ORB[o]] for o in ("L1", "L2", "DRO")]
    check_bool(Claim("E1", "Sec 9", "L1/L2/DRO counterparts continuous to < 0.4 m", True, "0.4\\,m"),
               max(r["max_continuity_km"] for r in others) < 4e-4,
               f"max {1e3 * max(r['max_continuity_km'] for r in others):.3g} m")
    dev = [r["median_dev_km"] for r in others]
    check_range(Claim("E2", "Sec 9", "median deviation from CR3BP shape, L1/L2/DRO [km]", (3500, 5200), "3500--5200"),
                min(dev), max(dev), 150)
    check_range(Claim("E3", "Sec 9", "NRHO continuity residual median-max [km]", (50, 124), "50--124"),
                g[ORB["NRHO"]]["median_continuity_km"], g[ORB["NRHO"]]["max_continuity_km"], 0.6)
    check_num(Claim("E4", "Sec 9", "number of ephemeris cases", 60, "60 custody cases"), float(len(e)), 0)
    tab5 = {("NRHO", 1): (8, 30, 28.3, 0.98), ("NRHO", 7): (7, 30, 30, None), ("L1", 1): (4, 8.5, 7.8, 0.93),
            ("L1", 7): (11, 15.9, 15.9, 1.01), ("L2", 1): (6, 12.5, 11.3, 0.94), ("L2", 7): (9, 15.9, 16.3, 0.99),
            ("DRO", 1): (6, 30, 30, 0.99), ("DRO", 7): (9, 30, 30, None)}
    ratios = []
    for (o, a), (n, tcc, tce, ratio) in tab5.items():
        sel = [r for r in e if r["orbit"] == ORB[o] and r["arc_d"] == a]
        cc = np.array([r["tc_cr3bp_ideal_N10"] for r in sel])
        ee = np.array([r["tc_ephem_ideal_N10"] for r in sel])
        fin = np.isfinite(cc) & np.isfinite(ee) & (cc > 0)
        rat = np.median(ee[fin] / cc[fin]) if fin.any() else None
        if rat is not None:
            ratios.append(rat)
        got = [len(sel), np.median(cap30(cc)), np.median(cap30(ee))]
        ok = close(n, got[0], 0) and all(close(p, q, 0.05) or (p >= 30 and q >= 30) for p, q in zip((tcc, tce), got[1:]))
        ok = ok and ((ratio is None and rat is None) or (ratio is not None and rat is not None and close(ratio, rat, 0.005)))
        weak = f"only n={len(sel)} cases" if len(sel) < 6 else ""
        record(Claim(f"E5.{o}.{a}", "Table 5", f"{o} {a}-d: n, T_c CR3BP, T_c ephem, ratio",
                     f"{n} / {fmt_tc(tcc)} / {fmt_tc(tce)} / {ratio}"),
               ("WEAK" if weak else "PASS") if ok else "FAIL",
               f"{got[0]} / {fmt_tc(got[1])} / {fmt_tc(got[2])} / {'-' if rat is None else f'{rat:.2f}'}", weak)
    check_range(Claim("E6", "abstract / Sec 9", "range of median ephemeris/CR3BP T_c ratios", (0.93, 1.01), "0.93--1.01"),
                min(ratios), max(ratios), 0.005)
    eu = distinct(e)
    check_num(Claim("E4b", "Sec 9", "distinct arcs among the ephemeris cases", 59, "59 distinct"), float(len(eu)), 0)
    cc = np.array([r["tc_cr3bp_ideal_N10"] for r in eu])
    ee = np.array([r["tc_ephem_ideal_N10"] for r in eu])
    fin = np.isfinite(cc) & np.isfinite(ee)
    n_out = int(np.sum(np.abs(cc[fin] - ee[fin]) > 3.0) + np.sum(np.isfinite(cc) != np.isfinite(ee)))
    check_num(Claim("E7", "Sec 9", "distinct arcs whose T_c(N=10) changes by > 3 d (or crosses 30 d)", 6, "6 of the 59"),
              n_out, 0)
    diffs = []
    for o in ORB:
        for a in (1.0, 7.0):
            sel = [r for r in e if r["orbit"] == ORB[o] and r["scenario"] == "blackout" and r["arc_d"] == a]
            if not sel:
                continue
            L = np.array([r["blackout_d"] for r in sel])
            sc = np.sum(np.array([r["tc_cr3bp_ideal_N10"] for r in sel]) >= L)
            se = np.sum(np.array([r["tc_ephem_ideal_N10"] for r in sel]) >= L)
            if sc != se:
                diffs.append(f"{o} {a:.0f}d: {sc}/{len(sel)} -> {se}/{len(sel)}")
    check_num(Claim("E8", "Sec 9", "blackout groups whose survival count differs CR3BP vs ephemeris", 2,
                    "two single-case"), float(len(diffs)), 0)
    RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], "; ".join(diffs))
    for N, pv in ((1, 0.90), (10, 1.00), (100, 0.80)):
        a_ = np.array([r[f"tc_ephem_ideal_N{N}"] for r in eu])
        b_ = np.array([r[f"tc_ephem_ut_claim_N{N}"] for r in eu])
        r2, n, (blo, bhi) = r2log(a_, b_)
        weak = f"n={n}; bootstrap 95% CI {blo:.2f}-{bhi:.2f}" if (bhi - blo) > 0.1 or n < 30 else None
        check_num(Claim(f"E9.N{N}", "Sec 9", f"ephemeris UT log-R2, N={N}", pv), r2, 0.005, weak=weak)
    for o, key, pv in (("L1", "contain_lin_ephem_14d", 12.6), ("L1", "contain_lin_cr3bp_14d", 10.3),
                       ("L2", "contain_lin_ephem_14d", 84.8), ("L2", "contain_lin_cr3bp_14d", 98.6)):
        check_num(Claim(f"E10.{o}.{key}", "Sec 9", f"{o} {key} median [%]", pv),
                  100 * np.median([r[key] for r in e if r["orbit"] == ORB[o]]), 0.05)
    ut = [100 * np.median([r["contain_ut_ephem_14d"] for r in e if r["orbit"] == ORB[o]]) for o in ORB]
    check_range(Claim("E11", "Sec 9", "ephemeris UT containment at 14 d, all orbits [%]", (98, 99), "98--99"),
                min(ut), max(ut), 0.5)


def claims_sensitivity():
    s = load("sensitivity_samples.csv")
    c = Claim("S1", "Sec 10 limitations", "T_c converged in Monte Carlo sample size (300 vs larger sample)", None)
    if s is None:
        return record(c, "SKIP", "-", "run scripts/sensitivity_check.py")
    big = [k for k in s[0] if k.startswith("tc_n") and not k.startswith("tc_n300")][0].split("_")[1]
    notes = []
    worst = 0.0
    for N in (1, 10, 100):
        a = np.array([r[f"tc_n300_N{N}"] for r in s])
        b = np.array([r[f"tc_{big}_N{N}"] for r in s])
        fin = np.isfinite(a) & np.isfinite(b)
        rel = np.abs(a[fin] - b[fin]) / b[fin]
        worst = max(worst, np.median(rel))
        notes.append(f"N={N}: median |dT_c| {np.median(np.abs(a[fin] - b[fin])):.2f} d ({100 * np.median(rel):.1f}%), "
                     f"max {np.max(np.abs(a[fin] - b[fin])):.2f} d")
        rep = np.array([r[f"tc_csv_N{N}"] for r in s])
        if not np.all((np.isfinite(rep) == np.isfinite(a)) & (~np.isfinite(a) | (np.abs(rep - a) < 1e-6))):
            notes.append(f"N={N}: 300-sample rerun does NOT reproduce custody_cases.csv")
    record(c, "PASS" if worst < 0.05 else "WEAK", f"{big} samples, {len(s)} cases", "; ".join(notes))
    meds, mx10 = [], 0.0
    for N in (1, 10, 100):
        a = np.array([r[f"tc_n300_N{N}"] for r in s])
        b = np.array([r[f"tc_{big}_N{N}"] for r in s])
        fin = np.isfinite(a) & np.isfinite(b)
        meds.append(np.median(np.abs(a[fin] - b[fin])))
        if N >= 10:
            mx10 = max(mx10, np.max(np.abs(a[fin] - b[fin])))
    check_num(Claim("S2", "Sec 10", "arcs in the sample-size check", 17, "17 representative arcs"), float(len(s)), 0)
    check_range(Claim("S3", "Sec 10", "median |dT_c| 300 vs 2000 samples, N=1..100 [d]", (0.09, 0.14),
                      "0.09--0.14\\,d"), min(meds), max(meds), 0.005)
    check_num(Claim("S4", "Sec 10", "max |dT_c| for N=10 and 100 [d]", 0.9, "at most 0.9\\,d"), mx10, 0.05)
    a = np.array([r["tc_n300_N1"] for r in s])
    b = np.array([r[f"tc_{big}_N1"] for r in s])
    fin = np.isfinite(a) & np.isfinite(b)
    i = int(np.flatnonzero(fin)[np.argmax(np.abs(a[fin] - b[fin]))])
    w = s[i]
    check_range(Claim("S5", "Sec 10", "N=1 worst case: T_c with 300 and 2000 samples [d]", (5.3, 17.6),
                      "from 5.3 to 17.6\\,d"), a[i], b[i], 0.05)
    RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], f"{w['orbit']} {w['scenario']} {w['arc_d']:g}-d arc")
    per = {"NRHO 9:2": json.load(open(DATA / "named_orbits.json"))["NRHO_9:2"]["period_days"]}
    ph = np.mod(w["t_end_day"] + a[i], per.get(w["orbit"], np.inf))
    ph = min(ph, per.get(w["orbit"], np.inf) - ph)
    check_bool(Claim("S6", "Sec 10", "N=1 worst case is an NRHO arc whose 300-sample horizon ends at perilune (< 0.1 d)",
                     True, "perilune passage"), w["orbit"] == "NRHO 9:2" and ph < 0.1,
               f"{w['orbit']}, {ph:.3f} d from perilune")
    if "tc_uni_N1" in s[0]:
        gm, gx = [], []
        for N in (1, 10, 100):
            a = np.array([r[f"tc_n300_N{N}"] for r in s])
            b = np.array([r[f"tc_uni_N{N}"] for r in s])
            fin = np.isfinite(a) & np.isfinite(b)
            gm.append(np.median(np.abs(a[fin] - b[fin])))
            gx.append(np.max(np.abs(a[fin] - b[fin])))
        check_num(Claim("G1", "Sec 4", "grid check: max median |dT_c|, 121-point vs uniform 0.02-d grid [d]", 0.02,
                        "median of at most 0.02\\,d"), max(gm), 0.005)
        check_num(Claim("G2", "Sec 4", "grid check: max |dT_c| over all N, 121-point vs uniform grid [d]", 0.6,
                        "at most 0.6\\,d"), max(gx), 0.05)
        check_num(Claim("G3", "Sec 4", "grid check: max |dT_c| for N=1 and 10 [d]", 0.15, "at most 0.15\\,d for $N = 1$"),
                  max(gx[:2]), 0.05)
    else:
        record(Claim("G1", "Sec 4", "grid convergence", None), "SKIP", "-", "sensitivity_check.py without uniform grid")


# ----------------------------------------------------------------------------- report
def tex_check(tex):
    if tex is None:
        return
    norm = lambda x: re.sub(r"\s+", " ", x)
    t = norm(tex)
    for i, (c, v, comp, note) in enumerate(RESULTS):
        if c.tex and norm(c.tex) not in t and v not in ("SKIP",):
            RESULTS[i] = (c, "TEXT", comp, f"paper does not contain '{c.tex}'" + (f"; {note}" if note else ""))


def main(only, texfile):
    for f in (claims_models, claims_visibility, claims_observability, claims_filters, claims_reacquisition,
              claims_custody, claims_ephemeris, claims_sensitivity):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            record(Claim(f.__name__, "-", f"{f.__name__} crashed", None), "FAIL", "-", repr(e))
    tex = Path(texfile).read_text() if Path(texfile).exists() else None
    tex_check(tex)
    lines = [f"{'id':18s} {'verdict':7s} {'where':18s} claim  |  paper -> data   [note]", "-" * 110]
    order = {"FAIL": 0, "TEXT": 1, "WEAK": 2, "SKIP": 3, "MANUAL": 4, "PASS": 5}
    for c, v, comp, note in RESULTS:
        if only and v != only:
            continue
        lines.append(f"{c.cid:18s} {v:7s} {c.where:18s} {c.text}  |  {c.paper} -> {comp}" + (f"   [{note}]" if note else ""))
    counts = {k: sum(1 for r in RESULTS if r[1] == k) for k in order}
    lines += ["-" * 110, "summary: " + ", ".join(f"{k} {n}" for k, n in counts.items() if n)]
    worst = [r for r in RESULTS if r[1] in ("FAIL", "TEXT")]
    if worst:
        lines.append("needs attention: " + ", ".join(r[0].cid for r in worst))
    out = "\n".join(lines)
    print(out)
    (DATA / "claims_report.txt").write_text(out + "\n")
    return 1 if worst else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["PASS", "FAIL", "WEAK", "TEXT", "SKIP", "MANUAL"])
    ap.add_argument("--tex", default=str(ROOT / "paper" / "main.tex"))
    a = ap.parse_args()
    sys.exit(main(a.only, a.tex))
