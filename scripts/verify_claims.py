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
nonlinear_filter_study -> custody_study -> ephemeris_check -> sensitivity_check -> family_sweep -> this.
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
UT_MIN = (65, 72)      # paper: minima of the median UT containment curves (Sec 8)


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
    paper = {"NRHO": 1.14, "L1": 1.83, "DRO": 7.03, "L2": 17.8}
    vel, frac, weak = [], [], {}
    for o, pv in paper.items():
        s = col(a, "sig_pos_max_km", orbit=ORB[o], arc_d=7.0)
        check_num(Claim(f"O1.{o}", "Sec 6 / abstract", f"{o} median CRLB, 7-d arcs [km]", pv), np.nanmedian(s),
                  0.005 if pv < 10 else 0.05)
        vel.append(np.nanmedian(col(a, "sig_vel_max_ms", orbit=ORB[o], arc_d=7.0)) * 1e3)
        obs = [r["observable"] in (True, "True", 1.0) for r in a if r["orbit"] == ORB[o] and r["arc_d"] == 7.0]
        frac.append(100 * np.mean(obs))
        weak[o] = np.nanmedian(col(a, "weak_los_deg", orbit=ORB[o], arc_d=7.0))
    check_range(Claim("O2", "Sec 6", "median velocity CRLB, 7-d arcs [mm/s]", (6, 108), "6--108"), min(vel), max(vel), 0.5)
    check_range(Claim("O3", "Sec 6", "share of 7-d arcs that are observable [%]", (75, 84), "75\\,\\% and 84"),
                min(frac), max(frac), 0.5)
    check_range(Claim("O4", "Sec 6", "median weak-direction angle to LOS, DRO & L2 [deg]", (6.4, 7.8), "6.4--7.8"),
                min(weak["DRO"], weak["L2"]), max(weak["DRO"], weak["L2"]), 0.05)
    check_range(Claim("O5", "Sec 6", "median weak-direction angle, NRHO & L1 [deg]", (12.8, 17.9), "12.8--17.9"),
                min(weak["NRHO"], weak["L1"]), max(weak["NRHO"], weak["L1"]), 0.05)


def claims_filters():
    f = load("filter_summary.csv")
    if f is None:
        return skip(Claim("F", "Sec 6", "tracking claims", None), "filter_summary.csv")
    from scipy.stats import chi2
    tab = {"NRHO": (1.16, 1.58, 5.13, 1.58, 5.14), "L1": (1.83, 1.94, 5.71, 1.97, 5.36),
           "L2": (17.88, 9.60, 9.25, 9.50, 6.05), "DRO": (7.16, 6.05, 5.13, 6.15, 5.15)}
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
                                f"[{lo:.2f}, {hi:.2f}]: only the EKF on the L2 Lyapunov orbit",
                 "L2/EKF", "except the EKF\non the $L_2$ Lyapunov orbit"),
           "PASS" if bad == [("L2", "EKF")] else "FAIL", ", ".join(f"{o}/{k}" for o, k in bad))


def claims_reacquisition():
    s = load("nonlinear_filter_summary.csv")
    if s is None:
        return skip(Claim("R", "Sec 6", "reacquisition claims", None), "nonlinear_filter_summary.csv")
    r = {x["filter"]: x for x in s if x["orbit"] == ORB["L1"]}
    n = int(r["EKF"]["runs"])
    k = {f: int(round(x["runs_consistent"] * x["runs"])) for f, x in r.items()}
    check_num(Claim("R0", "Sec 6 / abstract", "L1 reacquisition: gap before the first measurement [d]", 3.0,
                    "measurement arrives 3.0\\,d after"), r["EKF"]["gap_before_first_meas_d"], 0.05)
    paper = {"EKF": 37, "UKF": 77, "GM-UKF": 100, "PF→UKF": 88}
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
    med = {"EKF": 68.0, "UKF": 31.0, "GM-UKF": 24.3, "PF→UKF": 27.6}   # PF: resampling is platform-sensitive
    mx = {"EKF": 1792.7, "UKF": 251.5, "GM-UKF": 95.4, "PF→UKF": 1546.9}
    an = {"EKF": 3.4e3, "UKF": 64.0, "GM-UKF": 5.9, "PF→UKF": 1.3e3}
    for fl in paper:
        check_num(Claim(f"R3.{fl}.med", "Table 2", f"{fl} median final error [km]", med[fl]),
                  r[fl]["median_pos_err_km"], 0.05 if fl == "PF→UKF" or med[fl] >= 10 else 0.005)
        check_num(Claim(f"R3.{fl}.max", "Table 2", f"{fl} max final error [km]", mx[fl]), r[fl]["max_pos_err_km"], 0.05)
        tol = 0.05 * an[fl] if an[fl] > 100 else 0.05
        check_num(Claim(f"R3.{fl}.anees", "Table 2", f"{fl} ANEES", an[fl]), r[fl]["anees_2nd_half"], tol)
    check_num(Claim("R4", "Sec 6", "GM-UKF max components (L1)", 27, "up to 27"), r["GM-UKF"]["max_components"], 0)
    cons = lambda o, f_: 100 * next(x for x in s if x["orbit"] == ORB[o] and x["filter"] == f_)["runs_consistent"]
    gaps = [next(x for x in s if x["orbit"] == ORB[o])["gap_before_first_meas_d"] for o in ("NRHO", "L2", "DRO")]
    check_range(Claim("R9a", "Sec 6", "other orbits: reacquisition gaps [d]", (1.9, 3.2), "1.9--3.2\\,d"),
                min(gaps), max(gaps), 0.05)
    v = [cons(o, f_) for o in ("NRHO", "L2", "DRO") for f_ in ("UKF", "GM-UKF")]
    check_range(Claim("R9", "Sec 6", "other orbits: UKF and GM-UKF runs consistent [%]", (94, 99), "94--99"),
                min(v), max(v), 0.5)
    v = [cons(o, "EKF") for o in ("NRHO", "L2", "DRO")]
    check_range(Claim("R9b", "Sec 6", "other orbits: EKF runs consistent [%]", (60, 85), "60--85"), min(v), max(v), 0.5)
    v = [cons(o, "PF→UKF") for o in ("NRHO", "L2", "DRO")]
    check_range(Claim("R10", "Sec 6", "other orbits: PF→UKF runs consistent [%]", (91, 95), "91--95"),
                min(v), max(v), 0.5)
    check_num(Claim("R5", "Sec 6", "GM-UKF mean components (L1)", 3.0, "averaging 3.0"), r["GM-UKF"]["mean_components"], 0.05)
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
        for f_, pv, tx in (("EKF", 73, "contains only 73"), ("UKF", 94, "against 94"), ("GM-UKF", 96, "96\\,\\% for\nthe GM-UKF")):
            check_num(Claim(f"R7.{f_}", "Sec 6", f"L1: PF particles inside the {f_} 99% ellipsoid at first measurement [%]",
                            pv, tx), 100 * x[f"inside99_{f_}"], 0.5)
        check_range(Claim("R8", "Sec 6", "L1 prior cloud 1-99% span [10^3 km]", (42, 7.6), "42\\,000 \\times 7600"),
                    x["span1_km"] / 1e3, x["span2_km"] / 1e3, 0.5)


def claims_custody():
    c = load("custody_cases.csv")
    if c is None:
        return skip(Claim("K", "Sec 7", "custody claims", None), "custody_cases.csv")
    ph = [r for r in c if r["scenario"] == "phase"]
    bl = [r for r in c if r["scenario"] == "blackout"]
    check_num(Claim("K0", "Sec 4", "number of custody cases", 523, "523 cases"), float(len(c)), 0)
    cu = distinct(c)
    check_num(Claim("K0b", "Sec 4", "number of distinct arcs among the custody cases", 393, "393 are distinct"),
              float(len(cu)), 0)
    OP = "ut_actual"   # operator horizon: circle centred on the UT-predicted direction (headline since M8)
    tab3 = {  # Table 3 (operator horizon): (orbit, arc): sigma0, {N: (p50, p10, p90)}
        ("L1", 1): (20.8, {1: (8.8, 5.5, 9.9), 10: (10.9, 7.3, 12.2), 100: (12.6, 9.5, 13.7)}),
        ("L1", 3): (5.1, {1: (11.6, 7.9, 12.4), 10: (13.2, 10.0, 14.1), 100: (14.9, 13.4, 17.1)}),
        ("L1", 7): (0.8, {1: (14.3, 13.0, 15.3), 10: (16.4, 14.6, 18.3), 100: (18.3, 16.3, 19.9)}),
        ("L2", 1): (33.5, {1: (9.4, 7.6, 10.9), 10: (12.3, 9.7, 15.4), 100: (16.2, 12.8, 18.0)}),
        ("L2", 3): (16.2, {1: (10.3, 9.0, 13.4), 10: (14.9, 11.2, 16.4), 100: (18.0, 15.3, 18.8)}),
        ("L2", 7): (7.4, {1: (13.7, 11.6, 15.7), 10: (17.3, 15.4, 18.3), 100: (19.1, 18.3, 21.1)}),
        ("NRHO", 1): (24.5, {1: (30, 5.3, 30), 10: (30, 18.4, 30), 100: (30, 30, 30)}),
        ("NRHO", 3): (4.1, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("NRHO", 7): (0.9, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("DRO", 1): (31.9, {1: (30, 7.8, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("DRO", 3): (11.7, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
        ("DRO", 7): (1.4, {1: (30, 30, 30), 10: (30, 30, 30), 100: (30, 30, 30)}),
    }
    for (o, arc), (s0, cells) in tab3.items():
        sel = [r for r in ph if r["orbit"] == ORB[o] and r["arc_d"] == arc]
        check_num(Claim(f"K1.{o}.{arc}.s0", "Table 3", f"{o} {arc}-d median sigma0 [km]", s0),
                  np.median([r["sig0_pos_km"] for r in sel]), 0.05)
        for N, (p50, p10, p90) in cells.items():
            v = cap30([r[f"tc_{OP}_N{N}"] for r in sel])
            q10, q50, q90 = np.percentile(v, [10, 50, 90])
            ok = all(close(p, q, 0.05) or (p >= 30 and q >= 30) for p, q in ((p50, q50), (p10, q10), (p90, q90)))
            record(Claim(f"K1.{o}.{arc}.N{N}", "Table 3", f"{o} {arc}-d operator T_c(N={N}) median (p10-p90)",
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
                        5.3, "only 5.3"), l14, 0.05)
        check_num(Claim("K5", "Sec 8", "L1 UT containment at 14 d, median over all L1 arcs [%]", 98,
                        "contains 98"), u14, 0.5)
    else:
        record(Claim("K2", "Sec 7", "growth-curve claims", None), "SKIP", "-", "missing custody_curves.npz")
    med = lambda o, a, N: np.median(cap30([r[f"tc_{OP}_N{N}"] for r in ph if r["orbit"] == ORB[o] and r["arc_d"] == a]))
    gains = [med(o, a, 100) - med(o, a, 10) for o in ("L1", "L2") for a in (1, 3, 7)]
    check_range(Claim("K6", "Sec 7", "gain from 10 -> 100 fields, unstable orbits [d]", (1.7, 3.9), "1.7--3.9"),
                min(gains), max(gains), 0.05)
    t7 = [med(o, 7, N) for o in ("L1", "L2") for N in (1, 10, 100)]
    check_range(Claim("K7", "Sec 7", "7-d arcs: median operator T_c for N=1..100, unstable [d]", (13.7, 19.1), "13.7--19.1"),
                min(t7), max(t7), 0.05)
    t10 = [med(o, a, 10) for o in ("L1", "L2") for a in (1, 3, 7)]
    check_range(Claim("K8", "Sec 7", "median operator T_c(N=10), unstable orbits, all arcs [d]", (10.9, 17.3),
                      "10.9--17.3"), min(t10), max(t10), 0.05)
    check_range(Claim("K8b", "abstract / Sec 13", "median operator T_c(N=10), unstable orbits, rounded [d]", (11, 17),
                      "11--17\\,days"), round(min(t10)), round(max(t10)), 0)
    check_range(Claim("K8c", "Sec 7", "L1 halo median operator T_c(N=10), 1-d -> 7-d arcs [d]", (10.9, 16.4),
                      "from 10.9 to 16.4"), med("L1", 1, 10), med("L1", 7, 10), 0.05)
    # operator (UT-centred) vs ideal (true-mean) horizon, all distinct arcs
    dmax, rat = {}, []
    for N in (1, 10, 100):
        a_ = cap30([r[f"tc_ideal_N{N}"] for r in cu])
        b_ = cap30([r[f"tc_{OP}_N{N}"] for r in cu])
        dmax[N] = float(np.max(np.abs(b_ - a_)))
        ok = (a_ > 0) & (a_ < 30)
        rat.append(float(np.median(b_[ok] / a_[ok])))
    check_num(Claim("K14a", "Sec 7", "max |operator - ideal T_c| over distinct arcs, N=1 and 10 [d]", 0.8,
                    "within 0.8\\,d for $N \\le 10$"), max(dmax[1], dmax[10]), 0.05)
    check_num(Claim("K14b", "Sec 7", "max |operator - ideal T_c| over distinct arcs, N=100 [d]", 1.5,
                    "1.5\\,d for $N = 100$"), dmax[100], 0.05)
    check_range(Claim("K14c", "Sec 7", "median operator/ideal T_c ratio over distinct arcs, N=1..100", (1.00, 1.00),
                      "median ratio 1.00"), min(rat), max(rat), 0.005)
    # Table 4 blackouts
    tab4 = {("NRHO", 1): (9, 10.5, 67, 78, 78), ("NRHO", 3): (11, 10.5, 100, 100, 100),
            ("NRHO", 7): (11, 10.5, 100, 100, 100),
            ("L1", 1): (27, 3.1, 67, 89, 93), ("L1", 3): (27, 3.1, 93, 96, 96), ("L1", 7): (28, 3.1, 100, 100, 100),
            ("L2", 1): (33, 4.1, 67, 76, 94), ("L2", 3): (34, 4.1, 79, 97, 97), ("L2", 7): (34, 4.1, 88, 100, 100),
            ("DRO", 1): (13, 11.3, 46, 85, 85), ("DRO", 3): (13, 11.3, 100, 100, 100),
            ("DRO", 7): (13, 11.3, 100, 100, 100)}
    surv, n_diff = {}, []
    for (o, a), (n, ml, s1, s10, s100) in tab4.items():
        sel = [r for r in bl if r["orbit"] == ORB[o] and r["arc_d"] == a]
        L = np.array([r["blackout_d"] for r in sel])
        got = [len(sel), np.median(L)] + [100 * np.mean(np.array([r[f"tc_{OP}_N{N}"] for r in sel]) >= L)
                                            for N in (1, 10, 100)]
        ideal = [100 * np.mean(np.array([r[f"tc_ideal_N{N}"] for r in sel]) >= L) for N in (1, 10, 100)]
        surv[(o, a)] = got[3]
        paper = [n, ml, s1, s10, s100]
        ok = all(close(p, g, t) for p, g, t in zip(paper, got, (0, 0.05, 0.5, 0.5, 0.5)))
        n_diff += [(o, a, N, x) for N, x, y in zip((1, 10, 100), ideal, got[2:]) if not close(x, y, 1e-9)]
        kN = int(round(got[3] * len(sel) / 100))
        lo, hi = wilson(kN, len(sel))
        record(Claim(f"K9.{o}.{a}", "Table 4", f"{o} {a}-d blackouts: n, median length, survived N=1/10/100 (operator = ideal)",
                     " / ".join(f"{x:g}" for x in paper)),
               "PASS" if ok else "FAIL", " / ".join(f"{x:.3g}" for x in got),
               f"N=10 survival 95% CI {100 * lo:.0f}-{100 * hi:.0f}%")
    d_ = ", ".join(f"{o} {a:g}-d N={N}: ideal {x:.0f}%" for o, a, N, x in n_diff)
    check_bool(Claim("K9b", "Table 4", "ideal and operator survival shares differ in one cell only (L1 1-d N=1, 63%)", True,
                     "63\\,\\%)"), len(n_diff) == 1 and n_diff[0][:3] == ("L1", 1.0, 1) and round(n_diff[0][3]) == 63,
               d_ or "none")
    for a, pr in ((1, (76, 89)), (3, (96, 100)), (7, (100, 100))):
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
                            "in 2 of the 253"), float(k), 0)
            RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], f"{share:.1f}% of {fin.sum()} arcs")
            check_num(Claim("K11b", "Sec 8", "distinct arcs with a finite UT-centred horizon at N=10", 253,
                            "in 2 of the 253"), float(fin.sum()), 0)
        else:
            record(Claim("K12", "info", "linear false-custody share at N=10 (not in paper)", "-"), "PASS",
                   f"{share:.1f}% of {fin.sum()} cases", "information only")
    # predictors
    paper_r2 = {"ftle": {1: None, 10: None, 100: None}, "ut_claim": {1: 0.96, 10: 1.00, 100: 0.93},
                "lin_claim": {1: 0.92, 10: 0.99, 100: 0.97}}
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
    check_range(Claim("P2", "Sec 8", "log-R2 of orbit-only FTLE predictor, N=1..100", (-0.25, 0.04), "-0.25"),
                min(ftle), max(ftle), 0.005)
    check_range(Claim("P3", "Sec 8", "number of distinct arcs used for the R2 values", (250, 270), "250--270"),
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
    tab5 = {("NRHO", 1): (8, 30, 28.3, 0.98), ("NRHO", 7): (7, 30, 30, None), ("L1", 1): (4, 8.4, 7.8, 0.93),
            ("L1", 7): (11, 15.9, 15.8, 1.00), ("L2", 1): (5, 11.9, 11.3, 0.95), ("L2", 7): (10, 15.8, 16.4, 1.00),
            ("DRO", 1): (6, 30, 30, 1.00), ("DRO", 7): (9, 30, 30, None)}
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
    check_range(Claim("E6", "abstract / Sec 9", "range of median ephemeris/CR3BP T_c ratios", (0.93, 1.00), "0.93--1.00"),
                min(ratios), max(ratios), 0.005)
    eu = distinct(e)
    check_num(Claim("E4b", "Sec 9", "distinct arcs among the ephemeris cases", 58, "58 distinct"), float(len(eu)), 0)
    cc = np.array([r["tc_cr3bp_ideal_N10"] for r in eu])
    ee = np.array([r["tc_ephem_ideal_N10"] for r in eu])
    fin = np.isfinite(cc) & np.isfinite(ee)
    n_out = int(np.sum(np.abs(cc[fin] - ee[fin]) > 3.0) + np.sum(np.isfinite(cc) != np.isfinite(ee)))
    check_num(Claim("E7", "Sec 9", "distinct arcs whose T_c(N=10) changes by > 3 d (or crosses 30 d)", 5, "5 of the 58"),
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
    check_num(Claim("E8", "Sec 9", "blackout groups whose survival count differs CR3BP vs ephemeris", 1,
                    "one single-case"), float(len(diffs)), 0)
    RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], "; ".join(diffs))
    for N, pv in ((1, 0.80), (10, 1.00), (100, 0.83)):
        a_ = np.array([r[f"tc_ephem_ideal_N{N}"] for r in eu])
        b_ = np.array([r[f"tc_ephem_ut_claim_N{N}"] for r in eu])
        r2, n, (blo, bhi) = r2log(a_, b_)
        weak = f"n={n}; bootstrap 95% CI {blo:.2f}-{bhi:.2f}" if (bhi - blo) > 0.1 or n < 30 else None
        check_num(Claim(f"E9.N{N}", "Sec 9", f"ephemeris UT log-R2, N={N}", pv), r2, 0.005, weak=weak)
    for o, key, pv in (("L1", "contain_lin_ephem_14d", 12.0), ("L1", "contain_lin_cr3bp_14d", 9.9),
                       ("L2", "contain_lin_ephem_14d", 79.9), ("L2", "contain_lin_cr3bp_14d", 98.7)):
        check_num(Claim(f"E10.{o}.{key}", "Sec 9", f"{o} {key} median [%]", pv),
                  100 * np.median([r[key] for r in e if r["orbit"] == ORB[o]]), 0.05)
    ut = [100 * np.median([r["contain_ut_ephem_14d"] for r in e if r["orbit"] == ORB[o]]) for o in ORB]
    check_range(Claim("E11", "Sec 9", "ephemeris UT containment at 14 d, all orbits [%]", (98, 99), "98--99"),
                min(ut), max(ut), 0.5)


def claims_sensitivity():
    s = load("sensitivity_samples.csv")
    c = Claim("S1", "Sec 12 limitations", "T_c converged in Monte Carlo sample size (300 vs larger sample)", None)
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
    check_num(Claim("S2", "Sec 12", "arcs in the sample-size check", 17, "17 representative arcs"), float(len(s)), 0)
    check_range(Claim("S3", "Sec 12", "median |dT_c| 300 vs 2000 samples, N=1..100 [d]", (0.08, 0.13),
                      "0.08--0.13\\,d"), min(meds), max(meds), 0.005)
    check_num(Claim("S4", "Sec 12", "max |dT_c| for N=10 and 100 [d]", 0.7, "at most 0.7\\,d"), mx10, 0.05)
    a = np.array([r["tc_n300_N1"] for r in s])
    b = np.array([r[f"tc_{big}_N1"] for r in s])
    fin = np.isfinite(a) & np.isfinite(b)
    i = int(np.flatnonzero(fin)[np.argmax(np.abs(a[fin] - b[fin]))])
    w = s[i]
    check_range(Claim("S5", "Sec 12", "N=1 worst case: T_c with 300 and 2000 samples [d]", (5.3, 17.6),
                      "from 5.3 to 17.6\\,d"), a[i], b[i], 0.05)
    RESULTS[-1] = (RESULTS[-1][0], RESULTS[-1][1], RESULTS[-1][2], f"{w['orbit']} {w['scenario']} {w['arc_d']:g}-d arc")
    per = {"NRHO 9:2": json.load(open(DATA / "named_orbits.json"))["NRHO_9:2"]["period_days"]}
    ph = np.mod(w["t_end_day"] + a[i], per.get(w["orbit"], np.inf))
    ph = min(ph, per.get(w["orbit"], np.inf) - ph)
    check_bool(Claim("S6", "Sec 12", "N=1 worst case is an NRHO arc whose 300-sample horizon ends at perilune (< 0.1 d)",
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
        check_num(Claim("G1", "Sec 4", "grid check: max median |dT_c|, 121-point vs uniform 0.02-d grid [d]", 0.03,
                        "median of at most 0.03\\,d"), max(gm), 0.005)
        check_num(Claim("G2", "Sec 4", "grid check: max |dT_c| over all N, 121-point vs uniform grid [d]", 0.17,
                        "at most 0.17\\,d"), max(gx), 0.05)
        check_num(Claim("G3", "Sec 4", "grid check: max |dT_c| for N=1 and 10 [d]", 0.15, "at most 0.15\\,d for $N = 1$"),
                  max(gx[:2]), 0.05)
    else:
        record(Claim("G1", "Sec 4", "grid convergence", None), "SKIP", "-", "sensitivity_check.py without uniform grid")


def claims_families():
    """Milestone 8: operator horizon and stability index across orbit families (family_sweep.py)."""
    f = load("family_sweep.csv")
    if f is None:
        return skip(Claim("FS", "Sec 10", "family-sweep claims", None), "family_sweep.csv (run family_sweep.py)")
    from scipy.stats import spearmanr
    for r in f:
        r["member"] = int(r["member"])
    members = {(r["family"], r["member"]) for r in f}
    check_num(Claim("FS1a", "Sec 10", "family sweep: number of members", 30, "30 members remain"), float(len(members)), 0)
    check_num(Claim("FS1b", "Sec 10", "family sweep: number of cases", 413, "413 cases"), float(len(f)), 0)
    # FS2: operator (UT-centred circle) vs ideal (true-mean circle)
    rat, dmax = [], {}
    for N in (1, 10, 100):
        a = np.array([r[f"tc_ideal_N{N}"] for r in f])
        b = cap30([r[f"tc_utc_N{N}"] for r in f])
        ok = np.isfinite(a) & (a > 0) & (a < 30)
        rat.append(float(np.median(b[ok] / a[ok])))
        dmax[N] = float(np.max(np.abs(b - cap30(a))))
    check_range(Claim("FS2a", "Sec 10 / abstract", "family sweep: median operator/ideal ratio, N=1..100", (1.00, 1.00),
                      "is 1.00 for all three search sizes"), min(rat), max(rat), 0.005)
    for cid, N, pv in (("FS2b", 1, 0.17), ("FS2c", 10, 0.38), ("FS2d", 100, 1.18)):
        check_num(Claim(cid, "Sec 10", f"family sweep: max |operator - ideal T_c|, N={N} [d]", pv, f"{pv:.2f}\\,d"),
                  dmax[N], 0.005)

    # member medians (capped at 30 d), as in family_sweep.member_table
    def table(arc, key="utc"):
        out = []
        for fam, mem in sorted(members):
            sel = [r for r in f if r["family"] == fam and r["member"] == mem and r["arc_d"] == arc]
            if sel:
                out.append((fam, sel[0]["stability"], float(np.median(cap30([r[f"tc_{key}_N10"] for r in sel])))))
        return out

    def rho(m):
        r_, p_ = spearmanr(np.log10([x[1] for x in m]), [x[2] for x in m])
        return float(r_), float(p_), len(m)

    # FS3: rank correlation with log10(nu), operator circle, N=10
    hi = {arc: rho([x for x in table(arc) if x[1] > 10]) for arc in (3.0, 7.0)}
    al = {arc: rho([x for x in table(arc) if x[1] > 1.01]) for arc in (3.0, 7.0)}
    check_num(Claim("FS3a", "Sec 10 / abstract", "Spearman rho(T_c, log nu), nu > 10, 3-d arcs", -0.92,
                    "$\\rho = -0.92$ (3-day arcs)"),
              hi[3.0][0], 0.005)
    check_num(Claim("FS3b", "Sec 10", "Spearman rho(T_c, log nu), nu > 10, 7-d arcs", -0.91, "$-0.91$ (7-day arcs)"),
              hi[7.0][0], 0.005)
    check_num(Claim("FS3c", "Sec 10", "members with nu > 10", 15, "15 members with $\\nu > 10$"), float(hi[3.0][2]), 0)
    check_bool(Claim("FS3d", "Sec 10", "nu > 10 correlation significant, p < 1e-5 (both arcs)", True, "p < 10^{-5}"),
               max(hi[3.0][1], hi[7.0][1]) < 1e-5, f"p = {hi[3.0][1]:.1g}, {hi[7.0][1]:.1g}")
    check_range(Claim("FS3e", "Sec 10", "Spearman rho, all 25 members with nu > 1.01, 3-d and 7-d arcs", (-0.97, -0.91),
                      "$-0.97$ and $-0.91$"), al[3.0][0], al[7.0][0], 0.005)
    # FS4: within-family correlations; the L2 halo family is weak
    fr = {(fam, arc): rho([x for x in table(arc) if x[0] == fam and x[1] > 1.01])
          for fam in ("L1_halo_north", "L2_halo_south", "L1_lyapunov", "L2_lyapunov") for arc in (3.0, 7.0)}
    l2 = [fr[("L2_halo_south", a)] for a in (3.0, 7.0)]
    check_range(Claim("FS4a", "Sec 10", "L2 halo family rho, 3-d and 7-d arcs", (-0.61, -0.61), "$\\rho = -0.61$"),
                l2[0][0], l2[1][0], 0.005)
    check_num(Claim("FS4b", "Sec 10", "L2 halo family p-value", 0.14, "$p = 0.14$"), max(l2[0][1], l2[1][1]), 0.005,
              weak="7 members, 6 of them capped at 30 d: no evidence either way within this family")
    oth = [v[0] for k, v in fr.items() if k[0] != "L2_halo_south"]
    check_range(Claim("FS4c", "Sec 10", "within-family rho, L1 halo and both Lyapunov families", (-1.00, -0.91),
                      "between $-0.91$ and $-1.00$"), min(oth), max(oth), 0.005)
    l2t = table(7.0)
    nl = [x for x in l2t if x[0] == "L2_halo_south"]
    check_bool(Claim("FS4d", "Sec 10", "six of seven L2 halo members have nu <= 1.7 and T_c > 30 d", True,
                     "Six of its seven members"),
               sum(1 for x in nl if x[1] <= 1.7 and x[2] >= 30) == 6 and len(nl) == 7,
               f"{sum(1 for x in nl if x[1] <= 1.7 and x[2] >= 30)} of {len(nl)}")
    # FS5: stable members and the range of unstable horizons (7-d arcs)
    st = [x for x in l2t if x[1] < 3]
    check_bool(Claim("FS5a", "Sec 10", "all members with nu < 3 keep custody > 30 d after 7-d arcs", True,
                     "all 15 members with $\\nu < 3$"), len(st) == 15 and all(x[2] >= 30 for x in st), f"{len(st)} members")
    t3 = [x for x in table(3.0) if x[1] < 3 and x[2] < 30]
    t3 = sorted(t3, key=lambda x: x[1])
    check_bool(Claim("FS5c", "Sec 10", "3-d arcs: the nu < 3 members lost within 30 d have nu 2.4, 2.9 and T_c 29.1, 26.9 d",
                     True, "$\\nu = 2.4$ and 2.9, lost at 29.1 and 26.9\\,d"),
               len(t3) == 2 and all(close(a, x[1], 0.05) and close(b, x[2], 0.05)
                                    for (a, b), x in zip(((2.4, 29.1), (2.9, 26.9)), t3)),
               ", ".join(f"{x[0]} nu {x[1]:.2f} T_c {x[2]:.2f}" for x in t3))
    big = sorted(round(x[1]) for x in l2t if x[1] > 10 and x[2] >= 30 and x[0] == "L1_lyapunov")
    check_bool(Claim("FS5d", "Sec 10", "7-d arcs: nu > 10 members beyond 30 d are two L1 Lyapunov (nu 71, 113)", True,
                     "($\\nu = 71$ and 113)"),
               big == [71, 113] and sum(1 for x in l2t if x[1] > 10 and x[2] >= 30) == 2, str(big))
    un = [x[2] for x in l2t if x[1] > 10 and x[2] < 30]
    check_range(Claim("FS5b", "Sec 10 / Discussion", "finite member-median T_c, nu > 10, 7-d arcs [d]", (12.9, 25.2),
                      "12.9--25.2\\,d"), min(un), max(un), 0.05)
    # FS6: strip search (secondary)
    for cid, N, pv in (("FS6a", 1, 0.97), ("FS6b", 10, 1.08), ("FS6c", 100, 0.98)):
        a = np.array([r[f"tc_ideal_N{N}"] for r in f])
        b = cap30([r[f"tc_strip_N{N}"] for r in f])
        ok = np.isfinite(a) & (a > 0) & (a < 30)
        check_num(Claim(cid, "Sec 10", f"strip/ideal median T_c ratio, N={N}", pv, "0.97, 1.08 and 0.98"),
                  float(np.median(b[ok] / a[ok])), 0.005)
    a = np.array([r["tc_ideal_N10"] for r in f])
    b = np.array([r["tc_strip_N10"] for r in f])
    fin = np.isfinite(a) & (a < 30)
    gain = fin & (~np.isfinite(b) | (b > a + 3.0))
    check_num(Claim("FS6d", "Sec 10", "N=10 cases where the strip gains > 3 d", 54, "54 of the 206"), float(gain.sum()), 0)
    check_num(Claim("FS6e", "Sec 10", "N=10 cases with a finite ideal horizon", 206, "54 of the 206"), float(fin.sum()), 0)
    lyap = sum(1 for r, g in zip(f, gain) if g and "lyapunov" in r["family"])
    check_num(Claim("FS6f", "Sec 10", "gaining cases on the Lyapunov families", 49, "49 of them"), float(lyap), 0)
    cg = np.median([r["strip_cross99_arcsec"] for r, g in zip(f, gain) if g])
    co = np.median([r["strip_cross99_arcsec"] for r, g, x in zip(f, gain, fin) if x and not g])
    check_range(Claim("FS6g", "Sec 10", "cross-track 99% half-width, gaining vs other cases [arcsec]", (0.6, 1.5),
                      "0.6$''$"), cg, co, 0.05, weak="rests on the CRLB covariance (no biases); milestone 9 re-tests it")


def claims_consider():
    """Milestone 9: consider covariance (site biases, clock offsets, SRP errors), consider_study.py."""
    rows = load("consider_cases.csv")
    if rows is None:
        return skip(Claim("C", "Sec 11", "consider-covariance claims", None), "consider_cases.csv (run consider_study.py)")
    key = lambda r: (r["orbit"], r["arc_d"], round(r["t_end_day"], 6))
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[key(r)] = r

    def st(cfg, N=10):
        ref = by[next(iter(by[cfg].values()))["ref"]]
        pr = [(ref[k], r) for k, r in by[cfg].items() if k in ref]
        ok = [(b, r) for b, r in pr if b[f"tc_ut_actual_N{N}"] < 30]
        ra = np.array([min(r[f"tc_ut_actual_N{N}"], 30.0) / b[f"tc_ut_actual_N{N}"] for b, r in ok])
        ri = np.array([min(r[f"tc_ideal_N{N}"], 30.0) / b[f"tc_ideal_N{N}"] for b, r in pr if b[f"tc_ideal_N{N}"] < 30])
        fc = 100 * np.mean([r[f"tc_ut_claim_N{N}"] > 1.1 * r[f"tc_ut_actual_N{N}"] for _, r in ok])
        return {"n": len(ok), "med": float(np.median(ra)), "p10": float(np.percentile(ra, 10)),
                "n20": int(np.sum(ra < 0.8)), "f20": 100 * float(np.mean(ra < 0.8)), "fc": fc,
                "ideal": float(np.median(ri)), "pairs": pr}

    cc = load("custody_cases.csv")
    if cc is not None:
        ref = {key(r): r for r in cc if r["scenario"] == "phase"}
        d = 0.0
        for k, r in by["base"].items():
            for N in (1, 10, 100):
                for m in ("ideal", "ut_actual", "ut_claim"):
                    a, b = ref[k][f"tc_{m}_N{N}"], r[f"tc_{m}_N{N}"]
                    d = max(d, 0.0 if (not np.isfinite(a) and not np.isfinite(b)) else abs(a - b))
        check_bool(Claim("C1", "Sec 11", "baseline (no extra errors) reproduces custody_cases.csv", True,
                         "reproduces the horizons of Section~\\ref{sec:custody} exactly"), d < 1e-3, f"max |dT_c| {d:.2g} d")
    check_num(Claim("C2a", "Sec 11", "arcs in the consider study", 126, "126 arcs"), float(len(by["base"])), 0)
    nom = st("nominal")
    check_num(Claim("C2b", "Sec 11", "arcs with a reference horizon < 30 d (N=10)", 67, "the 67 arcs"), float(nom["n"]), 0)
    check_num(Claim("C3a", "Sec 11 / abstract", "nominal: median operator T_c ratio", 0.99, "0.99, 10th percentile 0.96"),
              nom["med"], 0.005)
    check_num(Claim("C3b", "Sec 11", "nominal: 10th percentile ratio", 0.96, "0.99, 10th percentile 0.96"), nom["p10"], 0.005)
    check_num(Claim("C4a", "Sec 11", "site bias 0.2 arcsec: median ratio", 0.99, "up to 0.2$''$ (ratio 0.99)"),
              st("bias0.2")["med"], 0.005)
    b5 = st("bias0.5")
    check_num(Claim("C4b", "Sec 11", "site bias 0.5 arcsec: median ratio", 0.95, "(ratio 0.95, with 5 arcs"), b5["med"], 0.005)
    check_num(Claim("C4c", "Sec 11", "site bias 0.5 arcsec: arcs losing > 20 %", 5, "with 5 arcs losing"), float(b5["n20"]), 0)
    c5 = st("common0.5")
    check_bool(Claim("C4d", "Sec 11", "a common bias matters less than independent site biases (0.5 arcsec)", True,
                     "A common bias matters less"), c5["med"] > b5["med"] and c5["fc"] < b5["fc"],
               f"median {c5['med']:.3f} vs {b5['med']:.3f}")
    ck = st("clock10ms")
    check_bool(Claim("C5a", "Sec 11", "clock offsets of 10 ms have no effect (all ratios 1.000)", True, "clock offsets have no effect"),
               abs(ck["med"] - 1) < 5e-4 and abs(ck["p10"] - 1) < 5e-4, f"median {ck['med']:.4f}, p10 {ck['p10']:.4f}")
    sm = [st(f"srp_am{a:g}") for a in (0.005, 0.01, 0.02, 0.05)]
    check_range(Claim("C6a", "Sec 11", "SRP modelled, A/m 0.005-0.05: median ratios", (0.98, 1.00), "0.98--1.00"),
                min(x["med"] for x in sm), max(x["med"] for x in sm), 0.005)
    check_num(Claim("C6b", "Sec 11", "SRP modelled, A/m 0.05: 10th percentile ratio", 0.82, "falls to 0.82"), sm[-1]["p10"], 0.005)
    ns = st("nosrp_am0.05")
    check_num(Claim("C7a", "Sec 11", "SRP not modelled, A/m 0.05: median ratio", 0.94, "the median ratio is 0.94"), ns["med"], 0.005)
    check_num(Claim("C7b", "Sec 11", "SRP not modelled, A/m 0.05: arcs losing > 20 %", 8, "8 arcs lose"), float(ns["n20"]), 0)
    check_num(Claim("C7c", "Sec 11", "SRP not modelled, A/m 0.05: ideal median ratio", 0.99, "hardly changes (ratio 0.99)"),
              ns["ideal"], 0.005)
    big = max((x for x in by), key=lambda c: 0 if c.startswith("base") else 1 - st(c)["med"] if "0.3" not in c and c != "conservative" else 0)
    RESULTS.append((Claim("C7d", "Sec 11", "unmodelled SRP (A/m 0.05) is the largest single effect", True,
                          "the largest single effect"), "PASS" if big == "nosrp_am0.05" else "FAIL", big, ""))
    cv = st("conservative")
    check_num(Claim("C8a", "Sec 11 / abstract", "conservative: median ratio", 0.93, "median ratio of 0.93"), cv["med"], 0.005)
    check_num(Claim("C8b", "Sec 11", "conservative: 10th percentile", 0.71, "percentile 0.71"), cv["p10"], 0.005)
    check_num(Claim("C8c", "Sec 11", "conservative: arcs losing > 20 %", 13, "13 arcs losing"), float(cv["n20"]), 0)
    worst = {o: [0, 0] for o in ("NRHO 9:2", "DRO (~70k km)")}
    ref = by["base"]
    for c in by:
        if c.startswith("base") or next(iter(by[c].values()))["ref"] != "base":
            continue
        for o in worst:
            pr = [r for k, r in by[c].items() if k in ref and r["orbit"] == o and ref[k]["tc_ut_actual_N10"] >= 30]
            worst[o] = [max(worst[o][0], sum(r["tc_ut_actual_N10"] < 30 for r in pr)), len(pr)]
    worst = {o: tuple(v) for o, v in worst.items()}
    check_bool(Claim("C8d", "Sec 11", "at most 2 of 28 NRHO arcs fall below 30 d, no DRO arc", True, "at most\n2 of the 28 NRHO arcs"),
               worst["NRHO 9:2"] == (2, 28) and worst["DRO (~70k km)"][0] == 0, str(worst))
    b0, b3 = by["base"], by["base_0.3"]
    dg = [r["tc_ut_actual_N10"] - b0[k]["tc_ut_actual_N10"] for k, r in b3.items()
          if k in b0 and b0[k]["tc_ut_actual_N10"] < 30 and r["tc_ut_actual_N10"] < 30]
    check_num(Claim("C9a", "Sec 11", "0.3 arcsec noise: median horizon gain [d]", 1.8, "by a median of 1.8\\,d"),
              float(np.median(dg)), 0.05)
    n3, c3 = st("nominal_0.3"), st("conservative_0.3")
    check_num(Claim("C9b", "Sec 11", "nominal, 0.3 arcsec: median ratio", 0.96, "(ratio 0.96)"), n3["med"], 0.005)
    check_num(Claim("C9c", "Sec 11", "conservative, 0.3 arcsec: median ratio", 0.81, "(ratio 0.81"), c3["med"], 0.005)
    check_num(Claim("C9d", "Sec 11", "conservative, 0.3 arcsec: share losing > 20 % [%]", 46, "46\\,\\% of the arcs"), c3["f20"], 0.5)
    for cid, cfg, pv, tx in (("C10a", "base", 1.5, "in 1.5\\,\\% of the arcs"), ("C10b", "nominal", 9, "in 9\\,\\% under"),
                             ("C10c", "bias0.5", 34, "in 34\\,\\% with"), ("C10d", "conservative_0.3", 82, "82\\,\\% for")):
        check_num(Claim(cid, "Sec 11", f"CRLB-claim false custody, {cfg} [%]", pv, tx), st(cfg)["fc"], 0.05 if pv < 2 else 0.5)
    try:
        from cislunar_custody.frames import Ephemeris
        from cislunar_custody.constants import MU_EM
        from cislunar_custody.sensors import SITES, NETWORKS, TELESCOPES, network_visibility, radec
        from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd
        sys.path.insert(0, str(ROOT / "scripts"))
        from visibility_study import representative_orbits, EPOCH, STEP_MIN, TARGET
        jd0 = jd_from_iso(EPOCH)
        jd = jd_grid(jd0, 365, STEP_MIN)
        eph = Ephemeris(jd)
        t = tu_from_jd(jd, jd0)
        keys = list(NETWORKS["Tri-3+S"])
        meds, p95 = [], []
        for orb in representative_orbits().values():
            r = eph.to_eci(orb.states_at(t), MU_EM)
            ps, _ = network_visibility(eph, r, keys, SITES, TELESCOPES["1m"], TARGET, moon_excl_deg=5.0)
            rr = []
            for k in keys:
                ra, dec = radec(r - eph.site_eci(SITES[k])[0])
                w = np.hypot(np.gradient(np.unwrap(ra), STEP_MIN * 60) * np.cos(dec), np.gradient(dec, STEP_MIN * 60))
                rr.append(w[ps[k].visible] * 206264.8)
            rr = np.concatenate(rr)
            meds.append(np.median(rr))
            p95.append(np.percentile(rr, 95))
        check_bool(Claim("C11", "Sec 11", "sky rate: median ~0.4 arcsec/s, 95th percentile < 0.7 arcsec/s (all orbits)", True,
                         "less than 0.7$''$\\,s$^{-1}$"), all(0.35 <= m <= 0.45 for m in meds) and max(p95) < 0.7,
                   f"medians {min(meds):.2f}-{max(meds):.2f}, p95 max {max(p95):.2f}")
    except Exception as e:  # noqa: BLE001
        record(Claim("C11", "Sec 11", "sky rate", None), "MANUAL", "-", f"recompute failed: {e}")


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
              claims_custody, claims_ephemeris, claims_sensitivity, claims_families, claims_consider):
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
