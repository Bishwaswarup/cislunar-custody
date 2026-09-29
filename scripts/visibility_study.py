"""Milestone 2: one year of ground-based visibility for representative cislunar orbits.

    python scripts/visibility_study.py            # epoch 2027-01-01, 10-min steps
Sweeps telescope class x network x Moon-exclusion angle for four orbits.
Outputs: data/visibility_summary.csv, figures/fig03_visibility_timeline.png,
         figures/fig04_gap_cdf.png, figures/fig05_moon_separation.png, and tables on stdout.
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cislunar_custody.catalogue import load_family, load_named  # noqa: E402
from cislunar_custody.constants import LU_KM, MU_EM  # noqa: E402
from cislunar_custody.frames import Ephemeris  # noqa: E402
from cislunar_custody.sensors import (SITES, NETWORKS, TELESCOPES, Target, REASONS,  # noqa: E402
                                      network_visibility, gap_stats)
from cislunar_custody.timeutil import jd_from_iso, jd_grid, tu_from_jd  # noqa: E402

CAT = ROOT / "data" / "catalogue.npz"
EPOCH, DAYS, STEP_MIN = "2027-01-01T00:00", 365, 10.0
TARGET = Target(radius_m=1.0, albedo=0.2)
EXCLUSIONS_DEG = (2.0, 5.0, 10.0)
REF = dict(tel="1m", excl=5.0, net="Tri-3+S")     # reference case for the figures


def representative_orbits():
    d = np.load(CAT)
    orbits = {"NRHO 9:2": load_named(CAT, "NRHO_9:2")}
    l1 = load_family(CAT, "L1_halo_north")
    orbits["L1 halo (Az~30k km)"] = l1[int(np.argmin(np.abs(d["L1_halo_north__states"][:, 2] * LU_KM - 30000)))]
    l2 = load_family(CAT, "L2_lyapunov")
    orbits["L2 Lyapunov (mid)"] = l2[len(l2) // 2]
    dro = load_family(CAT, "DRO")
    orbits["DRO (~70k km)"] = dro[int(np.argmin(np.abs(d["DRO__r_moon_min"] * LU_KM - 70000)))]
    return orbits


def geocentric_moon_separation_deg(eph, r_eci):
    a, b = r_eci, eph.moon
    c = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def main():
    t0 = time.time()
    jd0 = jd_from_iso(EPOCH)
    jd = jd_grid(jd0, DAYS, STEP_MIN)
    dt_days = STEP_MIN / 1440.0
    eph = Ephemeris(jd)
    t_tu = tu_from_jd(jd, jd0)
    orbits = representative_orbits()

    rows, cache, seps = [], {}, {}
    for oname, orb in orbits.items():
        r_eci = eph.to_eci(orb.states_at(t_tu), MU_EM)
        seps[oname] = geocentric_moon_separation_deg(eph, r_eci)
        for excl in EXCLUSIONS_DEG:
            for tkey, tel in TELESCOPES.items():
                for nkey, site_keys in NETWORKS.items():
                    per_site, union = network_visibility(eph, r_eci, site_keys, SITES, tel, TARGET,
                                                         moon_excl_deg=excl)
                    g = gap_stats(union, dt_days)
                    blocked_all = {r: float(np.mean(np.all(np.stack([v.blocked[r] for v in per_site.values()]), axis=0)))
                                   for r in REASONS}
                    rows.append({"orbit": oname, "moon_excl_deg": excl, "telescope": tkey, "network": nkey,
                                 **{k: v for k, v in g.items() if k != "gaps_d"},
                                 **{f"all_blocked_{r}": v for r, v in blocked_all.items()}})
                    cache[(oname, excl, tkey, nkey)] = (per_site, union, g)

    out = ROOT / "data" / "visibility_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nGeocentric Moon separation [deg]  (min / median / max)")
    for oname, s in seps.items():
        print(f"  {oname:22s} {s.min():5.1f} / {np.median(s):5.1f} / {s.max():5.1f}")

    for tkey in ("1m", "2m"):
        print(f"\n{TELESCOPES[tkey].name} telescopes: visible fraction % | max gap [d]   by Moon exclusion angle")
        print(f"  {'orbit':22s} {'network':8s} " + " ".join(f"{e:>5.0f} deg      " for e in EXCLUSIONS_DEG))
        for oname in orbits:
            for nkey in NETWORKS:
                cells = []
                for e in EXCLUSIONS_DEG:
                    r = next(x for x in rows if x["orbit"] == oname and x["telescope"] == tkey
                             and x["network"] == nkey and x["moon_excl_deg"] == e)
                    cells.append(f"{100 * r['frac_visible']:5.1f}% |{r['max_gap_d']:6.2f}")
                print(f"  {oname:22s} {nkey:8s} " + "   ".join(cells))

    plot_timeline(cache, "NRHO 9:2", **REF)
    plot_gap_cdf(cache, orbits, REF["tel"], REF["excl"])
    plot_separation(seps, jd, jd0)
    print(f"\ndone in {time.time() - t0:.1f} s -> {out.name}, fig03, fig04, fig05")


def plot_timeline(cache, oname, tel, excl, net, days=60):
    per_site, union, _ = cache[(oname, excl, tel, net)]
    n = int(days * 1440 / STEP_MIN)
    codes = [v.first_reason()[:n] for v in per_site.values()]
    codes.append(np.where(union[:n], 0, len(REASONS) + 1))
    labels = [SITES[k].name for k in per_site] + ["NETWORK (any site)"]
    cmap = ListedColormap(["#2a9d8f", "#264653", "#8d99ae", "#6a4c93", "#e9c46a", "#e76f51", "#f4a261", "#d62828"])
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.imshow(np.array(codes), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=7,
              extent=[0, days, len(codes) - 0.5, -0.5])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"days since {EPOCH} UTC")
    ax.set_title(f"{oname}: first blocking constraint per site "
                 f"({TELESCOPES[tel].name} telescopes, 2 m target, {excl:.0f} deg Moon exclusion)")
    names = ["visible"] + list(REASONS) + ["no site"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(8)]
    ax.legend(handles, names, ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig03_visibility_timeline.png", dpi=180)


def plot_gap_cdf(cache, orbits, tel, excl):
    fig, axes = plt.subplots(1, len(orbits), figsize=(16, 3.8), sharey=True)
    for ax, oname in zip(axes, orbits):
        drawn = False
        for nkey, col in zip(NETWORKS, ["#d1495b", "#2a6fdb", "#2a9d8f"]):
            g = cache[(oname, excl, tel, nkey)][2]["gaps_d"]
            if g.size == 0:
                continue
            x = np.sort(g)
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, label=nkey)
            drawn = True
        if drawn:
            ax.set_xscale("log")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "never visible", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(oname, fontsize=10)
        ax.set_xlabel("gap length [days]")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("CDF of observation gaps")
    fig.suptitle(f"Observation-gap distribution over one year "
                 f"({TELESCOPES[tel].name} telescopes, 2 m target, {excl:.0f} deg Moon exclusion)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig04_gap_cdf.png", dpi=180)


def plot_separation(seps, jd, jd0, days=60):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    t = jd - jd0
    m = t <= days
    cols = ["#000000", "#2a6fdb", "#e07a1f", "#2a9d8f"]
    for (oname, s), c in zip(seps.items(), cols):
        a1.plot(t[m], s[m], color=c, lw=0.9, label=oname)
        a2.hist(s, bins=np.linspace(0, 16, 65), histtype="step", color=c, density=True, label=oname)
    for e in EXCLUSIONS_DEG:
        a1.axhline(e, color="grey", ls="--", lw=0.7)
        a2.axvline(e, color="grey", ls="--", lw=0.7)
    a1.set_xlabel(f"days since {EPOCH} UTC")
    a1.set_ylabel("angle from Moon, seen from Earth [deg]")
    a1.legend(fontsize=8, ncol=2)
    a1.grid(alpha=0.3)
    a2.set_xlabel("angle from Moon [deg]")
    a2.set_ylabel("fraction of time (density)")
    a2.grid(alpha=0.3)
    fig.suptitle("Cislunar targets stay within a few degrees of the Moon (dashed: exclusion angles swept)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig05_moon_separation.png", dpi=180)


if __name__ == "__main__":
    main()
