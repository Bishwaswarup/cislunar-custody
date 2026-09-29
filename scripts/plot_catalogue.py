"""Figures for the orbit catalogue -> figures/fig01_families.png, fig02_family_properties.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cislunar_custody.catalogue import load_family, load_named  # noqa: E402
from cislunar_custody.constants import LU_KM, TU_DAYS, MU_EM  # noqa: E402
from cislunar_custody.plotstyle import use_jas_style, save, panel_label, DOUBLE  # noqa: E402

use_jas_style()

CAT = ROOT / "data" / "catalogue.npz"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
FAMS = {  # name: (label, colour, line style)   Okabe-Ito colours + distinct dashes
    "L1_halo_north": ("$L_1$ halo (N)", "#0072B2", "-"),
    "L2_halo_south": ("$L_2$ halo (S) / NRHO", "#D55E00", "--"),
    "L1_lyapunov": ("$L_1$ Lyapunov", "#CC79A7", "-."),
    "L2_lyapunov": ("$L_2$ Lyapunov", "#E69F00", ":"),
    "DRO": ("DRO", "#009E73", (0, (5, 1, 1, 1, 1, 1))),
}


def km(S):  # synodic LU -> km, Moon-centred x
    out = S[:, :3] * LU_KM
    out[:, 0] -= (1 - MU_EM) * LU_KM
    return out / 1e3  # thousand km


def fig_families():
    fig = plt.figure(figsize=(DOUBLE, 0.40 * DOUBLE))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    axy = fig.add_subplot(1, 3, 2)
    axz = fig.add_subplot(1, 3, 3)
    for name, (lab, col, ls) in FAMS.items():
        fam = load_family(CAT, name)
        planar = "lyap" in name.lower() or name == "DRO"
        idx = np.unique(np.linspace(0, len(fam) - 1, 12).astype(int))
        for j, i in enumerate(idx):
            _, S = fam[i].trajectory(600)
            P = km(S)
            kw = dict(color=col, ls=ls, lw=0.6, alpha=0.9, label=lab if j == 0 else None)
            ax3.plot(P[:, 0], P[:, 1], P[:, 2], **kw)
            if planar:
                axy.plot(P[:, 0], P[:, 1], **kw)
            else:
                axz.plot(P[:, 0], P[:, 2], **kw)
    nrho = load_named(CAT, "NRHO_9:2")
    _, S = nrho.trajectory(2000)
    P = km(S)
    ax3.plot(P[:, 0], P[:, 1], P[:, 2], "k", lw=1.6, label="9:2 NRHO")
    axz.plot(P[:, 0], P[:, 2], "k", lw=1.6, label="9:2 NRHO")
    ax3.scatter([0], [0], [0], s=30, c="grey")
    ax3.set_xlabel("x from Moon [10$^3$ km]", labelpad=-2)
    ax3.set_ylabel("y [10$^3$ km]", labelpad=-2)
    ax3.set_zlabel("z [10$^3$ km]", labelpad=-2)
    ax3.tick_params(pad=0)
    panel_label(ax3, "a", x=0.0, y=0.95)
    h, lab = ax3.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), handlelength=2.4)
    for ax, yl, letter, txt in ((axy, "y [10$^3$ km]", "b", "planar families (xy)"),
                                (axz, "z [10$^3$ km]", "c", "halo / NRHO families (xz)")):
        ax.scatter([0], [0], s=12, c="grey", zorder=5)
        ax.set_xlabel("x from Moon [10$^3$ km]"); ax.set_ylabel(yl)
        panel_label(ax, letter, txt)
        ax.set_aspect("equal"); ax.grid(True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "fig01_families", ROOT)


def fig_properties():
    d = np.load(CAT)
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 0.34 * DOUBLE))
    for name, (lab, col, ls) in FAMS.items():
        T = d[f"{name}__periods"] * TU_DAYS
        C = d[f"{name}__jacobi"]
        nu = d[f"{name}__stability"]
        rp = d[f"{name}__r_moon_min"] * LU_KM
        ax[0].plot(C, T, ls=ls, marker="o", ms=1.5, color=col, label=lab)
        ax[1].semilogy(C, nu, ls=ls, marker="o", ms=1.5, color=col)
        ax[2].loglog(rp, T, ls=ls, marker="o", ms=1.5, color=col)
    ax[0].set_xlabel("Jacobi constant C"); ax[0].set_ylabel("period [days]"); ax[0].legend()
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("Jacobi constant C"); ax[1].set_ylabel("stability index ν")
    ax[2].set_xlabel("closest Moon distance [km]"); ax[2].set_ylabel("period [days]")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    for a, l in zip(ax, "abc"):
        panel_label(a, l)
    fig.tight_layout()
    save(fig, "fig02_family_properties", ROOT)


if __name__ == "__main__":
    fig_families()
    fig_properties()
    print("saved figures to", FIG)
