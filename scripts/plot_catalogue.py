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

CAT = ROOT / "data" / "catalogue.npz"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
FAMS = {  # name: (label, colour)
    "L1_halo_north": ("L1 halo (N)", "#2a6fdb"),
    "L2_halo_south": ("L2 halo (S) / NRHO", "#d1495b"),
    "L1_lyapunov": ("L1 Lyapunov", "#6a4c93"),
    "L2_lyapunov": ("L2 Lyapunov", "#e07a1f"),
    "DRO": ("DRO", "#2a9d8f"),
}


def km(S):  # synodic LU -> km, Moon-centred x
    out = S[:, :3] * LU_KM
    out[:, 0] -= (1 - MU_EM) * LU_KM
    return out / 1e3  # thousand km


def fig_families():
    fig = plt.figure(figsize=(16, 5.2))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    axy = fig.add_subplot(1, 3, 2)
    axz = fig.add_subplot(1, 3, 3)
    for name, (lab, col) in FAMS.items():
        fam = load_family(CAT, name)
        planar = "lyap" in name.lower() or name == "DRO"
        idx = np.unique(np.linspace(0, len(fam) - 1, 12).astype(int))
        for j, i in enumerate(idx):
            _, S = fam[i].trajectory(600)
            P = km(S)
            kw = dict(color=col, lw=0.7, alpha=0.8, label=lab if j == 0 else None)
            ax3.plot(P[:, 0], P[:, 1], P[:, 2], **kw)
            if planar:
                axy.plot(P[:, 0], P[:, 1], **kw)
            else:
                axz.plot(P[:, 0], P[:, 2], **kw)
    nrho = load_named(CAT, "NRHO_9:2")
    _, S = nrho.trajectory(2000)
    P = km(S)
    ax3.plot(P[:, 0], P[:, 1], P[:, 2], "k", lw=2, label="9:2 NRHO")
    axz.plot(P[:, 0], P[:, 2], "k", lw=2, label="9:2 NRHO")
    ax3.scatter([0], [0], [0], s=30, c="grey")
    ax3.set_xlabel("x from Moon [10³ km]"); ax3.set_ylabel("y [10³ km]"); ax3.set_zlabel("z [10³ km]")
    ax3.set_title("Earth–Moon CR3BP families (12 members each)")
    ax3.legend(loc="upper left", fontsize=8)
    for ax, yl, t in ((axy, "y [10³ km]", "Planar families (xy)"), (axz, "z [10³ km]", "Halo / NRHO families (xz)")):
        ax.scatter([0], [0], s=40, c="grey", zorder=5)
        ax.set_xlabel("x from Moon [10³ km]"); ax.set_ylabel(yl); ax.set_title(t)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_families.png", dpi=180)


def fig_properties():
    d = np.load(CAT)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, (lab, col) in FAMS.items():
        T = d[f"{name}__periods"] * TU_DAYS
        C = d[f"{name}__jacobi"]
        nu = d[f"{name}__stability"]
        rp = d[f"{name}__r_moon_min"] * LU_KM
        ax[0].plot(C, T, "o-", ms=2, color=col, label=lab)
        ax[1].semilogy(C, nu, "o-", ms=2, color=col)
        ax[2].loglog(rp, T, "o-", ms=2, color=col)
    ax[0].set_xlabel("Jacobi constant C"); ax[0].set_ylabel("period [days]"); ax[0].legend(fontsize=8)
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("Jacobi constant C"); ax[1].set_ylabel("stability index ν (=1: linearly stable)")
    ax[2].set_xlabel("closest Moon distance [km]"); ax[2].set_ylabel("period [days]")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.suptitle("Family properties: the period and stability that later drive uncertainty growth between observations")
    fig.tight_layout()
    fig.savefig(FIG / "fig02_family_properties.png", dpi=180)


if __name__ == "__main__":
    fig_families()
    fig_properties()
    print("saved figures to", FIG)
