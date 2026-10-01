"""Figure style for The Journal of the Astronautical Sciences (Springer Nature).

JAS artwork rules applied here:
  * lettering in Helvetica/Arial, 8-12 pt (enforced: every text object is clamped to 8-12 pt)
  * figure width 84 mm (single column) or 174 mm (double column), height <= 234 mm
  * no titles inside the figure (figure and axes titles are removed; panels get "a", "b", ...)
  * vector output (PDF) and TIFF at 600 dpi for combination art, files named Fig1, Fig2, ...
  * colour is never the only cue: every orbit / filter also has its own line style or marker

Usage in a script:
    from cislunar_custody.plotstyle import use_jas_style, save, ORBIT_STYLE, DOUBLE
    use_jas_style()
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.45 * DOUBLE))
    ...
    save(fig, "fig12_uncertainty_growth", ROOT)
"""
import io
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text

MM = 1 / 25.4
SINGLE, DOUBLE, MAX_H = 84 * MM, 174 * MM, 234 * MM
MIN_PT, MAX_PT = 8.0, 12.0

# Okabe-Ito colour-blind-safe palette, plus a distinct line style and marker per orbit.
ORBIT_STYLE = {
    "NRHO 9:2":            dict(color="#000000", ls="-",  marker="o"),
    "L1 halo (Az~30k km)": dict(color="#0072B2", ls="--", marker="s"),
    "L2 Lyapunov (mid)":   dict(color="#D55E00", ls="-.", marker="^"),
    "DRO (~70k km)":       dict(color="#009E73", ls=":",  marker="D"),
}
ORBIT_LABEL = {
    "NRHO 9:2": "9:2 NRHO",
    "L1 halo (Az~30k km)": "$L_1$ halo",
    "L2 Lyapunov (mid)": "$L_2$ Lyapunov",
    "DRO (~70k km)": "DRO",
}
FILTER_STYLE = {
    "EKF":    dict(color="#D55E00", ls="-",  marker="v", hatch="///"),
    "UKF":    dict(color="#0072B2", ls="--", marker="s", hatch="..."),
    "GM-UKF": dict(color="#009E73", ls="-.", marker="o", hatch=""),
    "PF→UKF": dict(color="#CC79A7", ls=":",  marker="D", hatch="xx"),
}
NETWORK_STYLE = {
    "India-1": dict(color="#D55E00", ls=":"),
    "Tri-3":   dict(color="#0072B2", ls="--"),
    "Tri-3+S": dict(color="#009E73", ls="-"),
}

# Order of first citation in paper/main.tex -> JAS file name FigN.
JAS_ORDER = {
    "fig01_families": 1,
    "fig05_moon_separation": 2,
    "fig03_visibility_timeline": 3,
    "fig04_gap_cdf": 4,
    "fig06_crlb_vs_date": 5,
    "fig10_prior_cloud": 6,
    "fig11_reacquisition_success": 7,
    "fig12_uncertainty_growth": 8,
    "fig15_blackout_survival": 9,
    "fig13_tc_predictor": 10,
    "fig14_gaussian_containment": 11,
    "fig16_tc_cr3bp_vs_ephemeris": 12,
    "fig17_growth_cr3bp_vs_ephemeris": 13,
    "fig18_tc_vs_stability": 14,
    "fig19_operator_vs_ideal": 15,
    "fig20_consider_horizon": 16,
}


def use_jas_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.sf": "sans",
        "mathtext.cal": "sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "savefig.bbox": None,       # JAS files keep the exact 84 / 174 mm canvas
        "pdf.fonttype": 42,        # embed TrueType fonts
        "ps.fonttype": 42,
        "figure.dpi": 100,
    })


def panel_label(ax, letter, text=None, x=0.0, y=1.02):
    """Bold panel letter (Springer style 'a', 'b', ...) with an optional short identifier."""
    s = rf"$\mathbf{{{letter}}}$" + (f"   {text}" if text else "")
    if getattr(ax, "name", "") == "3d":
        ax.text2D(x, y, s, transform=ax.transAxes, ha="left", va="bottom", fontsize=MIN_PT, gid="panel")
    else:
        ax.text(x, y, s, transform=ax.transAxes, ha="left", va="bottom", fontsize=MIN_PT, gid="panel")


def _enforce(fig, stem):
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    for t in fig.findobj(Text):
        s = t.get_fontsize()
        if s < MIN_PT:
            t.set_fontsize(MIN_PT)
        elif s > MAX_PT:
            t.set_fontsize(MAX_PT)
    w, h = fig.get_size_inches()
    if not (abs(w - SINGLE) < 0.02 or abs(w - DOUBLE) < 0.02):
        warnings.warn(f"{stem}: width {w / MM:.0f} mm is neither 84 nor 174 mm")
    if h > MAX_H + 1e-6:
        warnings.warn(f"{stem}: height {h / MM:.0f} mm exceeds 234 mm")


def save(fig, stem, root):
    """figures/<stem>.png (300 dpi preview for the README) and, for figures used in the
    paper, figures/jas/FigN.pdf (vector) and figures/jas/FigN.tif (600 dpi, RGB, LZW)."""
    _enforce(fig, stem)
    figdir = Path(root) / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    n = JAS_ORDER.get(stem)
    if n is not None:
        jas = figdir / "jas"
        jas.mkdir(exist_ok=True)
        fig.savefig(jas / f"Fig{n}.pdf")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=600, facecolor="white")
        try:
            from PIL import Image
            buf.seek(0)
            Image.open(buf).convert("RGB").save(jas / f"Fig{n}.tif", compression="tiff_lzw", dpi=(600, 600))
        except ImportError:
            fig.savefig(jas / f"Fig{n}.tif", dpi=600, facecolor="white")
    plt.close(fig)
