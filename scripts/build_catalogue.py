"""Build the periodic-orbit catalogue (data/catalogue.npz + data/named_orbits.json).

    python scripts/build_catalogue.py            # all families
    python scripts/build_catalogue.py DRO        # one family (merged into existing file)
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cislunar_custody.catalogue import (FAMILY_SPECS, build_family, save_catalogue,  # noqa: E402
                                        nrho_by_resonance, load_family)
from cislunar_custody.constants import MU_EM  # noqa: E402

OUT = ROOT / "data" / "catalogue.npz"


def main(names):
    families = {}
    if OUT.exists():  # keep families already built
        d = np.load(OUT)
        for n in {k.split("__")[0] for k in d.files if "__" in k}:
            families[n] = load_family(OUT, n)
    for name in names:
        t = time.time()
        families[name] = build_family(name)
        print(f"{name:15s} {len(families[name]):4d} members  {time.time() - t:6.1f} s", flush=True)
    named = {}
    if "L2_halo_south" in families:
        named["NRHO_9:2"] = nrho_by_resonance(families["L2_halo_south"], 9, 2)
        named["NRHO_4:1"] = nrho_by_resonance(families["L2_halo_south"], 4, 1)
    summary = save_catalogue(OUT, families, named, MU_EM)
    for k, v in summary.items():
        print(f"{k}: T = {v['period_days']:.4f} d, perilune = {v['perilune_km']:.0f} km, "
              f"apolune = {v['apolune_km']:.0f} km, nu = {v['stability_index']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(FAMILY_SPECS))
