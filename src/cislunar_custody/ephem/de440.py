"""JPL DE440 access via jplephem (geocentric Moon and Sun, km, km/s).

Download once (32 MB, 1849-2150):
    curl -o data/de440s.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
Times are Julian dates, treated as TDB (UTC-TDB ~ 69 s is negligible here).
"""
from pathlib import Path

import numpy as np

DEFAULT_KERNEL = Path(__file__).resolve().parents[3] / "data" / "de440s.bsp"


class DE440:
    def __init__(self, path=DEFAULT_KERNEL):
        from jplephem.spk import SPK
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found (see module docstring to download)")
        self.k = SPK.open(str(self.path))

    def _pv(self, a, b, jd):
        p, v = self.k[a, b].compute_and_differentiate(np.asarray(jd, float))
        return np.asarray(p), np.asarray(v) / 86400.0          # km, km/s

    def moon_state(self, jd):
        """Geocentric Moon position [km] and velocity [km/s], shape (3,) or (3, N)."""
        pm, vm = self._pv(3, 301, jd)
        pe, ve = self._pv(3, 399, jd)
        return pm - pe, vm - ve

    def sun_position(self, jd):
        """Geocentric Sun position [km], shape (3,) or (3, N)."""
        ps = np.asarray(self.k[0, 10].compute(np.asarray(jd, float)))
        pb = np.asarray(self.k[0, 3].compute(np.asarray(jd, float)))
        pe = np.asarray(self.k[3, 399].compute(np.asarray(jd, float)))
        return ps - pb - pe

    def moon_position(self, jd):
        return self.moon_state(jd)[0]
