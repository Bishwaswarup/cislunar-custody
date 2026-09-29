"""Time helpers. v1 treats UTC = UT1 = TT (differences < 70 s are irrelevant for
sensor geometry at the 0.01 deg level)."""
from datetime import datetime, timezone

import numpy as np

from .constants import TU_S, DAY_S

JD_UNIX_EPOCH = 2440587.5
JD_J2000 = 2451545.0


def jd_from_datetime(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return JD_UNIX_EPOCH + dt.timestamp() / DAY_S


def jd_from_iso(s: str) -> float:
    """'2027-01-01T00:00' (UTC) -> Julian date."""
    return jd_from_datetime(datetime.fromisoformat(s))


def datetime_from_jd(jd: float) -> datetime:
    return datetime.fromtimestamp((jd - JD_UNIX_EPOCH) * DAY_S, tz=timezone.utc)


def jd_grid(jd0: float, days: float, step_min: float) -> np.ndarray:
    n = int(round(days * 1440.0 / step_min)) + 1
    return jd0 + np.arange(n) * step_min / 1440.0


def tu_from_jd(jd, jd0):
    """Elapsed CR3BP time units since jd0."""
    return (np.asarray(jd, float) - jd0) * DAY_S / TU_S
