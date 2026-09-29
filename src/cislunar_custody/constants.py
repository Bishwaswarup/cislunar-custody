"""Physical constants and Earth-Moon CR3BP characteristic units."""
import numpy as np

GM_EARTH = 398600.435436   # km^3/s^2 (DE440)
GM_MOON = 4902.800066      # km^3/s^2 (DE440)
MU_EM = GM_MOON / (GM_EARTH + GM_MOON)   # ~0.0121505856

LU_KM = 384400.0                                   # length unit [km]
TU_S = float(np.sqrt(LU_KM**3 / (GM_EARTH + GM_MOON)))  # time unit [s] (~4.343 d)
VU_KMS = LU_KM / TU_S                              # velocity unit [km/s]

DAY_S = 86400.0
TU_DAYS = TU_S / DAY_S
SYNODIC_MONTH_DAYS = 29.530589

R_EARTH_KM = 6378.137
R_MOON_KM = 1737.4
