"""Observatory sites, telescope classes and target models.

Site coordinates are approximate public values. Extinction k (V band) and dark-sky
brightness are NOMINAL ASSUMPTIONS, not measured values; they are swept or bracketed
in the study.
Telescope limiting magnitudes (V, dark sky, zenith, single detection) are assumptions
anchored to the LANL 36 cm detection of CAPSTONE at g ~ 19.6 (Sechrest et al. 2024)
and ~1.5 mag per doubling of aperture.
"""
from dataclasses import dataclass

import numpy as np

from ..frames.earth import geodetic_to_ecef, geodetic_up


@dataclass(frozen=True)
class Site:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float
    extinction: float = 0.15       # k_V [mag / airmass]
    sky_dark_mag: float = 21.5     # V mag / arcsec^2 at zenith, moonless

    @property
    def ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_km)

    @property
    def up(self) -> np.ndarray:
        return geodetic_up(self.lat_deg, self.lon_deg)


SITES = {
    "Hanle": Site("Hanle (IAO), India", 32.7794, 78.9642, 4.50, 0.12, 21.8),
    "Devasthal": Site("Devasthal (ARIES), India", 29.3608, 79.6850, 2.45, 0.15, 21.3),
    "LaPalma": Site("La Palma (ORM), Spain", 28.7606, -17.8816, 2.40, 0.13, 21.9),
    "Haleakala": Site("Haleakala, Hawaii, USA", 20.7083, -156.2571, 3.05, 0.12, 21.8),
    "SidingSpring": Site("Siding Spring, Australia", -31.2733, 149.0617, 1.16, 0.15, 21.7),
}

NETWORKS = {
    "India-1": ("Hanle",),
    "Tri-3": ("Hanle", "LaPalma", "Haleakala"),
    "Tri-3+S": ("Hanle", "LaPalma", "Haleakala", "SidingSpring"),
}


@dataclass(frozen=True)
class Telescope:
    name: str
    aperture_m: float
    m_lim: float                   # V, dark sky, zenith


TELESCOPES = {
    "0.36m": Telescope("0.36 m", 0.36, 19.5),
    "1m": Telescope("1 m", 1.0, 21.5),
    "2m": Telescope("2 m", 2.0, 23.0),
}


@dataclass(frozen=True)
class Target:
    """Diffuse (Lambertian) sphere."""
    radius_m: float = 1.0          # 2 m class spacecraft
    albedo: float = 0.2
