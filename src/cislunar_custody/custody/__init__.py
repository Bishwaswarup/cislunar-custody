"""Custody horizon: how long a post-fit uncertainty can go unobserved before the
target leaves the telescope search region."""
from .horizon import (search_radius_deg, predict_gap, crossing_time, custody_horizons,
                      strip_pattern, strip_coordinates, strip_containment, containment_horizon,
                      strip_horizon, MAX_STRIP_DEG, STRIP_TILT_DEG, CHI2_2_99)
