"""Custody horizon: how long a post-fit uncertainty can go unobserved before the
target leaves the telescope search region."""
from .horizon import (search_radius_deg, predict_gap, crossing_time, custody_horizons,
                      CHI2_2_99)
