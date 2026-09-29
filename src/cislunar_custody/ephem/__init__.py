"""High-fidelity cross-check: Earth-centred ephemeris dynamics (JPL DE440 Sun and Moon
point masses + cannonball solar radiation pressure) and the CR3BP <-> inertial mapping."""
from .de440 import DE440, DEFAULT_KERNEL
from .dynamics import EphemerisModel, GM_SUN
from .transform import synodic_to_inertial, inertial_to_synodic, transform_jacobian
from .gap import predict_gap_ephem
from .shooting import correct_ephemeris_orbit, EphemerisTrajectory
