"""Consider covariance: how error sources that the estimator does not model inflate the post-fit
covariance of an angles-only tracking arc (milestone 9).

fisher.py gives the Cramer-Rao bound, which assumes that white measurement noise is the only
error. Real tracking also has, per site,
    * a constant angle bias b (catalogue and astrometric systematics), sigma_b per sky axis,
    * a constant clock (timing) offset tau, sigma_tau,
and, for the whole arc and gap,
    * an error dp in the solar-radiation-pressure (SRP) area-to-mass ratio, sigma_p [m^2/kg].
They are 'consider' parameters c ~ N(0, C): the estimator does not solve for them. With
z_k = h_k(x) + G_k c + v_k, A_k = H_k Phi(t_k, t0) and W = R^-1 (as in fisher.py), the weighted
least-squares estimate of the arc-start state x0 has the error
    x0_hat - x0 = P_x sum_k A_k^T W (G_k c + v_k),        P_x = CRLB at t0,
so S0 = P_x sum_k A_k^T W G_k. The estimate is mapped to the arc end with the unperturbed model,
while the truth also carries the SRP displacement Psi(t) = dx(t)/dp, so the arc-end error is
    e = x_hat - x = Phi(t_e, t0) (x0_hat - x0) - Psi(t_e) dp,    S_end = Phi S0 - [0 ... 0, Psi(t_e)]
    P_end = Phi P_x Phi^T + S_end C S_end^T.
For the gap, the truth is sampled from the joint covariance of (x - x_hat, dp) (7 x 7); dp then
keeps acting during the gap (custody.predict_gap, argument 'consider').

Measurement partials (RA, Dec in radians, as in fisher.py):
    bias    G = diag(1 / cos(dec), 1)  (sigma_b is an on-sky angle in both axes)
    timing  G = d(ra, dec)/dt          (topocentric angular rate, supplied by the caller)
    SRP     G = H_k Psi(t_k)
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import LU_KM, TU_S, TU_DAYS, VU_KMS
from ..dynamics.cr3bp import eom, jacobian, propagate
from ..frames.ephemeris import AU_KM, sun_position
from ..frames.synodic import synodic_basis
from ..sensors.measurement import ARCSEC
from .fisher import crlb, eci_from_synodic_jacobian, radec_jacobian

P_SUN_1AU = 4.56e-6          # solar radiation pressure at 1 au [N m^-2]
CR_REF = 1.3                 # reflectivity coefficient (as in the ephemeris check)
AM_REF = 0.01                # reference area-to-mass ratio [m^2 kg^-1] (as in the ephemeris check)


class SRPModel:
    """Cannonball SRP acceleration per unit area-to-mass ratio, in synodic components and CR3BP
    units (LU / TU^2 per m^2/kg), tabulated against time t [TU] since jd_ref and interpolated
    linearly. The Sun direction is geocentric (the target is within 1.5e6 km of Earth, so the
    direction error is below 0.6 deg); no shadow (eclipses are rare and short)."""

    def __init__(self, jd_ref, t_min, t_max, cr=CR_REF, step_days=0.02):
        self.jd_ref = float(jd_ref)
        n = int(np.ceil((t_max - t_min) * TU_DAYS / step_days)) + 2
        self.t = np.linspace(t_min, t_max, max(n, 2))
        jd = self.jd_ref + self.t * TU_DAYS
        s = sun_position(jd)
        r = np.linalg.norm(s, axis=1)
        a_ms2 = cr * P_SUN_1AU * (AU_KM / r) ** 2                 # per unit A/m [m s^-2 / (m^2/kg)]
        u = -s / r[:, None]                                        # away from the Sun
        xh, yh, zh = synodic_basis(jd)
        comp = np.stack([np.sum(u * xh, 1), np.sum(u * yh, 1), np.sum(u * zh, 1)], axis=1)
        self.a = comp * (a_ms2 * 1e-3 * TU_S ** 2 / LU_KM)[:, None]

    def __call__(self, t):
        return np.array([np.interp(t, self.t, self.a[:, i]) for i in range(3)])

    def shifted(self, t_shift):
        """Same table with time measured from t_shift [TU] (for a gap that starts at t_shift)."""
        m = SRPModel.__new__(SRPModel)
        m.jd_ref, m.t, m.a = self.jd_ref + t_shift * TU_DAYS, self.t - t_shift, self.a
        return m


def propagate_srp_sensitivity(s0, grid, mu, accel, rtol=1e-12, atol=1e-12):
    """Psi(t) = dx(t)/dp along the unperturbed trajectory from s0 (Psi(0) = 0):
    dPsi/dt = A(t) Psi + [0, 0, 0, accel(t)]. grid [TU] starts at 0. Returns (len(grid), 6)."""
    def f(t, y):
        s, psi = y[:6], y[6:]
        return np.concatenate([eom(t, s, mu), jacobian(s, mu) @ psi + np.r_[0.0, 0.0, 0.0, accel(t)]])
    sol = solve_ivp(f, (0.0, grid[-1]), np.r_[s0, np.zeros(6)], method="DOP853", rtol=rtol, atol=atol,
                    t_eval=grid)
    if not sol.success:
        raise RuntimeError(f"sensitivity integration failed: {sol.message}")
    return sol.y[6:].T


@dataclass
class ConsiderArc:
    P_crlb_end: np.ndarray       # CRLB at the arc end (identical to fisher.arc_information)
    S_end: np.ndarray            # (6, n_c) sensitivity of the arc-end error to the consider parameters
    n_sites: int
    observable: bool
    sigma_pos_crlb_km: float     # largest 1-sigma position axis of the CRLB at the arc end

    def columns(self):
        n = self.n_sites
        return {"bias": np.arange(0, 2 * n), "timing": np.arange(2 * n, 3 * n), "srp": np.array([3 * n])}

    def C(self, bias_arcsec=0.0, timing_s=0.0, srp_sigma=0.0):
        n = self.n_sites
        return np.diag(np.r_[np.full(2 * n, (bias_arcsec * ARCSEC) ** 2), np.full(n, timing_s ** 2),
                             srp_sigma ** 2])

    def covariance(self, bias_arcsec=0.0, timing_s=0.0, srp_sigma=0.0):
        """(P_end, P_aug): consider covariance of the arc-end error (6, 6), and the joint covariance
        of (truth - estimate, dp) (7, 7) to sample the truth for the gap."""
        C = self.C(bias_arcsec, timing_s, srp_sigma)
        P = self.P_crlb_end + self.S_end @ C @ self.S_end.T
        P = 0.5 * (P + P.T)
        aug = np.zeros((7, 7))
        aug[:6, :6] = P
        aug[:6, 6] = aug[6, :6] = -(self.S_end @ C)[:, -1]       # truth - estimate = -e
        aug[6, 6] = srp_sigma ** 2
        return P, aug

    def sigma_pos_km(self, P):
        return float(np.sqrt(max(np.linalg.eigvalsh(P[:3, :3]).max(), 0.0)) * LU_KM)


def arc_consider(orbit, t0, t_meas, r_site, basis_meas, t_end, *, site_id, rate, srp=None,
                 sigma_arcsec=1.0, mu=None, prior_sigma_pos_km=1e5, prior_sigma_vel_kms=1.0):
    """Consider sensitivities for one arc from t0 to t0 + t_end [TU].

    t_meas, r_site, basis_meas: as in fisher.arc_information (only measurements with
    t_meas - t0 <= t_end are used); site_id: (M,) integer site index of each measurement;
    rate: (M, 2) topocentric d(ra, dec)/dt [rad s^-1]; srp: SRPModel with times in TU since
    jd_ref, where t0 is measured from the same jd_ref (None: no SRP column)."""
    mu = orbit.mu if mu is None else mu
    tau = np.asarray(t_meas, float) - t0
    use = tau <= t_end + 1e-12
    tau, r_site = tau[use], np.asarray(r_site)[use]
    basis_meas = tuple(np.asarray(b)[use] for b in basis_meas)
    site_id, rate = np.asarray(site_id)[use], np.asarray(rate, float)[use]
    n_sites = int(site_id.max()) + 1 if site_id.size else 1

    grid = np.unique(np.concatenate([[0.0], tau, [t_end]]))
    x0 = orbit.states_at(t0)[0]
    sol = propagate(x0, (0.0, grid[-1]), mu, with_stm=True, t_eval=grid)       # as in fisher.py
    Y = sol.y.T
    S, Phi = Y[:, :6], Y[:, 6:].reshape(-1, 6, 6)
    k = np.searchsorted(grid, tau)
    Sk, Phik = S[k], Phi[k]
    xh, yh, zh = basis_meas
    r_eci = LU_KM * ((Sk[:, 0:1] + mu) * xh + Sk[:, 1:2] * yh + Sk[:, 2:3] * zh)
    rho = r_eci - r_site
    H3 = radec_jacobian(rho) @ eci_from_synodic_jacobian(basis_meas)
    H = np.concatenate([H3, np.zeros_like(H3)], axis=2)                          # (M, 2, 6)
    A = H @ Phik
    w = (sigma_arcsec * ARCSEC) ** -2
    info = (np.einsum("mki,mkj->mij", A, A) * w).sum(axis=0)                     # same order as fisher.py
    P_x = crlb(info, prior_sigma_pos_km, prior_sigma_vel_kms)

    M = len(tau)
    n_c = 3 * n_sites + 1
    G = np.zeros((M, 2, n_c))
    cosd = np.sqrt(rho[:, 0] ** 2 + rho[:, 1] ** 2) / np.linalg.norm(rho, axis=1)
    m = np.arange(M)
    G[m, 0, 2 * site_id] = 1.0 / cosd
    G[m, 1, 2 * site_id + 1] = 1.0
    G[m, :, 2 * n_sites + site_id] = rate
    psi_end = np.zeros(6)
    if srp is not None:
        psi = propagate_srp_sensitivity(x0, grid, mu, lambda t: srp(t0 + t))
        G[:, :, -1] = np.einsum("mij,mj->mi", H, psi[k])
        psi_end = psi[np.searchsorted(grid, t_end)]
    S0 = P_x @ np.einsum("mki,mkc->ic", A, G) * w
    F = Phi[np.searchsorted(grid, t_end)]
    S_end = F @ S0
    S_end[:, -1] -= psi_end
    P_end = F @ P_x @ F.T
    sp = float(np.sqrt(max(np.linalg.eigvalsh(P_end[:3, :3]).max(), 0.0)) * LU_KM)
    P0_pos = np.sqrt(np.linalg.eigvalsh(P_x[:3, :3]).max()) * LU_KM
    return ConsiderArc(P_end, S_end, n_sites, bool(P0_pos < 0.1 * prior_sigma_pos_km), sp)
