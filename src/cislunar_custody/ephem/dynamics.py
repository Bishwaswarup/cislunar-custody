"""Earth-centred inertial (ICRF) dynamics: Earth + DE440 Moon and Sun third-body terms
+ cannonball SRP with a cylindrical Earth shadow. Internally scaled to LU and TU
(same units as the CR3BP) for well-conditioned integration."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from ..constants import GM_EARTH, GM_MOON, LU_KM, TU_S, DAY_S, R_EARTH_KM, R_MOON_KM
from ..frames.ephemeris import AU_KM

GM_SUN = 132712440041.9394     # km^3/s^2 (DE440)
P_SUN = 4.56e-6                # N/m^2 at 1 AU


class EphemerisModel:
    """Dynamics over [jd0, jd0 + span_days]; time argument tau in TU since jd0."""

    def __init__(self, de, jd0, span_days, *, area_to_mass=0.01, cr=1.3, srp=True, step_hours=1.0):
        self.jd0 = jd0
        jd = jd0 + np.arange(-2.0 / 24, span_days + 3.0 / 24, step_hours / 24.0)
        tau = (jd - jd0) * DAY_S / TU_S
        self.moon = CubicSpline(tau, de.moon_position(jd).T / LU_KM)
        self.sun = CubicSpline(tau, de.sun_position(jd).T / LU_KM)
        s = TU_S ** 2 / LU_KM ** 3
        self.gme, self.gmm, self.gms = GM_EARTH * s, GM_MOON * s, GM_SUN * s
        # SRP magnitude at 1 AU: P * Cr * A/m [m/s^2] -> km/s^2 -> LU/TU^2
        self.a_srp = (P_SUN * cr * area_to_mass * 1e-3) * TU_S ** 2 / LU_KM if srp else 0.0
        self.au = AU_KM / LU_KM
        self.re, self.rm = R_EARTH_KM / LU_KM, R_MOON_KM / LU_KM

    def rhs(self, tau, Y, offsets=None):
        """Time derivative. offsets (k,) [TU] lets each stacked state carry its own epoch
        (state i is at model time tau + offsets[i]); used by multiple shooting."""
        S = Y.reshape(-1, 6)
        r = S[:, :3]
        t = tau if offsets is None else tau + offsets
        rm = np.atleast_2d(self.moon(t))
        rs = np.atleast_2d(self.sun(t))
        rn = np.maximum(np.linalg.norm(r, axis=1, keepdims=True), self.re)
        dm = rm - r
        dmn = np.maximum(np.linalg.norm(dm, axis=1, keepdims=True), self.rm)
        ds = rs - r
        dsn = np.linalg.norm(ds, axis=1, keepdims=True)
        rmn = np.linalg.norm(rm, axis=1, keepdims=True)
        rsn = np.linalg.norm(rs, axis=1, keepdims=True)
        a = (-self.gme * r / rn ** 3
             + self.gmm * (dm / dmn ** 3 - rm / rmn ** 3)
             + self.gms * (ds / dsn ** 3 - rs / rsn ** 3))
        if self.a_srp:
            s_hat = rs / rsn
            along = np.sum(r * s_hat, axis=1)
            perp = np.linalg.norm(r - along[:, None] * s_hat, axis=1)
            lit = ~((along < 0) & (perp < self.re))
            a = a - self.a_srp * (self.au / dsn) ** 2 * (ds / dsn) * lit[:, None]
        out = np.empty_like(S)
        out[:, :3] = S[:, 3:]
        out[:, 3:] = a
        return out.ravel()

    def propagate_dense(self, S0, taus, rtol=1e-10, atol=1e-12):
        """States (k, 6) at tau = 0 -> (M, k, 6) at each taus (starting at 0)."""
        S0 = np.atleast_2d(np.asarray(S0, float))
        taus = np.asarray(taus, float)
        sol = solve_ivp(self.rhs, (0.0, taus[-1]), S0.ravel(), method="DOP853", rtol=rtol, atol=atol,
                        t_eval=taus)
        if not sol.success:
            raise RuntimeError(f"ephemeris integration failed: {sol.message}")
        return sol.y.T.reshape(len(taus), -1, 6)

    def rhs_stm(self, tau, Y, offsets):
        """State + STM (42 per stacked state); gravity-gradient variational equations
        (the SRP gradient is negligible and omitted)."""
        Z = Y.reshape(-1, 42)
        S = Z[:, :6]
        Phi = Z[:, 6:].reshape(-1, 6, 6)
        dS = self.rhs(tau, S.ravel(), offsets).reshape(-1, 6)
        t = tau + offsets
        r = S[:, :3]
        G = np.zeros((len(S), 3, 3))
        I3 = np.eye(3)
        for gm, rb, rmin in ((self.gme, np.zeros((1, 3)), self.re),
                             (self.gmm, np.atleast_2d(self.moon(t)), self.rm),
                             (self.gms, np.atleast_2d(self.sun(t)), 0.0)):
            d = r - rb
            dn = np.maximum(np.linalg.norm(d, axis=1), rmin)[:, None, None]
            G += -gm * (I3 / dn ** 3 - 3.0 * d[:, :, None] * d[:, None, :] / dn ** 5)
        A = np.zeros((len(S), 6, 6))
        A[:, :3, 3:] = I3
        A[:, 3:, :3] = G
        out = np.concatenate([dS, (A @ Phi).reshape(len(S), 36)], axis=1)
        return out.ravel()

    def propagate_offsets_stm(self, S0, offsets, dt, rtol=1e-10, atol=1e-12):
        """Like propagate_offsets but also returns the STMs: (k, 6), (k, 6, 6)."""
        S0 = np.atleast_2d(np.asarray(S0, float))
        k = len(S0)
        Y0 = np.concatenate([S0, np.tile(np.eye(6).ravel(), (k, 1))], axis=1)
        sol = solve_ivp(self.rhs_stm, (0.0, dt), Y0.ravel(), method="DOP853", rtol=rtol, atol=atol,
                        args=(np.asarray(offsets, float),))
        if not sol.success:
            raise RuntimeError(f"ephemeris integration failed: {sol.message}")
        Z = sol.y[:, -1].reshape(k, 42)
        return Z[:, :6], Z[:, 6:].reshape(k, 6, 6)

    def propagate_offsets(self, S0, offsets, dt, rtol=1e-10, atol=1e-12):
        """Propagate states that start at different epochs (offsets [TU]) by the same dt;
        dt may be a scalar or, via separate calls, per-state. Returns (k, 6)."""
        S0 = np.atleast_2d(np.asarray(S0, float))
        offsets = np.asarray(offsets, float)
        if dt == 0.0:
            return S0.copy()
        sol = solve_ivp(self.rhs, (0.0, dt), S0.ravel(), method="DOP853", rtol=rtol, atol=atol,
                        args=(offsets,))
        if not sol.success:
            raise RuntimeError(f"ephemeris integration failed: {sol.message}")
        return sol.y[:, -1].reshape(-1, 6)
