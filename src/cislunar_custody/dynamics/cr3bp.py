"""Earth-Moon circular restricted three-body problem (CR3BP).

Non-dimensional synodic frame: barycentre at the origin, Earth at (-mu, 0, 0),
Moon at (1 - mu, 0, 0); unit length = Earth-Moon distance, unit time = 1 / mean motion.
State s = [x, y, z, vx, vy, vz].
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Coriolis block of the variational equations
_OMEGA = np.array([[0.0, 2.0, 0.0],
                   [-2.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])


def distances(r, mu):
    """Distances to Earth (r1) and Moon (r2). r has shape (..., 3) or (..., 6)."""
    r = np.asarray(r, float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    return r1, r2


def pseudo_potential(r, mu):
    """Omega = (x^2 + y^2)/2 + (1-mu)/r1 + mu/r2."""
    r = np.asarray(r, float)
    r1, r2 = distances(r, mu)
    return 0.5 * (r[..., 0] ** 2 + r[..., 1] ** 2) + (1.0 - mu) / r1 + mu / r2


def jacobi(s, mu):
    """Jacobi constant C = 2*Omega - v^2 (vectorised over leading axes)."""
    s = np.asarray(s, float)
    v2 = np.sum(s[..., 3:6] ** 2, axis=-1)
    return 2.0 * pseudo_potential(s[..., :3], mu) - v2


def eom(t, s, mu):
    """CR3BP equations of motion."""
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    c1 = (1.0 - mu) / r1 ** 3
    c2 = mu / r2 ** 3
    return np.array([
        vx, vy, vz,
        2.0 * vy + x - c1 * (x + mu) - c2 * (x - 1.0 + mu),
        -2.0 * vx + y - (c1 + c2) * y,
        -(c1 + c2) * z,
    ])


def hessian(r, mu):
    """Hessian of the pseudo-potential Omega at position r (3x3)."""
    x, y, z = r[0], r[1], r[2]
    dx1, dx2 = x + mu, x - 1.0 + mu
    r1 = np.sqrt(dx1 ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt(dx2 ** 2 + y ** 2 + z ** 2)
    a, b = (1.0 - mu) / r1 ** 3, mu / r2 ** 3
    a5, b5 = 3.0 * (1.0 - mu) / r1 ** 5, 3.0 * mu / r2 ** 5
    uxx = 1.0 - a - b + a5 * dx1 ** 2 + b5 * dx2 ** 2
    uyy = 1.0 - a - b + (a5 + b5) * y ** 2
    uzz = -a - b + (a5 + b5) * z ** 2
    uxy = (a5 * dx1 + b5 * dx2) * y
    uxz = (a5 * dx1 + b5 * dx2) * z
    uyz = (a5 + b5) * y * z
    return np.array([[uxx, uxy, uxz], [uxy, uyy, uyz], [uxz, uyz, uzz]])


def jacobian(s, mu):
    """Jacobian A = d(eom)/ds (6x6) used by the variational equations."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = hessian(s[:3], mu)
    A[3:, 3:] = _OMEGA
    return A


def eom_many(t, Y, mu, r_floor=None):
    """Vectorised EOM for k stacked states: Y = [s_1, ..., s_k] (6k,).
    r_floor = (r1_min, r2_min) [LU] optionally floors the Earth/Moon distances used in the
    gravity terms. It only changes the force INSIDE the bodies (samples that would have
    impacted), keeping the integrator from stalling at the singularity; None = exact."""
    S = Y.reshape(-1, 6)
    x, y, z, vx, vy, vz = S.T
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    if r_floor is not None:
        r1 = np.maximum(r1, r_floor[0])
        r2 = np.maximum(r2, r_floor[1])
    c1 = (1.0 - mu) / r1 ** 3
    c2 = mu / r2 ** 3
    out = np.empty_like(S)
    out[:, 0], out[:, 1], out[:, 2] = vx, vy, vz
    out[:, 3] = 2.0 * vy + x - c1 * (x + mu) - c2 * (x - 1.0 + mu)
    out[:, 4] = -2.0 * vx + y - (c1 + c2) * y
    out[:, 5] = -(c1 + c2) * z
    return out.ravel()


def propagate_many(S0, dt, mu, rtol=1e-10, atol=1e-12, method="DOP853"):
    """Propagate k states (k, 6) together by dt [TU]; returns (k, 6)."""
    S0 = np.atleast_2d(np.asarray(S0, float))
    if dt == 0.0:
        return S0.copy()
    sol = solve_ivp(eom_many, (0.0, dt), S0.ravel(), method=method, args=(mu,), rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"CR3BP integration failed: {sol.message}")
    return sol.y[:, -1].reshape(-1, 6)


def propagate_many_dense(S0, t_eval, mu, rtol=1e-10, atol=1e-12, method="DOP853", r_floor=None):
    """Propagate k states (k, 6) and return them at every t_eval (M,) [TU] -> (M, k, 6).
    t_eval must start at 0 or later and be increasing. See eom_many for r_floor."""
    S0 = np.atleast_2d(np.asarray(S0, float))
    t_eval = np.asarray(t_eval, float)
    sol = solve_ivp(eom_many, (0.0, t_eval[-1]), S0.ravel(), method=method, args=(mu, r_floor),
                    rtol=rtol, atol=atol, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"CR3BP integration failed: {sol.message}")
    return sol.y.T.reshape(len(t_eval), -1, 6)


def eom_stm(t, y, mu):
    """State + STM equations: y = [s (6), vec(Phi) (36)], dPhi/dt = A Phi."""
    s = y[:6]
    phi = y[6:].reshape(6, 6)
    return np.concatenate([eom(t, s, mu), (jacobian(s, mu) @ phi).ravel()])


def propagate(s0, t_span, mu, *, with_stm=False, t_eval=None, events=None,
              dense_output=False, rtol=1e-12, atol=1e-12, method="DOP853"):
    """Integrate the CR3BP (optionally with the STM). t_span: float tf or (t0, tf)."""
    s0 = np.asarray(s0, float)
    if np.isscalar(t_span):
        t_span = (0.0, float(t_span))
    if with_stm:
        y0, fun = np.concatenate([s0, np.eye(6).ravel()]), eom_stm
    else:
        y0, fun = s0, eom
    sol = solve_ivp(fun, t_span, y0, method=method, args=(mu,), rtol=rtol, atol=atol,
                    t_eval=t_eval, events=events, dense_output=dense_output)
    if not sol.success:
        raise RuntimeError(f"CR3BP integration failed: {sol.message}")
    return sol


def stm(s0, t, mu, **kw):
    """Return (s(t), Phi(t, 0))."""
    sol = propagate(s0, t, mu, with_stm=True, **kw)
    yf = sol.y[:, -1]
    return yf[:6], yf[6:].reshape(6, 6)


def _dUdx_axis(x, mu):
    return (x - (1.0 - mu) * (x + mu) / abs(x + mu) ** 3
            - mu * (x - 1.0 + mu) / abs(x - 1.0 + mu) ** 3)


def lagrange_points(mu):
    """Positions of L1..L5 as a dict of 3-vectors."""
    eps = 1e-9
    kw = dict(args=(mu,), xtol=1e-15, rtol=1e-15, maxiter=500)
    x1 = brentq(_dUdx_axis, -mu + eps, 1.0 - mu - eps, **kw)
    x2 = brentq(_dUdx_axis, 1.0 - mu + eps, 2.0, **kw)
    x3 = brentq(_dUdx_axis, -2.0, -mu - eps, **kw)
    h = np.sqrt(3.0) / 2.0
    return {
        "L1": np.array([x1, 0.0, 0.0]),
        "L2": np.array([x2, 0.0, 0.0]),
        "L3": np.array([x3, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, h, 0.0]),
        "L5": np.array([0.5 - mu, -h, 0.0]),
    }
