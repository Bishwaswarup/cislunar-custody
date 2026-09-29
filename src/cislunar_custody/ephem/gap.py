"""Gap propagation in the ephemeris model: the same metrics as custody.predict_gap
(ideal / claim / actual sky radii, Gaussian containment, s_max), but with the linear
covariance from a finite-difference STM and positions already geocentric inertial."""
import numpy as np

from ..constants import LU_KM
from ..custody.horizon import CHI2_2_99, _Z, _tangent_basis, _unit
from ..filters.ukf import UKF, _sqrt_psd


def predict_gap_ephem(model, x0, P0, dt_grid, n_samples=300, rng=None, fd_step=1e-5):
    """x0, P0 inertial in LU, LU/TU at the gap start (tau = 0 of `model`)."""
    rng = np.random.default_rng() if rng is None else rng
    dt_grid = np.asarray(dt_grid, float)
    ukf = UKF(0.0, model=None)
    sig = ukf.sigma_points(x0, P0)
    fd = np.vstack([x0 + fd_step * e for e in np.eye(6)] + [x0 - fd_step * e for e in np.eye(6)])
    mc = x0 + rng.standard_normal((n_samples, 6)) @ np.linalg.cholesky(P0).T
    Y = model.propagate_dense(np.vstack([x0[None, :], fd, sig, mc]), dt_grid)
    nom, Yp, Ym = Y[:, 0], Y[:, 1:7], Y[:, 7:13]
    Ys, Ymc = Y[:, 13:13 + len(sig)], Y[:, 13 + len(sig):]
    Phi = np.transpose((Yp - Ym) / (2 * fd_step), (0, 2, 1))                      # (M, 6, 6)
    P_lin = np.einsum("kij,jl,kml->kim", Phi, P0, Phi)
    smax = np.linalg.svd(Phi, compute_uv=False)[:, 0]
    m_ut = np.einsum("s,ksj->kj", ukf.Wm, Ys)
    D = Ys - m_ut[:, None, :]
    P_ut = np.einsum("s,ksi,ksj->kij", ukf.Wc, D, D)

    M = len(dt_grid)
    out = {k: np.zeros(M) for k in ("theta_ideal", "theta_lin_claim", "theta_lin_actual", "contain_lin",
                                    "theta_ut_claim", "theta_ut_actual", "contain_ut")}
    out.update(dt=dt_grid, smax=smax, nominal=nom)
    for k in range(M):
        umc = _unit(Ymc[k, :, :3])
        c = _unit(umc.mean(axis=0))
        out["theta_ideal"][k] = np.degrees(np.quantile(np.arccos(np.clip(umc @ c, -1, 1)), 0.99))
        for tag, m, P in (("lin", nom[k], P_lin[k]), ("ut", m_ut[k], P_ut[k])):
            r = m[:3]
            u = r / np.linalg.norm(r)
            E = _tangent_basis(u)
            C_sky = E @ P[:3, :3] @ E.T / (r @ r)
            L = _sqrt_psd(0.5 * (C_sky + C_sky.T))
            out[f"theta_{tag}_claim"][k] = np.degrees(np.quantile(np.linalg.norm(_Z @ L.T, axis=1), 0.99))
            out[f"theta_{tag}_actual"][k] = np.degrees(np.quantile(np.arccos(np.clip(umc @ u, -1, 1)), 0.99))
            p = umc @ E.T
            d2 = np.einsum("ij,ij->i", p, np.linalg.solve(C_sky, p.T).T)
            out[f"contain_{tag}"][k] = np.mean(d2 < CHI2_2_99)
    return out
