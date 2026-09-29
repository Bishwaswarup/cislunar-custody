"""Gaussian-mixture UKF with nonlinearity-triggered splitting (after DeMars, Bishop &
Jah 2013, JGCD 36(4)).

Prediction: each component is propagated with the unscented transform. Its Gaussian
fit gains differential entropy when the flow bends the cloud (the true flow is
volume-preserving, so a linear map would leave the entropy unchanged). If
    dH = 0.5 [log det P_pred - log det P] > split_tol
the component is split BEFORE propagation into 3 moment-preserving children along the
direction of maximum stretching (top right-singular vector of A L, where A is the
statistical linearisation Pyx P^-1 of the flow and L = chol(P)), and the children are
re-checked, up to max_depth levels and max_comp components.
Update: per-component UKF update; weights multiplied by N(r; 0, S). Components
below prune_w are dropped; components closer than merge_d2 (Mahalanobis^2) are merged.
"""
import numpy as np

from ..dynamics.cr3bp import propagate_many
from .common import AnglesModel, process_noise
from .ukf import UKF, _sqrt_psd

# moment-preserving 3-way split of N(0, 1): weights, offsets (in sigma), child sigma
_SPLIT_W = np.array([0.2252, 0.5496, 0.2252])
_SPLIT_SIG = 0.6715
_SPLIT_A = np.sqrt((1.0 - _SPLIT_SIG ** 2) / (2 * _SPLIT_W[0]))     # = 1.1041
_SPLIT_OFF = np.array([-_SPLIT_A, 0.0, _SPLIT_A])


def _logdet(P):
    s, ld = np.linalg.slogdet(P)
    return ld if s > 0 else -np.inf


class GMUKF:
    name = "GM-UKF"

    def __init__(self, mu, model: AnglesModel, q_psd_km2_s3=0.0, split_tol=0.02, max_comp=40,
                 max_depth=3, min_dt_check=0.02, prune_w=1e-6, merge_d2=0.5, rtol=1e-10, atol=1e-12):
        self.mu, self.model, self.q = mu, model, q_psd_km2_s3
        self.ukf = UKF(mu, model, q_psd_km2_s3, rtol=rtol, atol=atol)
        self.split_tol, self.max_comp, self.max_depth = split_tol, max_comp, max_depth
        self.min_dt_check, self.prune_w, self.merge_d2 = min_dt_check, prune_w, merge_d2
        self.rtol, self.atol = rtol, atol

    # ---- belief = (W (k,), M (k, 6), P (k, 6, 6)) ----
    def init(self, x0, P0):
        return (np.array([1.0]), np.array([x0], float), np.array([P0], float))

    def size(self, b):
        return len(b[0])

    def moments(self, b):
        W, M, P = b
        m = W @ M
        D = M - m
        C = np.einsum("k,kij->ij", W, P) + (W[:, None] * D).T @ D
        return m, 0.5 * (C + C.T)

    @staticmethod
    def split(w, m, P, A=None):
        """3 moment-preserving children along the max-stretch direction of A (or the
        largest principal axis of P when A is None)."""
        L = _sqrt_psd(P)
        r = np.linalg.svd(L if A is None else A @ L)[2][0]
        u = L @ r                                       # 1-sigma vector
        Pc = P - (1.0 - _SPLIT_SIG ** 2) * np.outer(u, u)
        return ([w * wi for wi in _SPLIT_W], [m + a * u for a in _SPLIT_OFF], [Pc] * 3)

    def predict_belief(self, b, dt):
        if dt == 0.0:
            return b
        W, M, P = [list(x) for x in b]
        check = dt >= self.min_dt_check
        done_W, done_M, done_P = [], [], []
        items = list(zip(W, M, P))
        depth = 0
        n_sp = 2 * 6 + 1
        Qd = process_noise(dt, self.mu, self.q)
        while items:
            X = np.vstack([self.ukf.sigma_points(m, Pk) for _, m, Pk in items])
            Y = propagate_many(X, dt, self.mu, self.rtol, self.atol).reshape(len(items), n_sp, 6)
            X = X.reshape(len(items), n_sp, 6)
            nxt = []
            for (w, m, Pk), Xi, Yi in zip(items, X, Y):
                ym = self.ukf.Wm @ Yi
                dY, dX = Yi - ym, Xi - m
                Pp = (self.ukf.Wc[:, None] * dY).T @ dY
                n_total = len(done_W) + len(nxt) + len(items)
                if check and depth < self.max_depth and n_total + 2 <= self.max_comp:
                    dH = 0.5 * (_logdet(Pp) - _logdet(Pk))
                    if dH > self.split_tol:
                        Pyx = (self.ukf.Wc[:, None] * dY).T @ dX
                        A = np.linalg.solve(Pk.T, Pyx.T).T
                        for c in zip(*self.split(w, m, Pk, A)):
                            nxt.append(c)
                        continue
                done_W.append(w)
                done_M.append(ym)
                done_P.append(0.5 * (Pp + Pp.T) + Qd)
            items = nxt
            depth += 1
        return (np.array(done_W), np.array(done_M), np.array(done_P))

    def update_belief(self, b, meas):
        W, M, P = b
        logw, Ms, Ps = [], [], []
        for w, m, Pk in zip(W, M, P):
            x, Pn, r, S = self.ukf.update(m, Pk, meas)
            _, ld = np.linalg.slogdet(2 * np.pi * S)
            logw.append(np.log(w) - 0.5 * (r @ np.linalg.solve(S, r)) - 0.5 * ld)
            Ms.append(x)
            Ps.append(Pn)
        logw = np.array(logw)
        Wn = np.exp(logw - logw.max())
        Wn /= Wn.sum()
        return self._reduce(Wn, np.array(Ms), np.array(Ps))

    def _reduce(self, W, M, P):
        keep = W > self.prune_w * W.max()
        W, M, P = W[keep] / W[keep].sum(), M[keep], P[keep]
        order = np.argsort(-W)
        used = np.zeros(len(W), bool)
        oW, oM, oP = [], [], []
        for i in order:
            if used[i]:
                continue
            d = M - M[i]
            d2 = np.einsum("kj,kj->k", d, np.linalg.solve(P[i], d.T).T)
            grp = np.where((~used) & (d2 < self.merge_d2))[0]
            used[grp] = True
            wg = W[grp]
            ws = wg.sum()
            mg = (wg @ M[grp]) / ws
            dg = M[grp] - mg
            Pg = (np.einsum("k,kij->ij", wg, P[grp]) + (wg[:, None] * dg).T @ dg) / ws
            oW.append(ws)
            oM.append(mg)
            oP.append(0.5 * (Pg + Pg.T))
        # hard cap: keep the heaviest max_comp components
        oW, oM, oP = np.array(oW), np.array(oM), np.array(oP)
        if len(oW) > self.max_comp:
            top = np.argsort(-oW)[: self.max_comp]
            oW, oM, oP = oW[top], oM[top], oP[top]
        return (oW / oW.sum(), oM, oP)
