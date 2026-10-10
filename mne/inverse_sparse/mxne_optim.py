# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import functools
from math import sqrt

import numpy as np
from scipy import linalg

from ..time_frequency._stft import istft, stft, stft_norm1, stft_norm2
from ..utils import (
    _check_option,
    _get_blas_funcs,
    _validate_type,
    logger,
    sum_squared,
    verbose_static,
    warn,
)
from .mxne_debiasing import compute_bias


@functools.lru_cache(None)
def _get_dgemm():
    return _get_blas_funcs(np.float64, "gemm")


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups (does NOT modify A)."""
    n_positions = A.shape[0] // n_orient
    return np.sum(A.reshape(n_positions, -1) ** 2, axis=1)


def norm_l2inf(A, n_orient, copy=True):
    """L2-inf norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return sqrt(np.max(groups_norm2(A, n_orient)))


def norm_l21(A, n_orient, copy=True):
    """L21 norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


# ---------------------------------------------------------------------------
# Duality gap helpers
# ---------------------------------------------------------------------------

def _primal_l21(M, G, X, active_set, alpha, n_orient):
    """Primal objective for the mixed-norm inverse problem."""
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj, R, nR2, GX


def dgap_l21(M, G, X, active_set, alpha, n_orient):
    """Duality gap for the mixed norm inverse problem."""
    p_obj, R, nR2, GX = _primal_l21(M, G, X, active_set, alpha, n_orient)
    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm if dual_norm > 0 else 1.0
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(R * GX)
    gap = p_obj - d_obj
    return gap, p_obj, d_obj, R


# ---------------------------------------------------------------------------
# L21 solvers
# ---------------------------------------------------------------------------

def _mixed_norm_solver_cd(
    M, G, alpha, lipschitz_constant, maxit=10000, tol=1e-8,
    init=None, n_orient=1, dgap_freq=10,
):
    """Solve L21 inverse problem with coordinate descent."""
    from sklearn.linear_model import MultiTaskLasso

    assert M.ndim == G.ndim and M.shape[0] == G.shape[0]

    clf = MultiTaskLasso(
        alpha=alpha / len(M),
        tol=tol / sum_squared(M),
        fit_intercept=False,
        max_iter=maxit,
        random_state=0,
        warm_start=True,
    )
    if init is not None:
        clf.coef_ = init.T
    else:
        clf.coef_ = np.zeros((G.shape[1], M.shape[1])).T
    clf.fit(G, M)

    X = clf.coef_.T
    active_set = np.any(X, axis=1)
    X = X[active_set]
    gap, p_obj, d_obj, _ = dgap_l21(M, G, X, active_set, alpha, n_orient)
    return X, active_set, p_obj


def _mixed_norm_solver_em(
    M, G, alpha, lipschitz_constant=None, maxit=3000, tol=1e-6,
    init=None, n_orient=1, dgap_freq=10,
):
    """Solve Type-II sparse global inverse problem via EM."""
    n_sensors, n_times = M.shape
    n_sources = G.shape[1]

    if init is not None:
        gamma = np.sum(init ** 2, axis=1)
    else:
        gamma = np.ones(n_sources, dtype=G.dtype) * (alpha / max(n_sources, 1))

    sigma_noise = 1e-3 * np.trace(np.dot(M, M.T)) / n_sensors
    gamma_new = gamma.copy()
    active_set = np.ones(n_sources, dtype=bool)

    for i in range(maxit):
        gamma_old = gamma_new.copy()
        active_idx = np.where(active_set)[0]
        if len(active_idx) == 0:
            break

        G_active = G[:, active_idx]

        # E-step: Sigma_y = G_a diag(gamma) G_a^T + sigma^2 I
        Gamma_G_T = G_active * gamma_new[active_idx][None, :]
        Sigma_y = np.dot(Gamma_G_T, G_active.T)
        Sigma_y.flat[:: n_sensors + 1] += sigma_noise

        try:
            L_mat = linalg.cholesky(Sigma_y, lower=True)
            inv_Sigma_y_M = linalg.cho_solve((L_mat, True), M)
            inv_Sigma_y_G = linalg.cho_solve((L_mat, True), G_active)
        except linalg.LinAlgError:
            logger.debug("EM: matrix singular at iter %d; stopping.", i)
            break

        # M-step: update gamma in a temp array (Jacobi-style)
        gamma_upd = gamma_new.copy()
        for idx_loc, src_idx in enumerate(active_idx):
            g_i = G_active[:, idx_loc]
            mu_i = gamma_new[src_idx] * np.dot(g_i.T, inv_Sigma_y_M)
            mean_sq = np.mean(mu_i ** 2)
            cov_ii = gamma_new[src_idx] - (gamma_new[src_idx] ** 2) * np.dot(
                g_i.T, inv_Sigma_y_G[:, idx_loc]
            )
            gamma_upd[src_idx] = mean_sq + max(0.0, cov_ii)

        gamma_new = gamma_upd
        active_set = gamma_new > tol

        delta = linalg.norm(gamma_new - gamma_old) / (linalg.norm(gamma_old) + 1e-9)
        if delta < tol:
            break

    final_active = np.where(active_set)[0]
    X = np.zeros((n_sources, n_times), dtype=G.dtype)

    if len(final_active) > 0:
        Gamma_G_T = G[:, final_active] * gamma_new[final_active][None, :]
        Sigma_y = np.dot(Gamma_G_T, G[:, final_active].T)
        Sigma_y.flat[:: n_sensors + 1] += sigma_noise
        L_mat = linalg.cholesky(Sigma_y, lower=True)
        X[final_active] = gamma_new[final_active][:, None] * np.dot(
            G[:, final_active].T, linalg.cho_solve((L_mat, True), M)
        )

    X_out = X[active_set]
    p_obj = 0.5 * sum_squared(M - np.dot(G[:, active_set], X_out))
    return X_out, active_set, p_obj


def _bcd(G, X, R, active_set, one_ovr_lc, n_orient, alpha_lc, list_G_j_c):
    """One full pass of block coordinate descent."""
    X_j_new = np.zeros_like(X[:n_orient, :], order="C")
    dgemm = _get_dgemm()

    for j, G_j_c in enumerate(list_G_j_c):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        G_j = G[:, idx]
        X_j = X[idx]

        # X_j_new = (1/L_j) G_j^T R
        dgemm(
            alpha=one_ovr_lc[j], beta=0.0,
            a=R.T, b=G_j, c=X_j_new.T, overwrite_c=True,
        )

        was_non_zero = np.any(X_j)
        if was_non_zero:
            # R += G_j X_j
            dgemm(alpha=1.0, beta=1.0, a=X_j.T, b=G_j_c.T,
                  c=R.T, overwrite_c=True)
            X_j_new += X_j

        block_norm = sqrt(sum_squared(X_j_new))
        if block_norm <= alpha_lc[j]:
            X_j.fill(0.0)
            active_set[idx] = False
        else:
            shrink = max(1.0 - alpha_lc[j] / block_norm, 0.0)
            X_j_new *= shrink
            # R -= G_j X_j_new
            dgemm(alpha=-1.0, beta=1.0, a=X_j_new.T, b=G_j_c.T,
                  c=R.T, overwrite_c=True)
            X_j[:] = X_j_new
            active_set[idx] = True


def _mixed_norm_solver_bcd(
    M, G, alpha, lipschitz_constant, maxit=200, tol=1e-8,
    init=None, n_orient=1, dgap_freq=10, use_accel=True, K=5,
):
    """Solve L21 inverse problem with block coordinate descent."""
    _, n_times = M.shape
    _, n_sources = G.shape
    n_positions = n_sources // n_orient

    if init is None:
        X = np.zeros((n_sources, n_times))
        R = M.copy()
    else:
        X = init.copy()
        R = M - np.dot(G, X)

    E = []
    highest_d_obj = -np.inf
    active_set = np.zeros(n_sources, dtype=bool)

    alpha_lc = alpha / lipschitz_constant
    if use_accel:
        last_K_X = np.empty((K + 1, n_sources, n_times))
        U = np.zeros((K, n_sources * n_times))

    G = np.asfortranarray(G)
    assert R.dtype == np.float64 and G.dtype == np.float64
    one_ovr_lc = 1.0 / lipschitz_constant

    # Pre-extract per-block G columns for dgemm
    list_G_j_c = [
        np.ascontiguousarray(G[:, j * n_orient:(j + 1) * n_orient])
        for j in range(n_positions)
    ]

    for i in range(maxit):
        _bcd(G, X, R, active_set, one_ovr_lc, n_orient, alpha_lc, list_G_j_c)

        if (i + 1) % dgap_freq == 0:
            _, p_obj, d_obj, _ = dgap_l21(
                M, G, X[active_set], active_set, alpha, n_orient
            )
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj
            E.append(p_obj)
            logger.debug(
                "Iteration %d :: p_obj %f :: dgap %f :: n_active %d",
                i + 1, p_obj, gap, np.sum(active_set) // n_orient,
            )
            if gap < tol:
                logger.debug("Convergence reached ! (gap: %g < %g)", gap, tol)
                break

        if use_accel:
            last_K_X[i % (K + 1)] = X

            if i % (K + 1) == K:
                for k in range(K):
                    U[k] = last_K_X[k + 1].ravel() - last_K_X[k].ravel()
                C = U @ U.T
                u, s, _ = np.linalg.svd(C, hermitian=True)
                if s[-1] <= 1e-6 * s[0] or not np.isfinite(s).all():
                    logger.debug("Iteration %d: LinAlg Error", i + 1)
                    continue
                z = ((u * 1.0 / s) @ u.T).sum(0)
                c = z / z.sum()
                X_acc = np.sum(last_K_X[:-1] * c[:, None, None], axis=0)
                _grp_norm2_acc = groups_norm2(X_acc, n_orient)
                active_set_acc = _grp_norm2_acc != 0
                if n_orient > 1:
                    active_set_acc = np.kron(
                        active_set_acc, np.ones(n_orient, dtype=bool)
                    )
                p_obj = _primal_l21(
                    M, G, X[active_set], active_set, alpha, n_orient
                )[0]
                p_obj_acc = _primal_l21(
                    M, G, X_acc[active_set_acc], active_set_acc, alpha, n_orient
                )[0]
                if p_obj_acc < p_obj:
                    X = X_acc
                    active_set = active_set_acc
                    R = M - G[:, active_set] @ X[active_set]

    X = X[active_set]
    return X, active_set, E


# ---------------------------------------------------------------------------
# High-level mixed-norm driver
# ---------------------------------------------------------------------------

@verbose_static()
def mixed_norm_solver(
    M, G, alpha, maxit=3000, tol=1e-8, verbose=None,
    active_set_size=50, debias=True, n_orient=1, solver="auto",
    return_gap=False, dgap_freq=10, active_set_init=None, X_init=None,
):
    """Solve mixed-norm inverse problem with active set selection."""
    n_dipoles = G.shape[1]
    n_positions = n_dipoles // n_orient
    _, n_times = M.shape

    alpha_max = norm_l2inf(np.dot(G.T, M), n_orient, copy=False)
    logger.info("-- ALPHA MAX : %g", alpha_max)
    alpha = float(alpha)

    has_sklearn = True
    try:
        from sklearn.linear_model import MultiTaskLasso  # noqa: F401
    except ImportError:
        has_sklearn = False

    _validate_type(solver, str, "solver")
    _check_option("solver", solver, ("cd", "bcd", "em", "auto"))

    if solver == "auto":
        solver = "cd" if (has_sklearn and n_orient == 1) else "bcd"

    if solver == "cd":
        if n_orient > 1:
            warn("Coordinate descent restricted to fixed orientations. "
                 "Falling back to BCD.")
            solver = "bcd"
        elif not has_sklearn:
            warn("Scikit-learn cannot be found. Falling back to BCD.")
            solver = "bcd"

    if solver == "cd":
        logger.info("Using coordinate descent")
        l21_solver = _mixed_norm_solver_cd
        lc = None
    elif solver == "em":
        logger.info("Using Expectation-Maximization (EM) Type-II solver")
        l21_solver = _mixed_norm_solver_em
        lc = None
    else:
        logger.info("Using block coordinate descent")
        l21_solver = _mixed_norm_solver_bcd
        G = np.asfortranarray(G)
        if n_orient == 1:
            lc = np.sum(G * G, axis=0)
        else:
            lc = np.empty(n_positions)
            for j in range(n_positions):
                G_tmp = G[:, j * n_orient:(j + 1) * n_orient]
                lc[j] = np.linalg.norm(np.dot(G_tmp.T, G_tmp), ord=2)

    X = np.zeros((n_dipoles, n_times), dtype=G.dtype)

    if active_set_size is not None and solver != "em":
        E = []
        highest_d_obj = -np.inf

        if X_init is not None and X_init.shape != (n_dipoles, n_times):
            raise ValueError("Wrong dim for initialized coefficients.")

        active_set = (
            active_set_init.copy()
            if active_set_init is not None
            else np.zeros(n_dipoles, dtype=bool)
        )
        idx_large_corr = np.argsort(groups_norm2(np.dot(G.T, M), n_orient))
        new_active_idx = idx_large_corr[-active_set_size:]
        if n_orient > 1:
            new_active_idx = (
                n_orient * new_active_idx[:, None]
                + np.arange(n_orient)[None, :]
            ).ravel()
        active_set[new_active_idx] = True
        as_size = np.sum(active_set)
        gap = np.inf

        for k in range(maxit):
            if solver == "bcd":
                lc_tmp = lc[active_set[::n_orient]]
            elif solver == "cd":
                lc_tmp = None
            else:
                lc_tmp = 1.01 * np.linalg.norm(G[:, active_set], ord=2) ** 2

            # Build init in the reduced (active-set) space
            X_init_red = None
            if X_init is not None:
                if X_init.shape[0] == n_dipoles:
                    X_init_red = X_init[active_set]
                else:
                    X_init_red = X_init

            X, as_, _ = l21_solver(
                M, G[:, active_set], alpha, lc_tmp,
                maxit=maxit, tol=tol, init=X_init_red,
                n_orient=n_orient, dgap_freq=dgap_freq,
            )
            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            _, p_obj, d_obj, R = dgap_l21(
                M, G, X, active_set, alpha, n_orient
            )
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj
            E.append(p_obj)
            logger.info(
                "Iteration %d :: p_obj %f :: dgap %f :: n_active_start %d "
                ":: n_active_end %d",
                k + 1, p_obj, gap,
                as_size // n_orient, np.sum(active_set) // n_orient,
            )

            if gap < tol:
                logger.info("Convergence reached ! (gap: %g < %g)", gap, tol)
                break

            if k < (maxit - 1):
                idx_large_corr = np.argsort(
                    groups_norm2(np.dot(G.T, R), n_orient)
                )
                new_active_idx = idx_large_corr[-active_set_size:]
                if n_orient > 1:
                    new_active_idx = (
                        n_orient * new_active_idx[:, None]
                        + np.arange(n_orient)[None, :]
                    ).ravel()
                active_set[new_active_idx] = True
                idx_active_set = np.where(active_set)[0]
                as_size = np.sum(active_set)

                # Re-embed X into new active set
                X_init = np.zeros((as_size, n_times), dtype=X.dtype)
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                X_init[idx] = X
            else:
                warn("Did NOT converge ! (gap: %g > %g)", gap, tol)
    else:
        X, active_set, E = l21_solver(
            M, G, alpha, lc, maxit=maxit, tol=tol,
            n_orient=n_orient, init=X_init,
        )

    if return_gap:
        gap = dgap_l21(M, G, X, active_set, alpha, n_orient)[0]

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    logger.info(
        "Final active set size: %d", np.sum(active_set) // n_orient
    )

    return (X, active_set, E, gap) if return_gap else (X, active_set, E)


# ---------------------------------------------------------------------------
# Iterative reweighting (L0.5/L2)
# ---------------------------------------------------------------------------

@verbose_static()
def iterative_mixed_norm_solver(
    M, G, alpha, n_mxne_iter, maxit=3000, tol=1e-8, verbose=None,
    active_set_size=50, debias=True, n_orient=1, dgap_freq=10,
    solver="auto", weight_init=None,
):
    """Solve L0.5/L2 mixed-norm inverse problem with reweighting loops."""
    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2.0 * np.repeat(g(w), n_orient).ravel()

    E = []
    if weight_init is not None and weight_init.shape != (G.shape[1],):
        raise ValueError(
            f"Wrong dimension for weight initialization. Got {weight_init.shape}."
        )
    weights = weight_init if weight_init is not None else np.ones(G.shape[1])
    active_set = weights != 0
    weights = weights[active_set]
    X = np.zeros((G.shape[1], M.shape[1]))

    for k in range(n_mxne_iter):
        X0 = X.copy()
        active_set_0 = active_set.copy()
        G_tmp = G[:, active_set] * weights[np.newaxis, :]
        as_size = (
            active_set_size
            if (active_set_size is not None
                and np.sum(active_set) > (active_set_size * n_orient))
            else None
        )
        X, _active_set, _ = mixed_norm_solver(
            M, G_tmp, alpha, debias=False, n_orient=n_orient,
            maxit=maxit, tol=tol, active_set_size=as_size,
            dgap_freq=dgap_freq, solver=solver,
        )
        logger.info("active set size %d", _active_set.sum() / n_orient)
        if _active_set.sum() > 0:
            active_set[active_set] = _active_set
            X *= weights[_active_set][:, np.newaxis]
            weights = gprime(X)
            p_obj = (
                0.5 * np.linalg.norm(M - np.dot(G[:, active_set], X), "fro") ** 2.0
                + alpha * np.sum(g(X))
            )
            E.append(p_obj)
            if (k >= 1 and np.all(active_set == active_set_0)
                    and np.all(np.abs(X - X0) < tol)):
                logger.info("Convergence reached after %d reweightings!", k)
                break
        else:
            active_set = np.zeros_like(active_set)
            E.append(0.5 * np.linalg.norm(M) ** 2.0)
            break

    if np.any(active_set) and debias:
        bias = compute_bias(M, G[:, active_set], X, n_orient=n_orient)
        X *= bias[:, np.newaxis]

    return X, active_set, E


# ---------------------------------------------------------------------------
# Time-frequency operators
# ---------------------------------------------------------------------------

class _Phi:
    """STFT analysis operator."""

    def __init__(self, wsize, tstep, n_coefs, n_times):
        self.wsize = np.atleast_1d(wsize)
        self.tstep = np.atleast_1d(tstep)
        self.n_coefs = np.atleast_1d(n_coefs)
        self.n_dicts = len(self.tstep)
        self.n_freqs = self.wsize // 2 + 1
        self.n_steps = self.n_coefs // self.n_freqs
        self.n_times = n_times
        self.ops = []
        for ws, ts in zip(self.wsize, self.tstep):
            self.ops.append(
                stft(np.eye(n_times), ws, ts, verbose=False).reshape(n_times, -1)
            )

    def __call__(self, x):
        if self.n_dicts == 1:
            return x @ self.ops[0]
        return np.hstack([x @ op for op in self.ops]) / np.sqrt(self.n_dicts)

    def norm(self, z, ord=2):
        if ord not in (1, 2):
            raise ValueError(
                f"Only supported norm order are 1 and 2. Got ord = {ord}"
            )
        stft_norm = stft_norm1 if ord == 1 else stft_norm2
        norm = 0.0
        if len(self.n_coefs) > 1:
            z_split = np.array_split(
                np.atleast_2d(z), np.cumsum(self.n_coefs)[:-1], axis=1
            )
        else:
            z_split = [np.atleast_2d(z)]
        for i, z_i in enumerate(z_split):
            norm += stft_norm(
                z_i.reshape(-1, self.n_freqs[i], self.n_steps[i])
            )
        return norm


class _PhiT:
    """STFT synthesis operator."""

    def __init__(self, tstep, n_freqs, n_steps, n_times):
        self.tstep = tstep
        self.n_freqs = n_freqs
        self.n_steps = n_steps
        self.n_times = n_times
        self.n_dicts = len(tstep) if isinstance(tstep, np.ndarray) else 1
        self.n_coefs = []
        self.op_re = []
        self.op_im = []
        for nf, ns, ts in zip(self.n_freqs, self.n_steps, self.tstep):
            nc = nf * ns
            self.n_coefs.append(nc)
            eye = np.eye(nc).reshape(nf, ns, nf, ns)
            self.op_re.append(istft(eye, ts, n_times).reshape(nc, n_times))
            self.op_im.append(
                istft(eye * 1j, ts, n_times).reshape(nc, n_times)
            )

    def __call__(self, z):
        if self.n_dicts == 1:
            return z.real @ self.op_re[0] + z.imag @ self.op_im[0]
        x_out = np.zeros((z.shape[0], self.n_times))
        z_split = np.array_split(
            z, np.cumsum(self.n_coefs)[:-1], axis=1
        )
        for this_z, op_re, op_im in zip(z_split, self.op_re, self.op_im):
            x_out += this_z.real @ op_re + this_z.imag @ op_im
        return x_out / np.sqrt(self.n_dicts)


# ---------------------------------------------------------------------------
# TF norms
# ---------------------------------------------------------------------------

def norm_l21_tf(Z, phi, n_orient, w_space=None):
    if Z.shape[0]:
        l21_norm = np.sqrt(
            phi.norm(Z, ord=2).reshape(-1, n_orient).sum(axis=1)
        )
        if w_space is not None:
            l21_norm *= w_space
        return l21_norm.sum()
    return 0.0


def norm_l1_tf(Z, phi, n_orient, w_time):
    if Z.shape[0]:
        n_positions = Z.shape[0] // n_orient
        Z_ = np.sqrt(
            np.sum(
                (np.abs(Z) ** 2.0).reshape((n_orient, -1), order="F"), axis=0
            )
        ).reshape((n_positions, -1), order="F")
        if w_time is not None:
            Z_ *= w_time
        return phi.norm(Z_, ord=1).sum()
    return 0.0


def norm_epsilon(Y, l1_ratio, phi, w_space=1.0, w_time=None):
    freqs_count = np.full(len(Y), 2)
    for i, fc in enumerate(
        np.array_split(freqs_count, np.cumsum(phi.n_coefs)[:-1])
    ):
        fc[: phi.n_steps[i]] = 1
        fc[-phi.n_steps[i]:] = 1

    if w_time is not None:
        nonzero_weights = w_time != 0.0
        Y = Y[nonzero_weights]
        freqs_count = freqs_count[nonzero_weights]
        w_time = w_time[nonzero_weights]

    norm_inf_Y = np.max(Y / w_time) if w_time is not None else np.max(Y)
    if l1_ratio == 1.0:
        return norm_inf_Y
    elif l1_ratio == 0.0:
        return np.sqrt(phi.norm(Y[None, :], ord=2).sum())
    if norm_inf_Y == 0.0:
        return 0.0

    if w_time is not None:
        thresh = (
            l1_ratio * np.max(Y / (w_space * (1.0 - l1_ratio)
                                   + l1_ratio * w_time))
        )
    else:
        thresh = l1_ratio * norm_inf_Y
    idx = Y > thresh
    if idx.sum() == 1:
        return norm_inf_Y

    if w_time is not None:
        idx_sort = np.argsort(Y[idx] / w_time[idx])[::-1]
        w_time = w_time[idx][idx_sort]
    else:
        idx_sort = np.argsort(Y[idx])[::-1]
    Y = Y[idx][idx_sort]
    freqs_count = freqs_count[idx][idx_sort]
    Y = np.repeat(Y, freqs_count)
    if w_time is not None:
        w_time = np.repeat(w_time, freqs_count)

    K = Y.shape[0]
    Y2 = Y ** 2
    if w_time is None:
        p_sum_Y2 = np.cumsum(Y2)
        p_sum_w2 = np.arange(1, K + 1)
        p_sum_Yw = np.cumsum(Y)
        upper = p_sum_Y2 / Y2 - 2.0 * p_sum_Yw / Y + p_sum_w2
        denom = l1_ratio ** 2 * p_sum_w2 - w_space ** 2 * (1.0 - l1_ratio) ** 2
    else:
        w_time2 = w_time ** 2
        p_sum_Y2 = np.cumsum(Y2)
        p_sum_w2 = np.cumsum(w_time2)
        p_sum_Yw = np.cumsum(Y * w_time)
        upper = (
            p_sum_Y2 / (Y / w_time) ** 2
            - 2.0 * p_sum_Yw / (Y / w_time)
            + p_sum_w2
        )
        denom = l1_ratio ** 2 * p_sum_w2 - w_space ** 2 * (1.0 - l1_ratio) ** 2

    upper_greater = np.where(
        upper > w_space ** 2 * (1.0 - l1_ratio) ** 2 / l1_ratio ** 2
    )[0]
    i0 = upper_greater[0] - 1 if upper_greater.size else K - 1

    p_sum_Y2 = p_sum_Y2[i0]
    p_sum_w2 = p_sum_w2[i0]
    p_sum_Yw = p_sum_Yw[i0]

    if np.abs(denom) < 1e-10:
        return p_sum_Y2 / (2.0 * l1_ratio * p_sum_Yw)
    delta = (l1_ratio * p_sum_Yw) ** 2 - p_sum_Y2 * denom
    return (l1_ratio * p_sum_Yw - np.sqrt(delta)) / denom


def norm_epsilon_inf(G, R, phi, l1_ratio, n_orient, w_space=None, w_time=None):
    n_positions = G.shape[1] // n_orient
    GTRPhi = np.abs(phi(np.dot(G.T, R))).reshape((n_orient, -1), order="F")
    GTRPhi = np.linalg.norm(GTRPhi, axis=0).reshape(
        (n_positions, -1), order="F"
    )
    nu = 0.0
    for idx in range(n_positions):
        norm_eps = norm_epsilon(
            GTRPhi[idx], l1_ratio, phi,
            w_space=w_space[idx] if w_space is not None else 1.0,
            w_time=w_time[idx] if w_time is not None else None,
        )
        if norm_eps > nu:
            nu = norm_eps
    return nu


def dgap_l21l1(
    M, G, Z, active_set, alpha_space, alpha_time, phi, phiT,
    n_orient, highest_d_obj, w_space=None, w_time=None,
):
    X = phiT(Z)
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    nR2 = sum_squared(R)
    p_obj = (
        0.5 * nR2
        + alpha_space * norm_l21_tf(
            Z, phi, n_orient,
            w_space[active_set[::n_orient]] if w_space is not None else None,
        )
        + alpha_time * norm_l1_tf(
            Z, phi, n_orient,
            w_time[active_set[::n_orient]] if w_time is not None else None,
        )
    )
    l1_ratio = alpha_time / (alpha_space + alpha_time)
    scaling = min(
        1.0,
        (alpha_space + alpha_time)
        / norm_epsilon_inf(
            G, R, phi, l1_ratio, n_orient,
            w_space=w_space, w_time=w_time,
        ),
    )
    d_obj = max(
        highest_d_obj,
        (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(R * GX),
    )
    return p_obj - d_obj, p_obj, d_obj, R


# ---------------------------------------------------------------------------
# TF BCD solver
# ---------------------------------------------------------------------------

def tf_mixed_norm_solver_bcd(
    M, G, Z, active_set, candidates, alpha_space, alpha_time,
    lipschitz_constant, phi, phiT, *, w_space=None, w_time=None,
    n_orient=1, maxit=200, tol=1e-8, dgap_freq=10, perc=None,
):
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient

    # Reshape G to (n_positions, n_sensors, n_orient)
    Gd = np.asfortranarray(G)
    G = np.ascontiguousarray(
        Gd.T.reshape(n_positions, n_orient, -1).transpose(0, 2, 1)
    )

    R = M.copy()
    active = np.where(active_set[::n_orient])[0]
    for idx in active:
        R -= np.dot(G[idx], phiT(Z[idx]))

    E = []
    alpha_time_lc = (
        alpha_time / lipschitz_constant
        if w_time is None
        else alpha_time * w_time / lipschitz_constant[:, None]
    )
    alpha_space_lc = (
        alpha_space / lipschitz_constant
        if w_space is None
        else alpha_space * w_space / lipschitz_constant
    )
    d_obj = -np.inf

    for i in range(maxit):
        for jj in candidates:
            ids, ide = jj * n_orient, (jj + 1) * n_orient
            G_j, Z_j, active_set_j = G[jj], Z[jj], active_set[ids:ide]
            was_active = np.any(active_set_j)

            GTR = np.dot(G_j.T, R) / lipschitz_constant[jj]
            X_j_new = GTR.copy()
            if was_active:
                R += np.dot(G_j, phiT(Z_j))
                X_j_new += phiT(Z_j)

            if np.linalg.norm(X_j_new, "fro") <= alpha_space_lc[jj]:
                if was_active:
                    Z[jj], active_set_j[:] = 0.0, False
                continue

            Z_j_new = (
                Z_j + phi(GTR) if was_active else phi(GTR)
            )
            col_norm = np.linalg.norm(Z_j_new, axis=0)
            if np.all(col_norm <= alpha_time_lc[jj]):
                Z[jj], active_set_j[:] = 0.0, False
                continue

            shrink = np.maximum(
                1.0 - alpha_time_lc[jj] / np.maximum(col_norm, alpha_time_lc[jj]),
                0.0,
            )
            if w_time is not None:
                shrink[w_time[jj] == 0.0] = 0.0
            Z_j_new *= shrink[np.newaxis, :]

            shape_init = Z_j_new.shape
            row_norm = np.sqrt(phi.norm(Z_j_new, ord=2).sum())
            if row_norm <= alpha_space_lc[jj]:
                Z[jj], active_set_j[:] = 0.0, False
                continue

            Z_j_new *= np.maximum(
                1.0 - alpha_space_lc[jj]
                / np.maximum(row_norm, alpha_space_lc[jj]),
                0.0,
            )
            Z[jj] = Z_j_new.reshape(-1, *shape_init[1:]).copy()
            active_set_j[:] = True
            R -= np.dot(G_j, phiT(Z[jj]))

        if (i + 1) % dgap_freq == 0:
            Zd = np.vstack(
                [Z[pos] for pos in range(n_positions) if np.any(Z[pos])]
            )
            gap, p_obj, d_obj, _ = dgap_l21l1(
                M, Gd, Zd, active_set, alpha_space, alpha_time,
                phi, phiT, n_orient, d_obj,
                w_space=w_space, w_time=w_time,
            )
            E.append(p_obj)
            logger.info(
                "\n    Iteration %d :: n_active %d\n"
                "    dgap %.2e :: p_obj %s :: d_obj %s",
                i + 1, np.sum(active_set) // n_orient, gap, p_obj, d_obj,
            )
            if gap < tol:
                break

        if perc is not None and (
            np.sum(active_set) / float(n_orient) <= perc * n_positions
        ):
            break

    return Z, active_set, E, gap < tol


def _tf_mixed_norm_solver_bcd_active_set(
    M, G, alpha_space, alpha_time, lipschitz_constant, phi, phiT, *,
    Z_init=None, w_space=None, w_time=None, n_orient=1, maxit=200,
    tol=1e-8, dgap_freq=10,
):
    n_sensors, n_times = M.shape
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient
    Z = dict.fromkeys(np.arange(n_positions), 0.0)
    active_set = np.zeros(n_sources, dtype=bool)
    active = []

    if Z_init is not None:
        if Z_init.shape != (n_sources, phi.n_coefs.sum()):
            raise Exception(
                "Z_init must be None or an array with shape "
                "(n_sources, n_coefs)."
            )
        for ii in range(n_positions):
            if np.any(Z_init[ii * n_orient:(ii + 1) * n_orient]):
                active_set[ii * n_orient:(ii + 1) * n_orient] = True
                active.append(ii)
        if len(active):
            Z.update(
                dict(zip(active, np.vsplit(Z_init[active_set], len(active))))
            )

    E = []
    candidates = range(n_positions)
    d_obj = -np.inf

    while True:
        Z_init_dict = dict.fromkeys(np.arange(n_positions), 0.0)
        Z_init_dict.update(dict(zip(active, Z.values())))

        Z, active_set, E_tmp, _ = tf_mixed_norm_solver_bcd(
            M, G, Z_init_dict, active_set, candidates,
            alpha_space, alpha_time, lipschitz_constant, phi, phiT,
            w_space=w_space, w_time=w_time, n_orient=n_orient,
            maxit=1, tol=tol, perc=None,
        )
        E += E_tmp
        active = np.where(active_set[::n_orient])[0]
        Z_init_dict = dict(zip(range(len(active)), [Z[idx] for idx in active]))

        Z, as_, E_tmp, _ = tf_mixed_norm_solver_bcd(
            M, G[:, active_set], Z_init_dict,
            np.ones(len(active) * n_orient, dtype=bool),
            range(len(active)), alpha_space, alpha_time,
            lipschitz_constant[active_set[::n_orient]], phi, phiT,
            w_space=(
                w_space[active_set[::n_orient]]
                if w_space is not None else None
            ),
            w_time=(
                w_time[active_set[::n_orient]]
                if w_time is not None else None
            ),
            n_orient=n_orient, maxit=maxit, tol=tol,
            dgap_freq=dgap_freq, perc=0.5,
        )
        active = np.where(active_set[::n_orient])[0]
        active_set[active_set] = as_.copy()
        E += E_tmp

        Zd = np.vstack(
            [Z[pos] for pos in range(len(Z)) if np.any(Z[pos])]
        )
        gap, p_obj, d_obj, _ = dgap_l21l1(
            M, G, Zd, active_set, alpha_space, alpha_time,
            phi, phiT, n_orient, d_obj,
            w_space, w_time,
        )
        logger.info(
            "\ndgap %.2e :: p_obj %s :: d_obj %s :: n_active %d",
            gap, p_obj, d_obj, np.sum(active_set) // n_orient,
        )
        if gap < tol:
            break

    if active_set.sum():
        Z = np.vstack([Z[pos] for pos in range(len(Z)) if np.any(Z[pos])])
        X = phiT(Z)
    else:
        Z = np.zeros((0, phi.n_coefs.sum()), dtype=np.complex128)
        X = np.zeros((0, n_times))

    return X, Z, active_set, E, gap


# ---------------------------------------------------------------------------
# High-level TF drivers
# ---------------------------------------------------------------------------

@verbose_static()
def tf_mixed_norm_solver(
    M, G, alpha_space, alpha_time, wsize=64, tstep=4, n_orient=1,
    maxit=200, tol=1e-8, active_set_size=None, debias=True,
    return_gap=False, dgap_freq=10, verbose=None,
):
    n_sensors, n_times = M.shape
    _, n_sources = G.shape
    n_positions = n_sources // n_orient
    tstep, wsize = np.atleast_1d(tstep), np.atleast_1d(wsize)
    n_steps = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    phi = _Phi(wsize, tstep, n_steps * n_freqs, n_times)
    phiT = _PhiT(tstep, n_freqs, n_steps, n_times)

    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_j = G[:, j * n_orient:(j + 1) * n_orient]
            lc[j] = np.linalg.norm(np.dot(G_j.T, G_j), ord=2)

    X, Z, active_set, E, gap = _tf_mixed_norm_solver_bcd_active_set(
        M, G, alpha_space, alpha_time, lc, phi, phiT,
        Z_init=None, n_orient=n_orient, maxit=maxit, tol=tol,
        dgap_freq=dgap_freq,
    )

    if np.any(active_set) and debias:
        X *= compute_bias(M, G[:, active_set], X, n_orient=n_orient)[:, np.newaxis]

    return (X, active_set, E, gap) if return_gap else (X, active_set, E)


@verbose_static()
def iterative_tf_mixed_norm_solver(
    M, G, alpha_space, alpha_time, n_tfmxne_iter, wsize=64, tstep=4,
    maxit=3000, tol=1e-8, debias=True, n_orient=1, dgap_freq=10, verbose=None,
):
    n_sensors, n_times = M.shape
    n_sources = G.shape[1]
    n_positions = n_sources // n_orient
    tstep, wsize = np.atleast_1d(tstep), np.atleast_1d(wsize)
    n_steps = np.ceil(n_times / tstep.astype(float)).astype(int)
    n_freqs = wsize // 2 + 1
    phi = _Phi(wsize, tstep, n_steps * n_freqs, n_times)
    phiT = _PhiT(tstep, n_freqs, n_steps, n_times)

    if n_orient == 1:
        lc = np.sum(G * G, axis=0)
    else:
        lc = np.empty(n_positions)
        for j in range(n_positions):
            G_j = G[:, j * n_orient:(j + 1) * n_orient]
            lc[j] = np.linalg.norm(np.dot(G_j.T, G_j), ord=2)

    def g_space(Z):
        return np.sqrt(
            np.sqrt(phi.norm(Z, ord=2).reshape(-1, n_orient).sum(axis=1))
        )

    def g_space_prime_inv(Z):
        return 2.0 * g_space(Z)

    def g_time(Z):
        return np.sqrt(
            np.sqrt(
                np.sum(
                    (np.abs(Z) ** 2.0).reshape((n_orient, -1), order="F"),
                    axis=0,
                )
            ).reshape((-1, Z.shape[1]), order="F")
        )

    def g_time_prime_inv(Z):
        return 2.0 * g_time(Z)

    E = []
    active_set = np.ones(n_sources, dtype=bool)
    Z = np.zeros((n_sources, phi.n_coefs.sum()), dtype=np.complex128)

    for k in range(n_tfmxne_iter):
        active_set_0, Z0 = active_set.copy(), Z.copy()

        w_space = None if k == 0 else 1.0 / g_space_prime_inv(Z)
        if k == 0:
            w_time = None
        else:
            w_time = g_time_prime_inv(Z)
            w_time[w_time == 0.0] = -1.0
            w_time = 1.0 / w_time
            w_time[w_time < 0.0] = 0.0

        X, Z, active_set_, _, _ = _tf_mixed_norm_solver_bcd_active_set(
            M, G[:, active_set], alpha_space, alpha_time,
            lc[active_set[::n_orient]], phi, phiT, Z_init=Z,
            w_space=w_space, w_time=w_time, n_orient=n_orient,
            maxit=maxit, tol=tol, dgap_freq=dgap_freq,
        )
        active_set[active_set] = active_set_   # ← fixed

        if active_set.sum() > 0:
            p_obj = (
                0.5 * np.linalg.norm(
                    M - np.dot(G[:, active_set], X), "fro"
                ) ** 2.0
                + alpha_space * np.sum(g_space(Z.copy()))
                + alpha_time * phi.norm(g_time(Z.copy()), ord=1).sum()
            )
            E.append(p_obj)
            logger.info(
                "Iteration %d: active set size=%d, E=%s",
                k + 1, active_set.sum() // n_orient, p_obj,
            )
            if (np.array_equal(active_set, active_set_0)
                    and np.amax(np.abs(Z - Z0)) < tol):
                break
        else:
            E.append(0.5 * np.linalg.norm(M) ** 2.0)
            break

    if debias and active_set.sum() > 0:
        X *= compute_bias(M, G[:, active_set], X, n_orient=n_orient)[:, np.newaxis]

    return X, active_set, E
