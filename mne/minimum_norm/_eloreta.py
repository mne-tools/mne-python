# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial

import numpy as np

from ..defaults import _handle_default
from ..fixes import _safe_svd
from ..utils import eigh, logger, sqrtm_sym, warn


def _compute_eloreta(inv, lambda2, options):
    """Compute the eLORETA solution."""
    from .inverse import _compute_reginv, compute_rank_inverse

    options = _handle_default("eloreta_options", options)
    eps, max_iter = options["eps"], options["max_iter"]
    if options["force_equal"] is not None:
        warn(
            'eLORETA method_params["force_equal"] is deprecated, has no effect, '
            "and will be removed in 1.15. Remove it from method_params to avoid "
            "this warning.",
            FutureWarning,
        )

    # Reassemble the gain matrix (should be fast enough)
    if inv["eigen_leads_weighted"]:
        # We can probably relax this if we ever need to
        raise RuntimeError("eLORETA cannot be computed with weighted eigen leads")
    G = np.dot(
        inv["eigen_fields"]["data"].T * inv["sing"], inv["eigen_leads"]["data"].T
    )
    del inv["eigen_leads"]["data"]
    del inv["eigen_fields"]["data"]
    del inv["sing"]
    G = G.astype(np.float64)
    n_nzero = compute_rank_inverse(inv)
    G /= np.sqrt(inv["source_cov"]["data"])
    # Not sqrt: eLORETA cancels half a column scaling, this makes loose act like MNE's
    ori_scale = np.ones(G.shape[1])
    if inv["orient_prior"] is not None:
        ori_scale *= inv["orient_prior"]["data"]
    G *= ori_scale
    # We do not multiply by the depth prior, as eLORETA should compensate for
    # depth bias.
    n_src = inv["nsource"]
    n_chan, n_orient = G.shape
    n_orient //= n_src
    assert n_orient in (1, 3)
    logger.info("    Computing optimized source covariance (eLORETA)...")
    # src, sens, 3
    G_3 = _get_G_3(G, n_orient)
    # The following was adapted under BSD license by permission of Guido Nolte
    # Init so that max_iter=0 gives MNE (source cov orient_prior)
    if n_orient == 1:
        R_shape = (n_src,)
        R = 1.0 / ori_scale
    else:
        R_shape = (n_src, n_orient, n_orient)
        R = np.eye(n_orient) / ori_scale.reshape(n_src, 1, n_orient)
    _this_normalize_R = partial(
        _normalize_R,
        n_nzero=n_nzero,
        n_src=n_src,
        n_orient=n_orient,
    )
    G_R_Gt = _this_normalize_R(G, R, G_3)
    extra = " (this make take a while)" if n_orient == 3 else ""
    logger.info(f"        Fitting up to {max_iter} iterations{extra}...")
    for kk in range(max_iter):
        # 1. Compute inverse of the weights (stabilized) and C
        s, u = eigh(G_R_Gt)
        s = abs(s)
        sidx = np.argsort(s)[::-1][:n_nzero]
        s, u = s[sidx], u[:, sidx]
        with np.errstate(invalid="ignore"):
            s = np.where(s > 0, 1 / (s + lambda2), 0)
        N = np.dot(u * s, u.T)
        del s

        # Update the weights
        R_last = R.copy()
        if n_orient == 1:
            R[:] = 1.0 / np.sqrt((np.dot(N, G) * G).sum(0))
        else:
            NG_3 = (N @ G).reshape(n_chan, n_src, 3).transpose(1, 0, 2)
            M = np.matmul(G_3, NG_3)
            R[:], _ = sqrtm_sym(M, inv=True)
        G_R_Gt = _this_normalize_R(G, R, G_3)

        # Check for weight convergence
        delta = np.linalg.norm(R.ravel() - R_last.ravel()) / np.linalg.norm(
            R_last.ravel()
        )
        logger.debug(
            f"            Iteration {kk + 1} / {max_iter} ...{extra} ({delta:0.1e})"
        )
        if delta < eps:
            logger.info(
                f"        Converged on iteration {kk + 1} ({delta:.2g} < {eps:.2g})"
            )
            break
    else:
        warn(f"eLORETA weight fitting did not converge (>= {eps})")
    del G_R_Gt
    logger.info("        Updating inverse with weighted eigen leads")
    del G_3
    if n_orient == 1:
        R_sqrt = np.sqrt(R)
    else:
        R_sqrt = sqrtm_sym(R)[0]
    assert R_sqrt.shape == R_shape
    A = _R_sqrt_mult(G, R_sqrt)
    del R, G  # the rest will be done in terms of R_sqrt and A
    eigen_fields, sing, eigen_leads = _safe_svd(A, full_matrices=False)
    del A
    inv["sing"] = sing
    inv["reginv"] = _compute_reginv(inv, lambda2)
    inv["eigen_leads_weighted"] = True
    # R was fit to G @ diag(ori_scale), so the source cov is ori_scale * R * ori_scale
    inv["eigen_leads"]["data"] = (
        ori_scale[:, np.newaxis] * _R_sqrt_mult(eigen_leads, R_sqrt).T
    )
    inv["eigen_fields"]["data"] = eigen_fields.T
    # XXX in theory we should set inv['source_cov'] properly.
    # For fixed ori, we can as it's a diagonal matrix. But for free ori, it's a
    # block diagonal 3x3 and we have no efficient way of storing this (and
    # storing a covariance matrix with (20484 * 3) ** 2 elements is not going
    # to work. So let's just set to nan for now.
    # It's not used downstream anyway now that we set
    # eigen_leads_weighted = True.
    inv["source_cov"]["data"].fill(np.nan)
    logger.info("[done]")


def _normalize_R(G, R, G_3, n_nzero, n_src, n_orient):
    """Normalize R so that lambda2 is consistent."""
    if n_orient == 1:
        R_Gt = R[:, np.newaxis] * G.T
    else:
        R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
    G_R_Gt = G @ R_Gt
    norm = np.trace(G_R_Gt) / n_nzero
    G_R_Gt /= norm
    R /= norm
    return G_R_Gt


def _get_G_3(G, n_orient):
    if n_orient == 1:
        return None
    else:
        return G.reshape(G.shape[0], -1, n_orient).transpose(1, 2, 0)


def _R_sqrt_mult(other, R_sqrt):
    """Do other @ R ** 0.5."""
    if R_sqrt.ndim == 1:
        assert other.shape[1] == R_sqrt.size
        out = R_sqrt * other
    else:
        assert R_sqrt.shape[1:3] == (3, 3)
        assert other.shape[1] == np.prod(R_sqrt.shape[:2])
        assert other.ndim == 2
        n_src = R_sqrt.shape[0]
        n_chan = other.shape[0]
        out = (
            np.matmul(R_sqrt, other.reshape(n_chan, n_src, 3).transpose(1, 2, 0))
            .reshape(n_src * 3, n_chan)
            .T
        )
    return out
