# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import scipy.linalg

from ..cov import Covariance, _smart_eigh, compute_whitener
from ..utils import logger


def _handle_restr_map(C_ref, restr_type, info, rank):
    """Get restricting map to C_ref rank-dimensional principal subspace.

    Returns matrix of shape (rank, n_chs) used to restrict or
    restrict+rescale (whiten) covariances matrices.
    """
    if C_ref is None or restr_type is None:
        return None
    if restr_type == "whitening":
        projs = info["projs"]
        C_ref_cov = Covariance(C_ref, info.ch_names, info["bads"], projs, 0)
        restr_map = compute_whitener(C_ref_cov, info, rank=rank, pca=True)[0]
    elif restr_type == "ssd":
        restr_map = _get_ssd_whitener(C_ref, rank)
    elif restr_type == "restricting":
        restr_map = _get_restricting_map(C_ref, info, rank)
    elif isinstance(restr_type, callable):
        pass
    else:
        raise ValueError(
            "restr_map should either be callable or one of whitening, ssd, restricting"
        )
    return restr_map


def _smart_ged(S, R, restr_map=None, R_func=None, mult_order=None):
    """Perform smart generalized eigenvalue decomposition (GED) of S and R.

    If restr_map is provided S and R will be restricted to the principal subspace
    of a reference matrix with rank r (see _handle_restr_map), then GED is performed
    on the restricted S and R and then generalized eigenvectors are transformed back
    to the original space. The g-eigenvectors matrix is of shape (n_chs, r).
    If callable R_func is provided the GED will be performed on (S, R_func(S,R))
    """
    if restr_map is None:
        evals, evecs = scipy.linalg.eigh(S, R)
        return evals, evecs

    if mult_order == "ssd":
        S_restr = restr_map @ (S @ restr_map.T)
        R_restr = restr_map @ (R @ restr_map.T)
    else:
        S_restr = restr_map @ S @ restr_map.T
        R_restr = restr_map @ R @ restr_map.T
    if R_func is not None:
        R_restr = R_func([S_restr, R_restr])
    evals, evecs_restr = scipy.linalg.eigh(S_restr, R_restr)
    evecs = restr_map.T @ evecs_restr

    return evals, evecs


def _is_all_pos_def(covs):
    for cov in covs:
        try:
            _ = scipy.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False
    return True


def _smart_ajd(covs, restr_map=None, weights=None):
    """Perform smart approximate joint diagonalization.

    If restr_map is provided all the cov matrices will be restricted to the
    principal subspace of a reference matrix with rank r (see _handle_restr_map),
    then GED is performed on the restricted S and R and then generalized eigenvectors
    are transformed back to the original space.
    The matrix of generalized eigenvectors is of shape (n_chs, r).
    """
    from .csp import _ajd_pham

    if restr_map is None:
        is_all_pos_def = _is_all_pos_def(covs)
        if not is_all_pos_def:
            raise ValueError(
                "If C_ref is not provided by covariance estimator, "
                "all the covs should be positive definite"
            )
        evecs, D = _ajd_pham(covs)
        return evecs

    covs = np.array([restr_map @ cov @ restr_map.T for cov in covs], float)
    evecs_restr, D = _ajd_pham(covs)
    evecs = _normalize_eigenvectors(evecs_restr.T, covs, weights)
    evecs = restr_map.T @ evecs
    return evecs


def _get_restricting_map(C, info, rank):
    """Get map restricting covariance to rank-dimensional principal subspace of C."""
    _, ref_evecs, mask = _smart_eigh(
        C,
        info,
        rank,
        proj_subspace=True,
        do_compute_rank=False,
        log_ch_type="data",
    )
    restr_map = ref_evecs[mask]
    return restr_map


def _normalize_eigenvectors(evecs, covs, sample_weights):
    # Here we apply an euclidean mean. See pyRiemann for other metrics
    mean_cov = np.average(covs, axis=0, weights=sample_weights)

    for ii in range(evecs.shape[1]):
        tmp = np.dot(np.dot(evecs[:, ii].T, mean_cov), evecs[:, ii])
        evecs[:, ii] /= np.sqrt(tmp)
    return evecs


def _get_ssd_whitener(S, rank):
    """Perform dimensionality reduction on the covariance matrices."""
    n_channels = S.shape[0]
    if rank < n_channels:
        eigvals, eigvects = scipy.linalg.eigh(S)
        # sort in descending order
        ix = np.argsort(eigvals)[::-1]
        eigvals = eigvals[ix]
        eigvects = eigvects[:, ix]
        # compute rank subspace projection matrix
        rank_proj = np.matmul(
            eigvects[:, :rank], np.eye(rank) * (eigvals[:rank] ** -0.5)
        )
        logger.info(
            "Projecting covariance of %i channels to %i rank subspace",
            n_channels,
            rank,
        )
    else:
        rank_proj = np.eye(n_channels)
        logger.info("Preserving covariance rank (%i)", rank)

    return rank_proj.T
