# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import scipy.linalg

from ..cov import Covariance, _smart_eigh, compute_whitener


def _handle_restr_mat(C_ref, restr_type, info, rank):
    """Get restricting matrix to C_ref rank-dimensional principal subspace.

    Returns matrix of shape (rank, n_chs) used to restrict or
    restrict+rescale (whiten) covariances matrices.
    """
    if C_ref is None or restr_type is None:
        return None
    if restr_type == "whitening":
        C_ref_cov = Covariance(C_ref, info.ch_names, info["bads"], info["projs"], 0)
        restr_mat = compute_whitener(
            C_ref_cov, info, rank=rank, pca=True, verbose="error"
        )[0]
    elif restr_type == "restricting":
        restr_mat = _get_restr_mat(C_ref, info, rank)
    else:
        raise ValueError(
            "restr_type should either be callable or one of "
            "('whitening', 'restricting')"
        )
    return restr_mat


def _smart_ged(S, R, restr_mat=None, R_func=None):
    """Perform smart generalized eigenvalue decomposition (GED) of S and R.

    If restr_mat is provided S and R will be restricted to the principal subspace
    of a reference matrix with rank r (see _handle_restr_mat), then GED is performed
    on the restricted S and R and then generalized eigenvectors are transformed back
    to the original space. The g-eigenvectors matrix is of shape (n_chs, r).
    If callable R_func is provided the GED will be performed on (S, R_func(S,R))
    """
    if restr_mat is None:
        evals, evecs = scipy.linalg.eigh(S, R)
        return evals, evecs

    S_restr = restr_mat @ S @ restr_mat.T
    R_restr = restr_mat @ R @ restr_mat.T
    if R_func is not None:
        R_restr = R_func([S_restr, R_restr])
    evals, evecs_restr = scipy.linalg.eigh(S_restr, R_restr)
    evecs = restr_mat.T @ evecs_restr

    return evals, evecs


def _is_cov_symm_pos_semidef(
    cov, rtol=1e-10, atol=1e-11, eval_tol=1e-15, check_pos_semidef=True
):
    is_symm = scipy.linalg.issymmetric(cov, rtol=rtol, atol=atol)
    if not is_symm:
        return False

    if check_pos_semidef:
        # numerically slightly negative evals are considered 0
        is_pos_semidef = np.all(scipy.linalg.eigvalsh(cov) >= -eval_tol)
        return is_pos_semidef

    return True


def _is_cov_pos_def(cov, eval_tol=1e-15):
    is_symm = _is_cov_symm_pos_semidef(cov, check_pos_semidef=False)
    if not is_symm:
        return False
    # numerically slightly positive evals are considered 0
    is_pos_def = np.all(scipy.linalg.eigvalsh(cov) > eval_tol)
    return is_pos_def


def _smart_ajd(covs, restr_mat=None, weights=None):
    """Perform smart approximate joint diagonalization.

    If restr_mat is provided all the cov matrices will be restricted to the
    principal subspace of a reference matrix with rank r (see _handle_restr_mat),
    then GED is performed on the restricted S and R and then generalized eigenvectors
    are transformed back to the original space.
    The matrix of generalized eigenvectors is of shape (n_chs, r).
    """
    from .csp import _ajd_pham

    if restr_mat is None:
        is_all_pos_def = all([_is_cov_pos_def(cov) for cov in covs])
        if not is_all_pos_def:
            raise ValueError(
                "If C_ref is not provided by covariance estimator, "
                "all the covs should be positive definite"
            )
        evecs, D = _ajd_pham(covs)
        return evecs

    else:
        covs = np.array([restr_mat @ cov @ restr_mat.T for cov in covs], float)
        evecs_restr, D = _ajd_pham(covs)
        evecs = _normalize_eigenvectors(evecs_restr.T, covs, weights)
        evecs = restr_mat.T @ evecs
        return evecs


def _get_restr_mat(C, info, rank):
    """Get matrix restricting covariance to rank-dimensional principal subspace of C."""
    _, ref_evecs, mask = _smart_eigh(
        C,
        info,
        rank,
        proj_subspace=True,
        do_compute_rank=False,
        log_ch_type="data",
    )
    restr_mat = ref_evecs[mask]
    return restr_mat


def _normalize_eigenvectors(evecs, covs, sample_weights):
    # Here we apply an euclidean mean. See pyRiemann for other metrics
    mean_cov = np.average(covs, axis=0, weights=sample_weights)

    for ii in range(evecs.shape[1]):
        tmp = np.dot(np.dot(evecs[:, ii].T, mean_cov), evecs[:, ii])
        evecs[:, ii] /= np.sqrt(tmp)
    return evecs
