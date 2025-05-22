# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import scipy.linalg

from ..cov import Covariance, _smart_eigh, compute_whitener
from ..defaults import _handle_default
from ..rank import compute_rank
from ..utils import _verbose_safe_false, logger


def _handle_restr_map(C_ref, restr_map, info, rank):
    if C_ref is None or restr_map is None:
        return None
    if restr_map == "whitening":
        projs = info["projs"]
        C_ref_cov = Covariance(C_ref, info.ch_names, info["bads"], projs, 0)
        restr_map = compute_whitener(C_ref_cov, info, rank=rank, pca=True)
    elif restr_map == "ssd":
        restr_map = _get_ssd_whitener(C_ref, rank)
    elif restr_map == "restricting":
        restr_map = _get_restricting_map(C_ref, info, rank)
    elif isinstance(restr_map, callable):
        pass
    else:
        raise ValueError(
            "restr_map should either be callable or one of whitening, ssd, restricting"
        )
    return restr_map


def _smart_ged(S, R, restr_map, R_func=None, mult_order=None):
    """..."""
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


def _ajd_pham(X, eps=1e-6, max_iter=15):
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_channels)
        A set of covariance matrices to diagonalize.
    eps : float, default 1e-6
        The tolerance for stopping criterion.
    max_iter : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_epochs, n_channels, n_channels)
        The set of quasi diagonal matrices.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
           definite Hermitian matrices." SIAM Journal on Matrix Analysis and
           Applications 22, no. 4 (2001): 1136-1152.

    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(max_iter):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.0j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp**2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order="F")
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order="F")
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D


def _smart_ajd(covs, restr_map, weights):
    covs = np.array([restr_map @ cov @ restr_map.T for cov in covs], float)
    evecs_restr, D = _ajd_pham(covs)
    evecs = _normalize_eigenvectors(evecs_restr.T, covs, weights)
    evecs = restr_map.T @ evecs
    return evecs


def _get_restricting_map(C, info, rank):
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


def _get_ssd_rank(S, R, info, rank):
    # find ranks of covariance matrices
    rank_signal = list(
        compute_rank(
            Covariance(
                S,
                info.ch_names,
                list(),
                list(),
                0,
                verbose=_verbose_safe_false(),
            ),
            rank,
            _handle_default("scalings_cov_rank", None),
            info,
        ).values()
    )[0]
    rank_noise = list(
        compute_rank(
            Covariance(
                R,
                info.ch_names,
                list(),
                list(),
                0,
                verbose=_verbose_safe_false(),
            ),
            rank,
            _handle_default("scalings_cov_rank", None),
            info,
        ).values()
    )[0]
    rank = np.min([rank_signal, rank_noise])  # should be identical
    return rank


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
