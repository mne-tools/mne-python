"""Eigenvalue eigenvector modifiers for GED transformers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np


def _compute_mutual_info(covs, sample_weights, evecs):
    class_probas = sample_weights / sample_weights.sum()

    mutual_info = []
    for jj in range(evecs.shape[1]):
        aa, bb = 0, 0
        for cov, prob in zip(covs, class_probas):
            tmp = np.dot(np.dot(evecs[:, jj].T, cov), evecs[:, jj])
            aa += prob * np.log(np.sqrt(tmp))
            bb += prob * (tmp**2 - 1)
        mi = -(aa + (3.0 / 16) * (bb**2))
        mutual_info.append(mi)

    return mutual_info


def _csp_mod(evals, evecs, covs, evecs_order, sample_weights):
    n_classes = sample_weights.shape[0]
    if evecs_order == "mutual_info" and n_classes > 2:
        mutual_info = _compute_mutual_info(covs, sample_weights, evecs)
        ix = np.argsort(mutual_info)[::-1]
    elif evecs_order == "mutual_info" and n_classes == 2:
        ix = np.argsort(np.abs(evals - 0.5))[::-1]
    elif evecs_order == "alternate" and n_classes == 2:
        i = np.argsort(evals)
        ix = np.empty_like(i)
        ix[1::2] = i[: len(i) // 2]
        ix[0::2] = i[len(i) // 2 :][::-1]
    if evals is not None:
        evals = evals[ix]
    evecs = evecs[:, ix]
    return evals, evecs


def _xdawn_mod(evals, evecs, covs=None):
    evals, evecs = _sort_descending(evals, evecs)
    evecs /= np.linalg.norm(evecs, axis=0)
    return evals, evecs


def _ssd_mod(evals, evecs, covs=None):
    evals, evecs = _sort_descending(evals, evecs)
    return evals, evecs


def _spoc_mod(evals, evecs, covs=None):
    evals = evals.real
    evecs = evecs.real
    evals, evecs = _sort_descending(evals, evecs, by_abs=True)
    return evals, evecs


def _sort_descending(evals, evecs, by_abs=False):
    if by_abs:
        ix = np.argsort(np.abs(evals))[::-1]
    else:
        ix = np.argsort(evals)[::-1]
    evals = evals[ix]
    evecs = evecs[:, ix]
    return evals, evecs


def _no_op_mod(evals, evecs, *args, **kwargs):
    return evals, evecs
