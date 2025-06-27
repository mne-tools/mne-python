"""Eigenvalue eigenvector modifiers for GED transformers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..time_frequency import psd_array_welch
from ..utils import _time_mask


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
    sorter = ix
    return evals, evecs, sorter


def _xdawn_mod(evals, evecs, covs=None):
    evals, evecs, sorter = _sort_descending(evals, evecs)
    evecs /= np.linalg.norm(evecs, axis=0)
    return evals, evecs, sorter


def _get_spectral_ratio(ssd_sources, sfreq, n_fft, freqs_signal, freqs_noise):
    """Get the spectal signal-to-noise ratio for each spatial filter.

    Spectral ratio measure for best n_components selection
    See :footcite:`NikulinEtAl2011`, Eq. (24).

    Returns
    -------
    spec_ratio : array, shape (n_channels)
        Array with the sprectal ratio value for each component.
    sorter_spec : array, shape (n_channels)
        Array of indices for sorting spec_ratio.

    References
    ----------
    .. footbibliography::
    """
    psd, freqs = psd_array_welch(ssd_sources, sfreq=sfreq, n_fft=n_fft)
    sig_idx = _time_mask(freqs, *freqs_signal)
    noise_idx = _time_mask(freqs, *freqs_noise)
    if psd.ndim == 3:
        mean_sig = psd[:, :, sig_idx].mean(axis=2).mean(axis=0)
        mean_noise = psd[:, :, noise_idx].mean(axis=2).mean(axis=0)
        spec_ratio = mean_sig / mean_noise
    else:
        mean_sig = psd[:, sig_idx].mean(axis=1)
        mean_noise = psd[:, noise_idx].mean(axis=1)
        spec_ratio = mean_sig / mean_noise
    sorter_spec = spec_ratio.argsort()[::-1]
    return spec_ratio, sorter_spec


def _ssd_mod(
    evals,
    evecs,
    covs,
    X,
    picks,
    sfreq,
    n_fft,
    freqs_signal,
    freqs_noise,
    sort_by_spectral_ratio,
):
    evals, evecs, sorter = _sort_descending(evals, evecs)
    if sort_by_spectral_ratio:
        # We assume that ordering by spectral ratio is more important
        # than the initial ordering.
        filters = evecs.T
        ssd_sources = filters @ X[..., picks, :]
        _, sorter_spec = _get_spectral_ratio(
            ssd_sources, sfreq, n_fft, freqs_signal, freqs_noise
        )
        evecs = evecs[:, sorter_spec]
        evals = evals[sorter_spec]
        sorter = sorter_spec
    return evals, evecs, sorter


def _spoc_mod(evals, evecs, covs=None):
    evals = evals.real
    evecs = evecs.real
    evals, evecs, sorter = _sort_descending(evals, evecs, by_abs=True)
    return evals, evecs, sorter


def _sort_descending(evals, evecs, by_abs=False):
    if by_abs:
        ix = np.argsort(np.abs(evals))[::-1]
    else:
        ix = np.argsort(evals)[::-1]
    evals = evals[ix]
    evecs = evecs[:, ix]
    sorter = ix
    return evals, evecs, sorter


def _no_op_mod(evals, evecs, *args, **kwargs):
    return evals, evecs, None
