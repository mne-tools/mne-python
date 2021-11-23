# -*- coding: utf-8 -*-
# Authors: Samu Taulu <staulu@uw.edu>
#          Eric Larson <larson.eric.d@gmail.com>

# License: BSD-3-Clause

from functools import partial

import numpy as np

from .._ola import _COLA, _Storer
from ..io.pick import _picks_to_idx
from ..surface import _normalize_vectors
from ..utils import logger, verbose


def _svd_cov(cov, data):
    """Use a covariance matrix to compute the SVD faster."""
    # This makes use of mathematical equivalences between PCA and SVD
    # on zero-mean data
    s, u = np.linalg.eigh(cov)
    norm = np.ones((s.size,))
    mask = s > np.finfo(float).eps * s[-1]  # largest is last
    s = np.sqrt(s, out=s)
    norm[mask] = 1. / s[mask]
    u *= norm
    v = np.dot(u.T[mask], data)
    return u, s, v


@verbose
def oversampled_temporal_projection(raw, duration=10., picks=None,
                                    verbose=None):
    """Denoise MEG channels using leave-one-out temporal projection.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to denoise.
    duration : float | str
        The window duration (in seconds; default 10.) to use. Can also
        be "min" to use as short a window as possible.
    %(picks_all_data)s
    %(verbose)s

    Returns
    -------
    raw_clean : instance of Raw
        The cleaned data.

    Notes
    -----
    This algorithm is computationally expensive, and can be several times
    slower than realtime for conventional M/EEG datasets. It uses a
    leave-one-out procedure with parallel temporal projection to remove
    individual sensor noise under the assumption that sampled fields
    (e.g., MEG and EEG) are oversampled by the sensor array
    :footcite:`LarsonTaulu2018`.

    OTP can improve sensor noise levels (especially under visual
    inspection) and repair some bad channels. This noise reduction is known
    to interact with :func:`tSSS <mne.preprocessing.maxwell_filter>` such
    that increasing the ``st_correlation`` value will likely be necessary.

    Channels marked as bad will not be used to reconstruct good channels,
    but good channels will be used to process the bad channels. Depending
    on the type of noise present in the bad channels, this might make
    them usable again.

    Use of this algorithm is covered by a provisional patent.

    .. versionadded:: 0.16

    References
    ----------
    .. footbibliography::
    """
    logger.info('Processing MEG data using oversampled temporal projection')
    picks = _picks_to_idx(raw.info, picks, exclude=())
    picks_good, picks_bad = list(), list()  # these are indices into picks
    for ii, pi in enumerate(picks):
        if raw.ch_names[pi] in raw.info['bads']:
            picks_bad.append(ii)
        else:
            picks_good.append(ii)
    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)

    n_samples = int(round(float(duration) * raw.info['sfreq']))
    if n_samples < len(picks_good) - 1:
        raise ValueError('duration (%s) yielded %s samples, which is fewer '
                         'than the number of channels -1 (%s)'
                         % (n_samples / raw.info['sfreq'], n_samples,
                            len(picks_good) - 1))
    n_overlap = n_samples // 2
    raw_otp = raw.copy().load_data(verbose=False)
    otp = _COLA(
        partial(_otp, picks_good=picks_good, picks_bad=picks_bad),
        _Storer(raw_otp._data, picks=picks),
        len(raw.times), n_samples, n_overlap, raw.info['sfreq'])
    read_lims = list(range(0, len(raw.times), n_samples)) + [len(raw.times)]
    for start, stop in zip(read_lims[:-1], read_lims[1:]):
        logger.info('    Denoising % 8.2f - % 8.2f sec'
                    % tuple(raw.times[[start, stop - 1]]))
        otp.feed(raw[picks, start:stop][0])
    return raw_otp


def _otp(data, picks_good, picks_bad):
    """Perform OTP on one segment of data."""
    if not np.isfinite(data).all():
        raise RuntimeError('non-finite data (inf or nan) found in raw '
                           'instance')
    # demean our data
    data_means = np.mean(data, axis=-1, keepdims=True)
    data -= data_means
    # make a copy
    data_good = data[picks_good]
    # scale the copy that will be used to form the temporal basis vectors
    # so that _orth_svdvals thresholding should work properly with
    # different channel types (e.g., M-EEG)
    norms = _normalize_vectors(data_good)
    cov = np.dot(data_good, data_good.T)
    if len(picks_bad) > 0:
        full_basis = _svd_cov(cov, data_good)[2]
    for mi, pick in enumerate(picks_good):
        # operate on original data
        idx = list(range(mi)) + list(range(mi + 1, len(data_good)))
        # Equivalent: linalg.svd(data[idx], full_matrices=False)[2]
        t_basis = _svd_cov(cov[np.ix_(idx, idx)], data_good[idx])[2]
        x = np.dot(np.dot(data_good[mi], t_basis.T), t_basis)
        x *= norms[mi]
        x += data_means[pick]
        data[pick] = x
    for pick in picks_bad:
        data[pick] = np.dot(np.dot(data[pick], full_basis.T), full_basis)
        data[pick] += data_means[pick]
    return [data]
