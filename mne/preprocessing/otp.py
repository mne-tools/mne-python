# -*- coding: utf-8 -*-
# Authors: Samu Taulu <staulu@uw.edu>
#          Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..io.pick import _picks_to_idx
from ..surface import _normalize_vectors
from ..utils import logger, verbose
from .utils import _get_lims_cola


def _svd_cov(cov, data):
    """Use a covariance matrix to compute the SVD faster."""
    # This makes use of mathematical equivalences between PCA and SVD
    # on zero-mean data
    s, u = linalg.eigh(cov)
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
    (e.g., MEG and EEG) are oversampled by the sensor array [1]_.

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
    .. [1] Larson E, Taulu S (2017). Reducing Sensor Noise in MEG and EEG
           Recordings Using Oversampled Temporal Projection.
           IEEE Transactions on Biomedical Engineering.
    """
    logger.info('Processing MEG data using oversampled temporal projection')
    picks = _picks_to_idx(raw.info, picks, exclude=())
    picks_good, picks_bad = list(), list()
    for pi in picks:
        if raw.ch_names[pi] in raw.info['bads']:
            picks_bad.append(pi)
        else:
            picks_good.append(pi)
    del picks
    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)

    n_samp = int(round(float(duration) * raw.info['sfreq']))
    starts, stops, windows = _get_lims_cola(
        n_samp, len(raw.times), raw.info['sfreq'])
    min_samp = (stops - starts).min()
    if min_samp < len(picks_good) - 1:
        raise ValueError('duration (%s) yielded %s samples, which is fewer '
                         'than the number of channels -1 (%s)'
                         % (n_samp / raw.info['sfreq'], min_samp,
                            len(picks_good) - 1))
    raw_orig = raw.copy()
    raw = raw.copy().load_data(verbose=False)
    raw._data[picks_good] = 0.
    raw._data[picks_bad] = 0.
    for start, stop, window in zip(starts, stops, windows):
        logger.info('    Denoising % 8.2f - % 8.2f sec'
                    % tuple(raw.times[[start, stop - 1]]))
        data_picked = raw_orig[picks_good, start:stop][0]
        if not np.isfinite(data_picked).all():
            raise RuntimeError('non-finite data (inf or nan) found in raw '
                               'instance')
        # demean our slice and our copy
        data_picked_means = np.mean(data_picked, axis=-1, keepdims=True)
        data_picked -= data_picked_means
        # scale the copy that will be used to form the temporal basis vectors
        # so that _orth_svdvals thresholding should work properly with
        # different channel types (e.g., M-EEG)
        norms = _normalize_vectors(data_picked)
        cov = np.dot(data_picked, data_picked.T)
        if len(picks_bad) > 0:
            full_basis = _svd_cov(cov, data_picked)[2]
        for mi, pick in enumerate(picks_good):
            # operate on original data
            idx = list(range(mi)) + list(range(mi + 1, len(data_picked)))
            # Equivalent: linalg.svd(data[idx], full_matrices=False)[2]
            t_basis = _svd_cov(cov[np.ix_(idx, idx)], data_picked[idx])[2]
            x = np.dot(np.dot(data_picked[mi], t_basis.T), t_basis)
            x *= norms[mi]
            x += data_picked_means[mi]
            x *= window
            raw._data[pick, start:stop] += x
        for pick in picks_bad:
            this_data = raw_orig[pick, start:stop][0][0].copy()
            this_mean = this_data.mean()
            this_data -= this_mean
            x = np.dot(np.dot(this_data, full_basis.T), full_basis)
            x += this_mean
            raw._data[pick, start:stop] += window * x
    return raw
