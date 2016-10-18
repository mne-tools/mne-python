# -*- coding: utf-8 -*-
# Authors: Samu Taulu <staulu@uw.edu>
#          Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..cov import _check_scalings_user, _apply_scaling_array
from ..io.pick import pick_info, _picks_by_type, _pick_data_channels
from ..utils import logger, verbose


@verbose
def oversampled_temporal_projection(raw, duration=10., scalings=None,
                                    verbose=None):
    """Denoise MEG channels using oversampled temporal projection

    This algorithm uses a leave-one-out procedure with parallel
    temporal projection to remove individual sensor noise under the
    assumption that sampled fields (e.g., MEG and EEG) are oversampled
    by the sensor array [1]_.

    .. note:: Bad channels are not used or modified by this procedure.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to denoise.
    duration : float
        The window duration (in seconds) to use (default: 10.).
    scalings : dict | None
        If None (default), use ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale different channel types to approximately
        at the same unit.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose)

    Returns
    -------
    raw_clean : instance of Raw
        The cleaned data.

    Notes
    -----
    This algorithm is computationally expensive, and can be several times
    slower than realtime for conventional M/EEG datasets.

    This algorithm is covered by a provisional patent.

    .. versionadded:: 0.14

    References
    ----------
    .. [1] Taulu, S and Larson E (2016). "Reducing noise in
           electromagnetic sensor arrays using oversampled temporal
           projection", presented at the 2016 International Conference
           on Biomagnetism, Seoul, Korea 3 Oct 2016.
    """
    # XXX eventually offer option to treat "bads" by projecting them onto
    # others, but not use them in bases for others?
    logger.info('Processing MEG data using oversampled temporal projection')
    scalings = _check_scalings_user(scalings)
    n_samp = int(round(float(duration) * raw.info['sfreq']))
    del duration
    if n_samp / raw.info['sfreq'] > raw.times[-1]:
        raise ValueError('Effective duration (%s) must be at most the '
                         'duration of the raw instance (%s)'
                         % (n_samp / raw.info['sfreq'], raw.times[-1]))
    raw = raw.copy().load_data()
    picks = list(_pick_data_channels(raw.info, exclude='bads'))
    read_lims = np.arange(0, len(raw.times) + 1, n_samp)  # include endpt
    if read_lims[-1] - read_lims[-2] != n_samp:
        read_lims[-1] = len(raw.times)
    min_samp = np.diff(read_lims).min()
    if (min_samp <= len(picks)).any():
        raise ValueError('duration (%s) yielded %s samples, which is fewer '
                         'than the number of channels (%s)'
                         % (n_samp / raw.info['sfreq'], min_samp, len(picks)))
    pl = 's' if len(read_lims) != 2 else ''
    logger.info('    Processing %s data chunk%s of (at least) %0.1f sec'
                % (len(read_lims) - 1, pl, n_samp / raw.info['sfreq']))
    apply_picks_list = _picks_by_type(pick_info(raw.info, picks))
    for start, stop in zip(read_lims[:-1], read_lims[1:]):
        logger.info('    Denoising % 8.2f - % 8.2f sec'
                    % tuple(raw.times[[start, stop - 1]]))
        data_full = raw._data[:, start:stop]  # view (for correct operation)
        data_picked = data_full[picks]  # not a view (for correct operation)
        if not np.isfinite(data_picked).all():
            raise RuntimeError('non-finite data (inf or nan) found in raw '
                               'instance')
        data_picked_means = np.mean(data_picked, axis=-1)[:, np.newaxis]
        # demean our slice and our copy
        data_full[picks] -= data_picked_means
        data_picked -= data_picked_means
        # scale the copy that will be used to form the temporal basis vectors
        # so that _orth_svdvals thresholding should work properly with
        # different channel types (e.g., M-EEG)
        _apply_scaling_array(data_picked, apply_picks_list, scalings)
        for mi, pick in enumerate(picks):
            # operate on original data
            other_data = data_picked[list(range(mi)) +
                                     list(range(mi + 1, len(data_picked)))].T
            t_basis = linalg.orth(other_data)
            del other_data  # overwritten by _orth_svdvals
            data_full[pick] = np.dot(np.dot(data_full[pick].T, t_basis),
                                     t_basis.T)
        # we don't need to scale back because we only scaled the copied data
        # used to build the basis, but we do need to reset the DC value
        data_full[picks] += data_picked_means
    return raw
