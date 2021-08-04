"""Tools for creating Raw objects from numpy arrays."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

from ..base import BaseRaw
from ...utils import verbose, logger, _validate_type, fill_doc, _check_option


@fill_doc
class RawArray(BaseRaw):
    """Raw object from numpy array.

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
    %(info_not_none)s Consider using :func:`mne.create_info` to populate
        this structure. This may be modified in place by the class.
    first_samp : int
        First sample offset used during recording (default 0).

        .. versionadded:: 0.12
    copy : {'data', 'info', 'both', 'auto', None}
        Determines what gets copied on instantiation. "auto" (default)
        will copy info, and copy "data" only if necessary to get to
        double floating point precision.

        .. versionadded:: 0.18
    %(verbose)s

    See Also
    --------
    mne.EpochsArray
    mne.EvokedArray
    mne.create_info

    Notes
    -----
    Proper units of measure:
    * V: eeg, eog, seeg, dbs, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc
    """

    @verbose
    def __init__(self, data, info, first_samp=0, copy='auto',
                 verbose=None):  # noqa: D102
        _validate_type(info, 'info', 'info')
        _check_option('copy', copy, ('data', 'info', 'both', 'auto', None))
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        orig_data = data
        data = np.asanyarray(orig_data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError('Data must be a 2D array of shape (n_channels, '
                             'n_samples), got shape %s' % (data.shape,))
        if len(data) != len(info['ch_names']):
            raise ValueError('len(data) (%s) does not match '
                             'len(info["ch_names"]) (%s)'
                             % (len(data), len(info['ch_names'])))
        assert len(info['ch_names']) == info['nchan']
        if copy in ('auto', 'info', 'both'):
            info = info.copy()
        if copy in ('data', 'both'):
            if data is orig_data:
                data = data.copy()
        elif copy != 'auto' and data is not orig_data:
            raise ValueError('data copying was not requested by copy=%r but '
                             'it was required to get to double floating point '
                             'precision' % (copy,))
        logger.info('Creating RawArray with %s data, n_channels=%s, n_times=%s'
                    % (dtype.__name__, data.shape[0], data.shape[1]))
        super(RawArray, self).__init__(info, data,
                                       first_samps=(int(first_samp),),
                                       dtype=dtype, verbose=verbose)
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    self.first_samp, self.last_samp,
                    float(self.first_samp) / info['sfreq'],
                    float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')
