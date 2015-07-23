"""Tools for creating Raw objects from numpy arrays"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..base import _BaseRaw
from ...utils import verbose, logger


class RawArray(_BaseRaw):
    """Raw object from numpy array

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.EpochsArray, mne.EvokedArray
    """
    @verbose
    def __init__(self, data, info, verbose=None):
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)

        if data.ndim != 2:
            raise ValueError('Data must be a 2D array of shape (n_channels, '
                             'n_samples')

        logger.info('Creating RawArray with %s data, n_channels=%s, n_times=%s'
                    % (dtype.__name__, data.shape[0], data.shape[1]))

        if len(data) != len(info['ch_names']):
            raise ValueError('len(data) does not match len(info["ch_names"])')
        assert len(info['ch_names']) == info['nchan']
        super(RawArray, self).__init__(info, data, verbose=verbose)
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    self.first_samp, self.last_samp,
                    float(self.first_samp) / info['sfreq'],
                    float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')
