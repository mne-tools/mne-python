
"""Tools for creating Raw objects from numpy arrays"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..constants import FIFF
from ..meas_info import Info
from ..base import _BaseRaw
from ...utils import verbose, logger
from ...externals.six import string_types


class RawArray(_BaseRaw):
    """Raw object from numpy array

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
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

        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        self.verbose = verbose
        self.cals = cals
        self.rawdir = None
        self.proj = None
        self.comp = None
        self._filenames = list()
        self.preload = True
        self.info = info
        self._data = data
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp,
                                self.last_samp + 1) / info['sfreq']
        self._projectors = list()
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    self.first_samp, self.last_samp,
                    float(self.first_samp) / info['sfreq'],
                    float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')
