"""Tools for creating Raw objects from numpy arrays."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..base import BaseRaw
from ..meas_info import Info
from ...utils import verbose, logger


class RawArray(BaseRaw):
    """Raw object from numpy array.

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
    info : instance of Info
        Info dictionary. Consider using :func:`mne.create_info` to populate
        this structure. This may be modified in place by the class.
    first_samp : int
        First sample offset used during recording (default 0).

        .. versionadded:: 0.12

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    Proper units of measure:
    * V: eeg, eog, seeg, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc

    See Also
    --------
    mne.EpochsArray
    mne.EvokedArray
    mne.create_info
    """

    @verbose
    def __init__(self, data, info, first_samp=0, verbose=None):  # noqa: D102
        if not isinstance(info, Info):
            raise TypeError('info must be an instance of Info, got %s'
                            % type(info))
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
        if info.get('buffer_size_sec', None) is None:
            info['buffer_size_sec'] = 1.  # reasonable default
        info = info.copy()  # do not modify original info
        super(RawArray, self).__init__(info, data,
                                       first_samps=(int(first_samp),),
                                       dtype=dtype, verbose=verbose)
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    self.first_samp, self.last_samp,
                    float(self.first_samp) / info['sfreq'],
                    float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')
