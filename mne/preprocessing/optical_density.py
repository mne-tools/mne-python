# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from ..io import BaseRaw
from ..io.constants import FIFF
from ..utils import _validate_type
from ..io.pick import _picks_to_idx

from numpy import mean, divide, log, ones


def optical_density(raw):
    r"""Convert NIRS raw data to optical density.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance. Operates in place.

    """
    _validate_type(raw, BaseRaw, 'raw')
    picks = _picks_to_idx(raw.info, 'fnirs_raw')
    data_means = mean(raw.get_data(), 1)
    for ii in picks:
        data = -1.0 * log(divide(raw.get_data(ii).T,
                          ones((len(raw), 1)) * data_means[ii]))
        raw._data[ii, :] = data.T
        raw.info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_FNIRS_OD

    return raw
