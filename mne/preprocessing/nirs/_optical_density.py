# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import numpy as np

from ...io import BaseRaw
from ...io.constants import FIFF
from ...utils import _validate_type, warn, verbose
from ...io.pick import _picks_to_idx
from ..nirs import _channel_frequencies, _check_channels_ordered


@verbose
def optical_density(raw, *, verbose=None):
    r"""Convert NIRS raw data to optical density.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')
    _check_channels_ordered(
        raw.info, np.unique(_channel_frequencies(raw.info, nominal=True)))

    picks = _picks_to_idx(raw.info, 'fnirs_cw_amplitude')

    # The devices measure light intensity. Negative light intensities should
    # not occur. If they do it is likely due to hardware or movement issues.
    # Set all negative values to abs(x), this also has the benefit of ensuring
    # that the means are all greater than zero for the division below.
    if np.any(raw._data[picks] <= 0):
        warn("Negative intensities encountered. Setting to abs(x)")
        min_ = np.inf
        for pi in picks:
            np.abs(raw._data[pi], out=raw._data[pi])
            min_ = min(min_, raw._data[pi].min() or min_)
        # avoid == 0
        for pi in picks:
            np.maximum(raw._data[pi], min_, out=raw._data[pi])

    data_means = np.mean(raw.get_data(picks=picks), axis=1)
    for ii, pi in enumerate(picks):
        raw._data[pi] /= data_means[ii]
        np.log(raw._data[pi], out=raw._data[pi])
        raw._data[pi] *= -1
        raw.info['chs'][pi]['coil_type'] = FIFF.FIFFV_COIL_FNIRS_OD

    return raw
