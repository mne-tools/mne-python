# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np

from ...io import BaseRaw
from ...io.constants import FIFF
from ...utils import _validate_type, warn
from ...io.pick import _picks_to_idx


def optical_density(raw):
    r"""Convert NIRS raw data to optical density.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')
    picks = _picks_to_idx(raw.info, 'fnirs_cw_amplitude')
    data_means = np.mean(raw.get_data(), axis=1)

    # The devices measure light intensity. Negative light intensities should
    # not occur. If they do it is likely due to hardware or movement issues.
    # Set all negative values to abs(x), this also has the benefit of ensuring
    # that the means are all greater than zero for the division below.
    if np.any(raw._data[picks] <= 0):
        warn("Negative intensities encountered. Setting to abs(x)")
        raw._data[picks] = np.abs(raw._data[picks])

    for ii in picks:
        raw._data[ii] /= data_means[ii]
        np.log(raw._data[ii], out=raw._data[ii])
        raw._data[ii] *= -1
        raw.info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_FNIRS_OD

    return raw
