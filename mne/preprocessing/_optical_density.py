# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from ..io import BaseRaw
from ..io.constants import FIFF
from ..utils import _validate_type
from ..io.pick import _picks_to_idx

from numpy import mean, abs, log, where


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

    # The devices measure light intensity. Negative light intensities should
    # not occur. If they do it is likely due to hardware or movement issues.
    # Set all negative values to abs(x), this also has the benefit of ensuring
    # that the means are all greater than zero for the division below.
    if len(where(raw._data[picks] <= 0)[0]) > 0:
        # TODO How to throw warning without exiting tests
        print("    Negative intensities encountered. Setting to abs(x)")
        raw._data[picks] = abs(raw._data[picks])

    for ii in picks:
        raw._data[ii] /= data_means[ii]
        log(raw._data[ii], out=raw._data[ii])
        raw.info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_FNIRS_OD

    return raw
