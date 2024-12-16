# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..._fiff.constants import FIFF
from ...io import BaseRaw
from ...utils import _validate_type, verbose, warn
from ..nirs import _validate_nirs_info


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
    _validate_type(raw, BaseRaw, "raw")
    picks = _validate_nirs_info(raw.info, fnirs="cw_amplitude")

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

    for pi in picks:
        data_mean = np.mean(raw._data[pi])
        raw._data[pi] /= data_mean
        np.log(raw._data[pi], out=raw._data[pi])
        raw._data[pi] *= -1
        raw.info["chs"][pi]["coil_type"] = FIFF.FIFFV_COIL_FNIRS_OD

    return raw
