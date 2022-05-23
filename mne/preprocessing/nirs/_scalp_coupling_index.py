# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import numpy as np

from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _validate_nirs_info


@verbose
def scalp_coupling_index(raw, l_freq=0.7, h_freq=1.5,
                         l_trans_bandwidth=0.3, h_trans_bandwidth=0.3,
                         verbose=False):
    r"""Calculate scalp coupling index.

    This function calculates the scalp coupling index
    :footcite:`pollonini2014auditory`. This is a measure of the quality of the
    connection between the optode and the scalp.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(l_freq)s
    %(h_freq)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    %(verbose)s

    Returns
    -------
    sci : array of float
        Array containing scalp coupling index for each channel.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, 'raw')
    picks = _validate_nirs_info(
        raw.info, fnirs='od', which='Scalp coupling index')

    raw = raw.copy().pick(picks).load_data()
    zero_mask = np.std(raw._data, axis=-1) == 0
    filtered_data = raw.filter(
        l_freq, h_freq, l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth, verbose=verbose).get_data()

    sci = np.zeros(picks.shape)
    for ii in range(0, len(picks), 2):
        with np.errstate(invalid='ignore'):
            c = np.corrcoef(filtered_data[ii], filtered_data[ii + 1])[0][1]
        if not np.isfinite(c):  # someone had std=0
            c = 0
        sci[ii] = c
        sci[ii + 1] = c
    sci[zero_mask] = 0
    sci = sci[np.argsort(picks)]  # restore original order
    return sci
