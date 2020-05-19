# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np

from ... import pick_types
from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _channel_frequencies, _check_channels_ordered
from ...filter import filter_data


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
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    if not len(pick_types(raw.info, fnirs='fnirs_od')):
        raise RuntimeError('Scalp coupling index '
                           'should be run on optical density data.')

    freqs = np.unique(_channel_frequencies(raw))
    picks = _check_channels_ordered(raw, freqs)

    filtered_data = filter_data(raw._data, raw.info['sfreq'], l_freq, h_freq,
                                picks=picks, verbose=verbose,
                                l_trans_bandwidth=l_trans_bandwidth,
                                h_trans_bandwidth=h_trans_bandwidth)

    sci = np.zeros(picks.shape)
    for ii in picks[::2]:
        c = np.corrcoef(filtered_data[ii], filtered_data[ii + 1])[0][1]
        sci[ii] = c
        sci[ii + 1] = c

    return sci
