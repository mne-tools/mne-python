# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _channel_frequencies, _validate_nirs_info


@verbose
def scalp_coupling_index(
    raw,
    l_freq=0.7,
    h_freq=1.5,
    l_trans_bandwidth=0.3,
    h_trans_bandwidth=0.3,
    verbose=False,
):
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
    _validate_type(raw, BaseRaw, "raw")
    picks = _validate_nirs_info(raw.info, fnirs="od", which="Scalp coupling index")

    raw = raw.copy().pick(picks).load_data()
    zero_mask = np.std(raw._data, axis=-1) == 0
    filtered_data = raw.filter(
        l_freq,
        h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        verbose=verbose,
    ).get_data()

    # Determine number of wavelengths per source-detector pair
    # We use nominal wavelengths as the info structure may contain arbitrary data.
    freqs = _channel_frequencies(raw.info)
    n_wavelengths = len(np.unique(freqs))

    # freqs = np.array([raw.info["chs"][pick]["loc"][9] for pick in picks], float)
    # n_wavelengths = len(set(unique_freqs))

    sci = np.zeros(picks.shape)

    # Calculate all pairwise correlations within each group and use the minimum as SCI
    pair_indices = np.triu_indices(n_wavelengths, k=1)

    for gg in range(0, len(picks), n_wavelengths):
        group_data = filtered_data[gg : gg + n_wavelengths]

        # Calculate pairwise correlations within the group
        correlations = np.zeros(pair_indices[0].shape[0])

        for n, (ii, jj) in enumerate(zip(*pair_indices)):
            with np.errstate(invalid="ignore"):
                c = np.corrcoef(group_data[ii], group_data[jj])[0][1]
            if np.isfinite(c):
                correlations[n] = c

        # Use minimum correlation as SCI
        group_sci = correlations.min()

        # Assign the same SCI value to all channels in the group
        sci[gg : gg + n_wavelengths] = group_sci

    sci[zero_mask] = 0
    sci = sci[np.argsort(picks)]  # restore original order
    return sci
