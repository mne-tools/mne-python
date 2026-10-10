# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...io import BaseRaw
from ...utils import _validate_type, _verbose_safe_false, verbose_static
from ..nirs import _channel_frequencies, _validate_nirs_info


@verbose_static("l_freq", "h_freq", "l_trans_bandwidth", "h_trans_bandwidth")
def scalp_coupling_index(
    raw,
    l_freq=0.7,
    h_freq=1.5,
    l_trans_bandwidth=0.3,
    h_trans_bandwidth=0.3,
    verbose=None,
):
    r"""Calculate scalp coupling index.

    This function calculates the scalp coupling index
    :footcite:`pollonini2014auditory`. This is a measure of the quality of the
    connection between the optode and the scalp.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    l_freq : float | None
        For FIR filters, the lower pass-band edge; for IIR filters, the lower
        cutoff frequency. If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper
        cutoff frequency. If None the data are only high-passed.
    l_trans_bandwidth : float | str
        Width of the transition band at the low cut-off frequency in Hz
        (high pass or cutoff 1 in bandpass). Can be "auto"
        (default) to use a multiple of ``l_freq``::

            min(max(l_freq * 0.25, 2), l_freq)

        Only used for ``method='fir'``.
    h_trans_bandwidth : float | str
        Width of the transition band at the high cut-off frequency in Hz
        (low pass or cutoff 2 in bandpass). Can be "auto"
        (default in 0.14) to use a multiple of ``h_freq``::

            min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

        Only used for ``method='fir'``.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

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
        verbose=_verbose_safe_false(),
    ).get_data()

    # Determine number of wavelengths per source-detector pair
    # We use nominal wavelengths as the info structure may contain arbitrary data.
    freqs = _channel_frequencies(raw.info)
    n_wavelengths = len(np.unique(freqs))

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
