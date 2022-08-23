# Authors: Robert Luke <mail@robertluke.net>
#          Frank Fishburn
#
# License: BSD-3-Clause


import numpy as np

from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _validate_nirs_info


@verbose
def temporal_derivative_distribution_repair(raw, *, verbose=None):
    """Apply temporal derivative distribution repair to data.

    Applies temporal derivative distribution repair (TDDR) to data
    :footcite:`FishburnEtAl2019`. This approach removes baseline shift
    and spike artifacts without the need for any user-supplied parameters.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
         Data with TDDR applied.

    Notes
    -----
    TDDR was initially designed to be used on optical density fNIRS data but
    has been enabled to be applied on hemoglobin concentration fNIRS data as
    well in MNE. We recommend applying the algorithm to optical density fNIRS
    data as intended by the original author wherever possible.

    There is a shorter alias ``mne.preprocessing.nirs.tddr`` that can be used
    instead of this function (e.g. if line length is an issue).

    References
    ----------
    .. footbibliography::
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError('TDDR should be run on optical density or '
                           'hemoglobin data.')
    for pick in picks:
        raw._data[pick] = _TDDR(raw._data[pick], raw.info['sfreq'])

    return raw


# provide a short alias
tddr = temporal_derivative_distribution_repair


# Taken from https://github.com/frankfishburn/TDDR/ (MIT license).
# With permission https://github.com/frankfishburn/TDDR/issues/1.
# The only modification is the name, scipy signal import and flake fixes.
def _TDDR(signal, sample_rate):
    # This function is the reference implementation for the TDDR algorithm for
    #   motion correction of fNIRS data, as described in:
    #
    #   Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    #   Temporal Derivative Distribution Repair (TDDR): A motion correction
    #   method for fNIRS. NeuroImage, 184, 171-179.
    #   https://doi.org/10.1016/j.neuroimage.2018.09.025
    #
    # Usage:
    #   signals_corrected = TDDR( signals , sample_rate );
    #
    # Inputs:
    #   signals: A [sample x channel] matrix of uncorrected optical density or
    #            hemoglobin data
    #   sample_rate: A scalar reflecting the rate of acquisition in Hz
    #
    # Outputs:
    #   signals_corrected: A [sample x channel] matrix of corrected optical
    #   density data
    from scipy.signal import butter, filtfilt
    signal = np.array(signal)
    if len(signal.shape) != 1:
        for ch in range(signal.shape[1]):
            signal[:, ch] = _TDDR(signal[:, ch], sample_rate)
        return signal

    # Preprocess: Separate high and low frequencies
    filter_cutoff = .5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    signal_mean = np.mean(signal)
    signal -= signal_mean
    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        signal_low = filtfilt(fb, fa, signal, padlen=0)
    else:
        signal_low = signal

    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:

        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev)

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        if sigma == 0:
            break
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within
        # machine-precision of old estimate
        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high + signal_mean

    return signal_corrected
