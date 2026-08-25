# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The core logic for this implementation was adapted from the Cedalion project
# (https://github.com/ibs-lab/cedalion), which is originally based on Homer3
# (https://github.com/BUNPC/Homer3).

import numpy as np

from ...io import BaseRaw
from ...utils import _validate_type, verbose
from ..nirs import _validate_nirs_info


def _pad_to_power_2(signal):
    """Pad a 1-D signal to the next power of 2.

    Parameters
    ----------
    signal : array-like, shape (n,)
        Input signal.

    Returns
    -------
    padded : ndarray, shape (2**k,)
        Zero-padded signal.
    original_length : int
        Length of the original signal before padding.
    """
    original_length = len(signal)
    if original_length <= 1:
        n = 1
    else:
        n = int(np.ceil(np.log2(original_length)))
    padded_length = 2**n
    padded = np.zeros(padded_length)
    padded[:original_length] = signal
    return padded, original_length


def _mad(x):
    """Compute Median Absolute Deviation."""
    median = np.median(x)
    return np.median(np.abs(x - median))


def _normalize_signal(signal, wavelet_name, pywt_module):
    """Normalize signal by its noise level using MAD of detail coefficients.

    Implements Homer3's ``NormalizationNoise`` function
    :footcite:`HuppertEtAl2009`.

    Parameters
    ----------
    signal : ndarray, shape (n,)
        Input signal (should already be padded to a power of 2).
    wavelet_name : str
        Wavelet to use (e.g. ``'db2'``).
    pywt_module : module
        The PyWavelets module.

    Returns
    -------
    normalized_signal : ndarray, shape (n,)
        Normalized version of the input signal.
    norm_coef : float
        Multiply ``normalized_signal`` by ``1 / norm_coef`` to recover the
        original scale.
    """
    wvlt = pywt_module.Wavelet(wavelet_name)
    # Homer3 uses qmf(db2, 4) which produces the HIGH-pass decomposition
    # filter from the scaling (low-pass) filter.  In PyWavelets this is dec_hi.
    qmf = np.array(wvlt.dec_hi)

    # Circular convolution matching MATLAB's cconv(signal, qmf, len(signal))
    n = len(signal)
    c = np.real(np.fft.ifft(np.fft.fft(signal, n) * np.fft.fft(qmf, n)))

    # Downsample by 2 — first-level detail coefficients for noise estimation
    y_ds = c[::2]

    median_abs_dev = _mad(y_ds)

    if median_abs_dev != 0:
        norm_coef = 1.0 / (1.4826 * median_abs_dev)
        normalized_signal = signal * norm_coef
    else:
        norm_coef = 1.0
        normalized_signal = signal.copy()

    return normalized_signal, norm_coef


def _process_wavelet_coefficients(coeffs, iqr_factor, signal_length):
    """Zero out outlier wavelet coefficients using IQR thresholding.

    Parameters
    ----------
    coeffs : ndarray, shape (n_padded, n_levels + 1)
        Stacked wavelet coefficient array (first column = approx, rest =
        detail per level).
    iqr_factor : float
        Interquartile-range multiplier for the outlier threshold.
    signal_length : int
        Original (unpadded) signal length used to compute per-block valid
        lengths.

    Returns
    -------
    coeffs : ndarray
        Coefficient array with outliers zeroed out.
    """
    n = coeffs.shape[0]
    n_levels = coeffs.shape[1] - 1

    for j in range(n_levels):
        curr_length = signal_length // (2**j) if j > 0 else signal_length
        n_blocks = 2**j
        block_length = n // n_blocks

        for b in range(n_blocks):
            start_idx = b * block_length
            end_idx = start_idx + block_length
            coeff_block = coeffs[start_idx:end_idx, j + 1]

            valid_coeffs = coeff_block[:curr_length]
            q25, q75 = np.percentile(valid_coeffs, [25, 75])
            iqr_val = q75 - q25

            upper = q75 + iqr_factor * iqr_val
            lower = q25 - iqr_factor * iqr_val

            coeffs[start_idx:end_idx, j + 1] = np.where(
                (coeff_block > upper) | (coeff_block < lower),
                0,
                coeff_block,
            )

    return coeffs


@verbose
def motion_correct_wavelet(raw, iqr=1.5, wavelet="db2", level=4, *, verbose=None):
    """Apply wavelet-based motion correction to fNIRS data.

    Decomposes each channel with a stationary wavelet transform (SWT), zeroes
    out detail coefficients that are statistical outliers (IQR-based), and
    reconstructs the corrected signal.  Specialises in spike removal.

    Based on Homer3 v1.80.2 ``hmrR_MotionCorrectWavelet.m``
    :footcite:`HuppertEtAl2009` and the approach described in
    :footcite:`MolaviDumont2012`.

    Parameters
    ----------
    raw : instance of Raw
        The raw fNIRS data (optical density or hemoglobin).
    iqr : float
        Interquartile-range multiplier used as the outlier threshold for
        wavelet coefficients.  Larger values remove fewer coefficients.  Set
        to ``-1`` to disable thresholding entirely.  Default is ``1.5``.
    wavelet : str
        Mother wavelet name recognised by PyWavelets (e.g. ``'db2'``).
        Default is ``'db2'``.
    level : int
        Number of decomposition levels for the SWT.  Default is ``4``.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        Data with wavelet motion correction applied (copy).

    Notes
    -----
    Requires the ``PyWavelets`` package (``pip install PyWavelets``).

    There is a shorter alias ``mne.preprocessing.nirs.wavelet``
    that can be used instead of this function.

    References
    ----------
    .. footbibliography::
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "PyWavelets is required for wavelet motion correction. "
            "Install it with: pip install PyWavelets"
        ) from exc

    _validate_type(raw, BaseRaw, "raw")
    raw = raw.copy().load_data()
    picks = _validate_nirs_info(raw.info)

    if not len(picks):
        raise RuntimeError(
            "Wavelet motion correction should be run on optical density "
            "or hemoglobin data."
        )

    if iqr < 0:
        return raw

    for pick in picks:
        signal = raw._data[pick].copy()

        # Pad to power of 2 (required by SWT)
        padded_signal, original_length = _pad_to_power_2(signal)

        # Remove DC component
        dc_val = np.mean(padded_signal)
        padded_signal -= dc_val

        # Normalise by estimated noise level
        normalized_signal, norm_coef = _normalize_signal(padded_signal, wavelet, pywt)

        # Stationary wavelet transform
        n_log2 = int(np.log2(len(normalized_signal)))
        actual_level = min(level, n_log2 - 1)
        coeffs = pywt.swt(normalized_signal, wavelet, level=actual_level)

        # Stack into a 2-D array: col 0 = approx, cols 1..L = detail levels
        coeffs_array = np.column_stack([coeffs[0][0]] + [c[1] for c in coeffs])

        # Threshold outlier coefficients
        coeffs_array = _process_wavelet_coefficients(coeffs_array, iqr, original_length)

        # Rebuild list of (approx, detail) tuples for iswt
        coeffs_list = [
            (coeffs_array[:, 0], coeffs_array[:, i])
            for i in range(1, coeffs_array.shape[1])
        ]

        # Reconstruct, denormalise, restore DC and trim to original length
        corrected = pywt.iswt(coeffs_list, wavelet)
        corrected = corrected / norm_coef
        corrected = corrected[:original_length] + dc_val

        raw._data[pick] = corrected

    return raw


# provide a short alias
wavelet = motion_correct_wavelet
