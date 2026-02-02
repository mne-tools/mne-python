# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os.path as op

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat

from ..._fiff.constants import FIFF
from ...io import BaseRaw
from ...utils import _validate_type, pinv, warn
from ..nirs import _channel_frequencies, _validate_nirs_info, source_detector_distances


def beer_lambert_law(raw, ppf=6.0):
    r"""Convert NIRS optical density data to haemoglobin concentration.

    Parameters
    ----------
    raw : instance of Raw
        The optical density data.
    ppf : tuple | float
        The partial pathlength factors for each wavelength.

        .. versionchanged:: 1.7
           Support for different factors for the two wavelengths.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(ppf, ("numeric", "array-like"), "ppf")
    ppf = np.array(ppf, float)
    picks = _validate_nirs_info(raw.info, fnirs="od", which="Beer-lambert")

    # Use nominal channel frequencies
    #
    # Notes on implementation:
    # 1. Frequencies are calculated the same way as in nirs._validate_nirs_info().
    # 2. Wavelength values in the info structure may contain actual frequencies,
    #    which may be used for more accurate calculation in the future.
    # 3. nirs._channel_frequencies uses both cw_amplitude and OD data to determine
    #    frequencies, whereas we only need those from OD here. Is there any chance
    #    that they're different?
    # 4. If actual frequencies were used, using np.unique() like below will lead to
    #    errors. Instead, absorption coefficients will need to be calculated for
    #    each individual frequency.
    freqs = _channel_frequencies(raw.info)

    # Get unique wavelengths and determine number of wavelengths
    unique_freqs = np.unique(freqs)
    n_wavelengths = len(unique_freqs)

    # PPF validation for multiple wavelengths
    if ppf.ndim == 0:  # single float
        # same PPF for all wavelengths, shape (n_wavelengths, 1)
        ppf = np.full((n_wavelengths, 1), ppf)
    elif ppf.ndim == 1 and len(ppf) == n_wavelengths:
        # separate ppf for each wavelength
        ppf = ppf[:, np.newaxis]  # shape (n_wavelengths, 1)
    else:
        raise ValueError(
            f"ppf must be a single float or an array-like of length {n_wavelengths} "
            f"(number of wavelengths), got shape {ppf.shape}"
        )

    abs_coef = _load_absorption(unique_freqs)  # shape (n_wavelengths, 2)
    distances = source_detector_distances(raw.info, picks="all")
    bad = ~np.isfinite(distances[picks])
    bad |= distances[picks] <= 0
    if bad.any():
        warn(
            "Source-detector distances are zero or NaN, some resulting "
            "concentrations will be zero. Consider setting a montage "
            "with raw.set_montage."
        )
    distances[picks[bad]] = 0.0
    if (distances[picks] > 0.1).any():
        warn(
            "Source-detector distances are greater than 10 cm. "
            "Large distances will result in invalid data, and are "
            "likely due to optode locations being stored in a "
            " unit other than meters."
        )

    rename = dict()
    channels_to_drop_all = []  # Accumulate all channels to drop

    # Iterate over channel groups ([Si_Di all wavelengths, Sj_Dj all wavelengths, ...])
    for ii in range(0, len(picks), n_wavelengths):
        group_picks = picks[ii : ii + n_wavelengths]
        # Calculate Δc based on the system: ΔOD = E * L * PPF * Δc
        # where E is (n_wavelengths, 2), Δc is (2, n_timepoints)
        # using pseudo-inverse
        EL = abs_coef * distances[group_picks[0]] * ppf
        iEL = pinv(EL)
        conc_data = iEL @ raw._data[group_picks] * 1e-3

        # Replace the first two channels with HbO and HbR
        raw._data[group_picks[:2]] = conc_data[:2]  # HbO, HbR

        # Update channel information
        coil_dict = dict(hbo=FIFF.FIFFV_COIL_FNIRS_HBO, hbr=FIFF.FIFFV_COIL_FNIRS_HBR)
        for ki, kind in zip(group_picks[:2], ("hbo", "hbr")):
            ch = raw.info["chs"][ki]
            ch.update(coil_type=coil_dict[kind], unit=FIFF.FIFF_UNIT_MOL)
            new_name = f"{ch['ch_name'].split(' ')[0]} {kind}"
            rename[ch["ch_name"]] = new_name

        # Accumulate extra wavelength channels to drop (keep only HbO and HbR)
        if n_wavelengths > 2:
            channels_to_drop = group_picks[2:]
            channel_names_to_drop = [raw.ch_names[idx] for idx in channels_to_drop]
            channels_to_drop_all.extend(channel_names_to_drop)

    # Drop all accumulated extra wavelength channels after processing all groups
    if channels_to_drop_all:
        raw.drop_channels(channels_to_drop_all)

    raw.rename_channels(rename)

    # Validate the format of data after transformation is valid
    _validate_nirs_info(raw.info, fnirs="hb")
    return raw


def _load_absorption(freqs):
    """Load molar extinction coefficients."""
    # Data from https://omlc.org/spectra/hemoglobin/summary.html
    # The text was copied to a text file. The text before and
    # after the table was deleted. The the following was run in
    # matlab
    # extinct_coef=importdata('extinction_coef.txt')
    # save('extinction_coef.mat', 'extinct_coef')
    #
    # Returns data as [[HbO2(freq1), Hb(freq1)],
    #                  [HbO2(freq2), Hb(freq2)],
    #                  ...,
    #                  [HbO2(freqN), Hb(freqN)]]
    extinction_fname = op.join(
        op.dirname(__file__), "..", "..", "data", "extinction_coef.mat"
    )
    a = loadmat(extinction_fname)["extinct_coef"]

    interp_hbo = interp1d(a[:, 0], a[:, 1], kind="linear")
    interp_hb = interp1d(a[:, 0], a[:, 2], kind="linear")

    # Build coefficient matrix for all wavelengths
    # Shape: (n_wavelengths, 2) where columns are [HbO2, Hb]
    ext_coef = np.zeros((len(freqs), 2))
    for i, freq in enumerate(freqs):
        ext_coef[i, 0] = interp_hbo(freq)  # HbO2
        ext_coef[i, 1] = interp_hb(freq)  # Hb

    abs_coef = ext_coef * 0.2303
    return abs_coef
