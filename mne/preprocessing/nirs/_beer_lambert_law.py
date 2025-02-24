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
from ..nirs import _validate_nirs_info, source_detector_distances


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
    if ppf.ndim == 0:  # upcast single float to shape (2,)
        ppf = np.array([ppf, ppf])
    if ppf.shape != (2,):
        raise ValueError(
            f"ppf must be float or array-like of shape (2,), got shape {ppf.shape}"
        )
    ppf = ppf[:, np.newaxis]  # shape (2, 1)
    picks = _validate_nirs_info(raw.info, fnirs="od", which="Beer-lambert")
    # This is the one place we *really* need the actual/accurate frequencies
    freqs = np.array([raw.info["chs"][pick]["loc"][9] for pick in picks], float)
    abs_coef = _load_absorption(freqs)
    distances = source_detector_distances(raw.info, picks="all")
    bad = ~np.isfinite(distances[picks])
    bad |= distances[picks] <= 0
    if bad.any():
        warn(
            "Source-detector distances are zero on NaN, some resulting "
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
    for ii, jj in zip(picks[::2], picks[1::2]):
        EL = abs_coef * distances[ii] * ppf
        iEL = pinv(EL)

        raw._data[[ii, jj]] = iEL @ raw._data[[ii, jj]] * 1e-3

        # Update channel information
        coil_dict = dict(hbo=FIFF.FIFFV_COIL_FNIRS_HBO, hbr=FIFF.FIFFV_COIL_FNIRS_HBR)
        for ki, kind in zip((ii, jj), ("hbo", "hbr")):
            ch = raw.info["chs"][ki]
            ch.update(coil_type=coil_dict[kind], unit=FIFF.FIFF_UNIT_MOL)
            new_name = f"{ch['ch_name'].split(' ')[0]} {kind}"
            rename[ch["ch_name"]] = new_name
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
    #                  [HbO2(freq2), Hb(freq2)]]
    extinction_fname = op.join(
        op.dirname(__file__), "..", "..", "data", "extinction_coef.mat"
    )
    a = loadmat(extinction_fname)["extinct_coef"]

    interp_hbo = interp1d(a[:, 0], a[:, 1], kind="linear")
    interp_hb = interp1d(a[:, 0], a[:, 2], kind="linear")

    ext_coef = np.array(
        [
            [interp_hbo(freqs[0]), interp_hb(freqs[0])],
            [interp_hbo(freqs[1]), interp_hb(freqs[1])],
        ]
    )
    abs_coef = ext_coef * 0.2303

    return abs_coef
