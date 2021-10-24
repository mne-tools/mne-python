# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np

from ...io import BaseRaw
from ...io.constants import FIFF
from ...utils import _validate_type, warn
from ..nirs import source_detector_distances, _channel_frequencies,\
    _check_channels_ordered, _channel_chromophore


def beer_lambert_law(raw, ppf=None):
    r"""Convert NIRS optical density data to haemoglobin concentration.

    Parameters
    ----------
    raw : instance of Raw
        The optical density data.
    ppf : float
        The partial pathlength factor.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    from scipy import linalg
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    if ppf is None:
        ppf = 0.1
        warn('The default value of ppf=0.1 will change to ppf=6 in '
             'v0.25. To utilise the future default '
             'value set ppf=6.', DeprecationWarning)

    freqs = np.unique(_channel_frequencies(raw.info, nominal=True))
    picks = _check_channels_ordered(raw.info, freqs)
    abs_coef = _load_absorption(freqs)
    distances = source_detector_distances(raw.info)
    if (distances == 0).any():
        warn('Source-detector distances are zero, some resulting '
             'concentrations will be zero. Consider setting a montage '
             'with raw.set_montage.')
    if (distances > 0.1).any():
        warn('Source-detector distances are greater than 10 cm. '
             'Large distances will result in invalid data, and are '
             'likely due to optode locations being stored in a '
             ' unit other than meters.')
    rename = dict()
    for ii in picks[::2]:
        EL = abs_coef * distances[ii] * ppf
        iEL = linalg.pinv(EL)

        raw._data[[ii, ii + 1]] = iEL @ raw._data[[ii, ii + 1]] * 1e-3

        # Update channel information
        coil_dict = dict(hbo=FIFF.FIFFV_COIL_FNIRS_HBO,
                         hbr=FIFF.FIFFV_COIL_FNIRS_HBR)
        for ki, kind in enumerate(('hbo', 'hbr')):
            ch = raw.info['chs'][ii + ki]
            ch.update(coil_type=coil_dict[kind], unit=FIFF.FIFF_UNIT_MOL)
            new_name = f'{ch["ch_name"].split(" ")[0]} {kind}'
            rename[ch['ch_name']] = new_name
    raw.rename_channels(rename)

    # Validate the format of data after transformation is valid
    chroma = np.unique(_channel_chromophore(raw.info))
    _check_channels_ordered(raw.info, chroma)
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
    from scipy.io import loadmat
    from scipy.interpolate import interp1d

    extinction_fname = op.join(op.dirname(__file__), '..', '..', 'data',
                               'extinction_coef.mat')
    a = loadmat(extinction_fname)['extinct_coef']

    interp_hbo = interp1d(a[:, 0], a[:, 1], kind='linear')
    interp_hb = interp1d(a[:, 0], a[:, 2], kind='linear')

    ext_coef = np.array([[interp_hbo(freqs[0]), interp_hb(freqs[0])],
                         [interp_hbo(freqs[1]), interp_hb(freqs[1])]])
    abs_coef = ext_coef * 0.2303

    return abs_coef
