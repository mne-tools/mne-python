# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

import re as re
import numpy as np
from scipy import linalg

from ...io import BaseRaw
from ...io.pick import _picks_to_idx
from ...io.constants import FIFF
from ...utils import _validate_type
from ..nirs import source_detector_distances


def beer_lambert_law(raw, ppf=0.1):
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
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, 'raw')

    freqs = np.unique(_channel_frequencies(raw))
    picks = _check_channels_ordered(raw, freqs)
    abs_coef = _load_absorption(freqs)
    distances = source_detector_distances(raw.info)

    for ii in picks[::2]:

        EL = abs_coef * distances[ii] * ppf
        iEL = linalg.pinv(EL)

        raw._data[[ii, ii + 1]] = (raw._data[[ii, ii + 1]].T @ iEL.T).T * 1e-3

        # Update channel information
        coil_dict = dict(hbo=FIFF.FIFFV_COIL_FNIRS_HBO,
                         hbr=FIFF.FIFFV_COIL_FNIRS_HBR)
        for ki, kind in enumerate(('hbo', 'hbr')):
            ch = raw.info['chs'][ii + ki]
            ch.update(coil_type=coil_dict[kind], unit=FIFF.FIFF_UNIT_MOL)
            raw.rename_channels({
                ch['ch_name']: '%s %s' % (ch['ch_name'][:-4], kind)})

    return raw


def _channel_frequencies(raw):
    """Return the light frequency for each channel."""
    picks = _picks_to_idx(raw.info, 'fnirs_od')
    freqs = np.empty(picks.size, int)
    for ii in picks:
        freqs[ii] = raw.info['chs'][ii]['loc'][9]
    return freqs


def _check_channels_ordered(raw, freqs):
    """Check channels followed expected fNIRS format."""
    # Every second channel should be same SD pair
    # and have the specified light frequencies.
    picks = _picks_to_idx(raw.info, 'fnirs_od')
    for ii in picks[::2]:
        ch1_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 raw.info['chs'][ii]['ch_name'])
        ch2_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 raw.info['chs'][ii + 1]['ch_name'])

        if (ch1_name_info.groups()[0] != ch2_name_info.groups()[0]) or \
           (ch1_name_info.groups()[1] != ch2_name_info.groups()[1]) or \
           (int(ch1_name_info.groups()[2]) != freqs[0]) or \
           (int(ch2_name_info.groups()[2]) != freqs[1]):
            raise RuntimeError('NIRS channels not ordered correctly')

    return picks


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
