# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import re
import numpy as np
from scipy import linalg

from ...io.pick import _picks_to_idx
from ...utils import fill_doc


@fill_doc
def source_detector_distances(info, picks=None):
    r"""Determine the distance between NIRS source and detectors.

    Parameters
    ----------
    info : Info
        The measurement info.
    %(picks_all)s

    Returns
    -------
    dists : array of float
        Array containing distances in meters.
        Of shape equal to number of channels, or shape of picks if supplied.
    """
    dist = [linalg.norm(ch['loc'][3:6] - ch['loc'][6:9])
            for ch in info['chs']]
    picks = _picks_to_idx(info, picks, exclude=[])
    return np.array(dist, float)[picks]


def short_channels(info, threshold=0.01):
    r"""Determine which NIRS channels are short.

    Channels with a source to detector distance of less than
    ``threshold`` are reported as short. The default threshold is 0.01 m.

    Parameters
    ----------
    info : Info
        The measurement info.
    threshold : float
        The threshold distance for what is considered short in meters.

    Returns
    -------
    short : array of bool
        Array indicating which channels are short.
        Of shape equal to number of channels.
    """
    return source_detector_distances(info) < threshold


def _channel_frequencies(raw):
    """Return the light frequency for each channel."""
    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[], allow_empty=True)
    freqs = np.empty(picks.size, int)
    for ii in picks:
        freqs[ii] = raw.info['chs'][ii]['loc'][9]
    return freqs


def _check_channels_ordered(raw, freqs):
    """Check channels followed expected fNIRS format."""
    # Every second channel should be same SD pair
    # and have the specified light frequencies.
    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[], allow_empty=True)
    if len(picks) % 2 != 0:
        raise ValueError(
            'NIRS channels not ordered correctly. An even number of NIRS '
            'channels is required. %d channels were provided: %r'
            % (len(raw.ch_names), raw.ch_names))
    for ii in picks[::2]:
        ch1_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 raw.info['chs'][ii]['ch_name'])
        ch2_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 raw.info['chs'][ii + 1]['ch_name'])

        if raw.info['chs'][ii]['loc'][9] != \
                float(ch1_name_info.groups()[2]) or \
                raw.info['chs'][ii + 1]['loc'][9] != \
                float(ch2_name_info.groups()[2]):
            raise ValueError(
                'NIRS channels not ordered correctly. Channel name and NIRS'
                ' frequency do not match: %s -> %s & %s -> %s'
                % (raw.info['chs'][ii]['ch_name'],
                   raw.info['chs'][ii]['loc'][9],
                   raw.info['chs'][ii + 1]['ch_name'],
                   raw.info['chs'][ii + 1]['loc'][9]))

        if (ch1_name_info.groups()[0] != ch2_name_info.groups()[0]) or \
           (ch1_name_info.groups()[1] != ch2_name_info.groups()[1]) or \
           (int(ch1_name_info.groups()[2]) != freqs[0]) or \
           (int(ch2_name_info.groups()[2]) != freqs[1]):
            raise ValueError(
                'NIRS channels not ordered correctly. Channels must be ordered'
                ' as source detector pairs with frequencies: %d & %d'
                % (freqs[0], freqs[1]))

    return picks


def _fnirs_check_bads(raw):
    """Check consistent labeling of bads across fnirs optodes."""
    # For an optode pair, if one component (light frequency or chroma) is
    # marked as bad then they all should be. This function checks that all
    # optodes are marked bad consistently.
    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[])
    for ii in picks[::2]:
        bad_opto = set(raw.info['bads']).intersection(raw.ch_names[ii:ii + 2])
        if len(bad_opto) == 1:
            raise RuntimeError('NIRS bad labelling is not consistent')


def _fnirs_spread_bads(raw):
    """Spread bad labeling across fnirs channels."""
    # For an optode if any component (light frequency or chroma) is marked
    # as bad, then they all should be. This function will find any pairs marked
    # as bad and spread the bad marking to all components of the optode pair.
    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[])
    new_bads = list()
    for ii in picks[::2]:
        bad_opto = set(raw.info['bads']).intersection(raw.ch_names[ii:ii + 2])
        if len(bad_opto) > 0:
            new_bads.extend(raw.ch_names[ii:ii + 2])
    raw.info['bads'] = new_bads

    return raw
