# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import re
import numpy as np

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
    dist = [np.linalg.norm(ch['loc'][3:6] - ch['loc'][6:9])
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


def _channel_frequencies(info):
    """Return the light frequency for each channel."""
    # Only valid for fNIRS data before conversion to haemoglobin
    picks = _picks_to_idx(info, ['fnirs_cw_amplitude', 'fnirs_od'],
                          exclude=[], allow_empty=True)
    freqs = np.empty(picks.size, int)
    for ii in picks:
        freqs[ii] = info['chs'][ii]['loc'][9]
    return freqs


def _channel_chromophore(info):
    """Return the chromophore of each channel."""
    # Only valid for fNIRS data after conversion to haemoglobin
    picks = _picks_to_idx(info, ['hbo', 'hbr'], exclude=[], allow_empty=True)
    chroma = []
    for ii in picks:
        chroma.append(info['ch_names'][ii].split(" ")[1])
    return chroma


def _check_channels_ordered(info, pair_vals):
    """Check channels follow expected fNIRS format."""
    # Every second channel should be same SD pair
    # and have the specified light frequencies.

    # All wavelength based fNIRS data.
    picks_wave = _picks_to_idx(info, ['fnirs_cw_amplitude', 'fnirs_od'],
                               exclude=[], allow_empty=True)
    # All chromophore fNIRS data
    picks_chroma = _picks_to_idx(info, ['hbo', 'hbr'],
                                 exclude=[], allow_empty=True)
    # All continuous wave fNIRS data
    picks_cw = np.hstack([picks_chroma, picks_wave])

    if (len(picks_wave) > 0) & (len(picks_chroma) > 0):
        raise ValueError(
            'MNE does not support a combination of amplitude, optical '
            'density, and haemoglobin data in the same raw structure.')

    if len(picks_cw) % 2 != 0:
        raise ValueError(
            'NIRS channels not ordered correctly. An even number of NIRS '
            f'channels is required. {len(info.ch_names)} channels were'
            f'provided: {info.ch_names}')

    # Ensure wavelength info exists for waveform data
    all_freqs = [info["chs"][ii]["loc"][9] for ii in picks_wave]
    if np.any(np.isnan(all_freqs)):
        raise ValueError(
            'NIRS channels is missing wavelength information in the'
            f'info["chs"] structure. The encoded wavelengths are {all_freqs}.')

    for ii in picks_cw[::2]:
        ch1_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 info['chs'][ii]['ch_name'])
        ch2_name_info = re.match(r'S(\d+)_D(\d+) (\d+)',
                                 info['chs'][ii + 1]['ch_name'])

        if bool(ch2_name_info) & bool(ch1_name_info):

            if info['chs'][ii]['loc'][9] != \
                    float(ch1_name_info.groups()[2]) or \
                    info['chs'][ii + 1]['loc'][9] != \
                    float(ch2_name_info.groups()[2]):
                raise ValueError(
                    'NIRS channels not ordered correctly. '
                    'Channel name and NIRS'
                    ' frequency do not match: %s -> %s & %s -> %s'
                    % (info['chs'][ii]['ch_name'],
                       info['chs'][ii]['loc'][9],
                       info['chs'][ii + 1]['ch_name'],
                       info['chs'][ii + 1]['loc'][9]))

            first_value = int(ch1_name_info.groups()[2])
            second_value = int(ch2_name_info.groups()[2])
            error_word = "frequencies"

        else:
            ch1_name_info = re.match(r'S(\d+)_D(\d+) (\w+)',
                                     info['chs'][ii]['ch_name'])
            ch2_name_info = re.match(r'S(\d+)_D(\d+) (\w+)',
                                     info['chs'][ii + 1]['ch_name'])

            if bool(ch2_name_info) & bool(ch1_name_info):

                first_value = ch1_name_info.groups()[2]
                second_value = ch2_name_info.groups()[2]
                error_word = "chromophore"

                if (first_value not in ["hbo", "hbr"] or
                        second_value not in ["hbo", "hbr"]):
                    raise ValueError(
                        "NIRS channels have specified naming conventions."
                        "Chromophore data must be labeled either hbo or hbr."
                        "Failing channels are "
                        f"{info['chs'][ii]['ch_name']}, "
                        f"{info['chs'][ii + 1]['ch_name']}")

            else:
                raise ValueError(
                    'NIRS channels have specified naming conventions.'
                    'The provided channel names can not be parsed.'
                    f'Channels are {info.ch_names}')

        if (ch1_name_info.groups()[0] != ch2_name_info.groups()[0]) or \
           (ch1_name_info.groups()[1] != ch2_name_info.groups()[1]) or \
           (first_value != pair_vals[0]) or \
           (second_value != pair_vals[1]):
            raise ValueError(
                'NIRS channels not ordered correctly. Channels must be ordered'
                ' as source detector pairs with alternating'
                f' {error_word}: {pair_vals[0]} & {pair_vals[1]}')

    return picks_cw


def _validate_nirs_info(info):
    """Apply all checks to fNIRS info. Works on all continuous wave types."""
    freqs = np.unique(_channel_frequencies(info))
    if freqs.size > 0:
        picks = _check_channels_ordered(info, freqs)
    else:
        picks = _check_channels_ordered(info,
                                        np.unique(_channel_chromophore(info)))
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
