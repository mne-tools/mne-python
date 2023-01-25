# Author: Dominik Welke <dominik.welke@web.de>
#
# License: BSD-3-Clause

import numpy as np
from ..constants import FIFF
from ...utils import warn


def set_channelinfo_eyetrack(inst, mapping=None,
                             overwrite=False, verbose=None):
    """
    updates channel information for eyetrack channels
    (if overwrite=True, all channels that are not recognized as proper eyetrack
    channels will be set to MISC)
    this handles the deeper implementation using FIFF types

    user can either rely on automatic extraction from channel name (str), or
    provide a dictionary with entries for each channel name.

    example:
    mapping = {
        "L-GAZE-X": {
            "ch_type": "GAZE",
            "unit": "PIX",
            "eye": "L",
            "xy": "X"
            },
        "R-AREA": {
            "ch_type": "PUPIL",
            "unit": "AU",
            "eye": "R"
            }
        }
    """
    inst.info = _set_fiff_channelinfo_eyetrack(
        inst.info, mapping=mapping, overwrite=overwrite, verbose=verbose)
    return inst


def _eyetrack_channelinfo_from_chname(ch_name):
    """
    guess specs of the eyetrack channel from ch_name:
    ch_type, unit, eye (l/r), axis (x/y)
    """

    # try improving detection by splitting the channel name
    seplist = ['-', '_', '/']  # assume only one is present
    ch_parts = []
    for sep in seplist:
        if sep in ch_name:
            ch_parts = ch_name.lower().split(sep)

    # extract info
    ch_type = (
        'eyetrack_pupil' if (
            'pupil' in ch_parts or
            'area' in ch_parts or
            'pupil' in ch_name.lower() or
            'area' in ch_name.lower()
        ) else 'eyetrack_pos' if (
            'gaze' in ch_parts or 'gaze' in ch_name.lower() or
            'pos' in ch_parts or 'pos' in ch_name.lower() or
            'y' in ch_parts or 'y' in ch_name.lower() or
            'x' in ch_parts or 'x' in ch_name.lower()  # potential problem:
            # "x" is in "px"  too (though this would be pos anyway)
        ) else 'STIM' if (
            'din' in ch_parts or 'din' in ch_name.lower()
        ) else 'MISC')

    ch_unit = (
        FIFF.FIFF_UNIT_PX if (  # px
            'pixel' in ch_name.lower() or
            'pix' in ch_name.lower() or
            'px' in ch_name.lower()
        ) else FIFF.FIFF_UNIT_DEG if (
            'deg' in ch_name.lower()
        ) else FIFF.FIFF_UNIT_MM if (
            'mm' in ch_name.lower()
        ) else FIFF.FIFF_UNIT_NONE if (
            'au' in ch_name.lower()
        ) else FIFF.FIFF_UNIT_NONE)

    ch_eye = (
        1 if (  # right eye
            'r' in ch_parts or
            'right' in ch_parts or
            'right' in ch_name.lower()
        ) else -1 if (  # left eye
            'l' in ch_parts or
            'left' in ch_parts or
            'left' in ch_name.lower()
        ) else np.nan)  # unknown
    ch_xy = (
        1 if (  # Y pos
            'y' in ch_parts or
            'y' in ch_name.lower()
        ) else -1 if (  # X pos
            'x' in ch_parts or  # potentially problematic, as x is in px.
            ('x' in ch_name.lower() and 'px' not in ch_name.lower()) or
            ('x' in ch_name.lower() and 'pix' not in ch_name.lower())
        ) else np.nan)  # unknown

    return ch_type, ch_unit, ch_eye, ch_xy


def _set_unit_channelinfo_eyetrack(info, i_ch, unit, verbose=None):
    ch_name = info['chs'][i_ch]['ch_name']
    # set unit
    if info['chs'][i_ch]['kind'] == FIFF.FIFFV_STIM_CH:
        unit = FIFF.FIFF_UNIT_V
    elif info['chs'][i_ch]['kind'] == FIFF.FIFFV_EYETRACK_CH:
        if unit not in [FIFF.FIFF_UNIT_PX, FIFF.FIFF_UNIT_DEG,
                        FIFF.FIFF_UNIT_MM, FIFF.FIFF_UNIT_NONE]:
            unit = FIFF.FIFF_UNIT_NONE
            if verbose:
                warn("couldn't determine unit for channel '{}'. "
                     "defaults to 'arbitrary units'.".format(ch_name))
    else:
        unit = info['chs'][i_ch]['unit']  # dont change

    info['chs'][i_ch]['unit'] = unit
    return info


def _set_loc_channelinfo_eyetrack(info, i_ch, eye, xy, verbose=None):
    ch_name = info['chs'][i_ch]['ch_name']
    loc = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    # set loc (eye and x/y coordinate)
    if info['chs'][i_ch]['kind'] == FIFF.FIFFV_STIM_CH:
        loc[3], loc[7], loc[11] = 1, 1, 1
    elif info['chs'][i_ch]['kind'] == FIFF.FIFFV_EYETRACK_CH:
        # l/r eye
        if eye not in [-1, 1]:
            eye = np.nan
            if verbose:
                warn("couldn't determine eye for channel '{}'. "
                     "defaults to 'NaN'.".format(ch_name))
        loc[3] = eye
        # x/y pos
        if (xy not in [-1, 1]) and (
                info['chs'][i_ch]['coil_type'] ==
                FIFF.FIFFV_COIL_EYETRACK_POS):
            xy = np.nan
            if verbose:
                warn("couldn't determine X/Y for channel '{}'. "
                     "defaults to 'NaN'.".format(ch_name))
        loc[4] = xy
    else:
        loc = info['chs'][i_ch]['loc']  # dont change

    info['chs'][i_ch]['loc'] = loc.copy()
    return info


def _set_fiff_channelinfo_eyetrack(info, mapping=None,
                                   overwrite=False, verbose=None):
    """
    set the correct FIFF constants for eyetrack channels
    """
    info = info.copy()
    # set correct channel type and location
    for i_ch, ch_name in enumerate(info.ch_names):
        # if no channel dict is provided, try finding out from ch_name
        # possible units:
        # pupil - arbitrary units, mm (diameter)
        # gaze position - px, deg, ...
        # eye - left, right

        # first, guess channel specs from ch_name
        ch_type, ch_unit, ch_eye, ch_xy = _eyetrack_channelinfo_from_chname(
            ch_name)
        # if there is user input, overwrite guesses
        if type(mapping) == dict:
            if ch_name not in mapping.keys():
                if verbose:
                    warn("'channel_dict' was provided, but contains no info "
                         "on channel '{}'".format(ch_name))
            else:
                keys = mapping[ch_name].keys()
                ch_type = mapping[ch_name]['ch_type'] \
                    if 'ch_type' in keys else ch_type
                if 'unit' in keys:
                    ch_unit = (
                        FIFF.FIFF_UNIT_PX if (  # px
                            'pixel' in str(
                                mapping[ch_name]['unit']).lower() or
                            'pix' in str(
                                mapping[ch_name]['unit']).lower() or
                            'px' in str(
                                mapping[ch_name]['unit']).lower()
                        ) else FIFF.FIFF_UNIT_DEG if (
                            'deg' in str(
                                mapping[ch_name]['unit']).lower()
                        ) else FIFF.FIFF_UNIT_MM if (
                            'mm' in str(
                                mapping[ch_name]['unit']).lower()
                        ) else FIFF.FIFF_UNIT_NONE)
                if 'eye' in keys:
                    ch_eye = (
                        1 if (  # right eye
                            mapping[ch_name]['eye'] == 1 or
                            'r' in str(
                                mapping[ch_name]['eye']).lower() or
                            'right' in str(
                                mapping[ch_name]['eye']).lower()
                        ) else -1 if (  # left eye
                            mapping[ch_name]['eye'] == -1 or
                            'l' in str(
                                mapping[ch_name]['eye']).lower() or
                            'left' in str(
                                mapping[ch_name]['eye']).lower()
                        ) else np.nan)  # unknown
                if 'xy' in keys:
                    ch_xy = (
                        1 if (  # Y pos
                            mapping[ch_name]['xy'] == 1 or
                            'y' in str(mapping[ch_name]['xy']).lower()
                        ) else -1 if (  # X pos
                            mapping[ch_name]['xy'] == -1 or
                            'x' in str(mapping[ch_name]['xy']).lower()
                        ) else np.nan)  # unknown

        # now, set coil type and update ch_type
        if ch_type not in ['eyetrack_pupil', 'eyetrack_pos', 'STIM']:
            # try to find out from other entries
            if ch_unit in [FIFF.FIFF_UNIT_PX, FIFF.FIFF_UNIT_DEG]:
                ch_type = 'eyetrack_pos'
            elif (ch_type == 'MISC') and overwrite:  # change ch_type to "MISC"
                info['chs'][i_ch]['kind'] = FIFF.FIFFV_MISC_CH
                info['chs'][i_ch]['coil_type'] = FIFF.FIFFV_COIL_NONE
                info['chs'][i_ch]['unit'] = FIFF.FIFF_UNIT_NONE
                info['chs'][i_ch]['loc'] = np.array(
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                if verbose:
                    warn("couldn't determine channel type for channel '{}'. "
                         "defaults to 'MISC'.".format(ch_name))
                continue
            else:
                continue

        # only moves on for ['eyetrack_pupil', 'eyetrack_pos', 'STIM']
        if ch_type in ['eyetrack_pupil', 'eyetrack_pos']:
            info['chs'][i_ch]['kind'] = FIFF.FIFFV_EYETRACK_CH
        elif ch_type == 'STIM':
            info['chs'][i_ch]['kind'] = FIFF.FIFFV_STIM_CH

        if ch_type == 'eyetrack_pupil':
            info['chs'][i_ch]['coil_type'] = FIFF.FIFFV_COIL_EYETRACK_PUPIL
        elif ch_type == 'eyetrack_pos':
            info['chs'][i_ch]['coil_type'] = FIFF.FIFFV_COIL_EYETRACK_POS
        else:  # for DIN/STIM channel
            info['chs'][i_ch]['coil_type'] = FIFF.FIFFV_COIL_NONE

        # ..then set unit
        info = _set_unit_channelinfo_eyetrack(
            info, i_ch, ch_unit, verbose)

        # ..and loc (eye and x/y coordinate)
        info = _set_loc_channelinfo_eyetrack(
            info, i_ch, ch_eye, ch_xy, verbose)

    return info
