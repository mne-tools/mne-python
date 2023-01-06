# Author: Dominik Welke <dominik.welke@web.de>
#
# License: BSD-3-Clause

import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..constants import FIFF

from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, warn


def set_channelinfo_eyetrack(inst, channel_dict=None):
    """
    updates channel information for eyetrack channels
    (channel type has to be set to "eyetrack" before, as other types will be
    ignored)
    this handles the deeper implementation using FIFF types

    user can either rely on automatic extraction from channel name (str), or
    provide a dictionary with entries for each channel name (e.g. "R-AREA")

    example:
    channel_dict = {
        "L-GAZE-X": {
            "ch_type": "GAZE",
            "ch_unit": "PIX",
            "ch_eye": "L",
            "ch_xy": "X"
            },
        "R-AREA": {
            "ch_type": "PUPIL",
            "ch_unit": "AU",
            "ch_eye": "R"
            }
        }
    """
    inst.info = _set_fiff_channelinfo_eyetrack(
        inst.info, channel_dict=channel_dict)
    return inst


def _eyetrack_channelinfo_from_chname(ch_name):
    """
    try finding out what specs the eyetrack channel has
    possible units:
    pupil - arbitrary units, mm (diameter)
    gaze position - px, deg, ...
    eye - left, right
    """

    # try improveing detection by spliting the channel name
    seplist = ['-', '_', '/']  # assume only one is present
    sep_present = False
    ch_parts = []
    for sep in seplist:
        if sep in ch_name:
            sep_present = True
            ch_parts = ch_name.lower().split(sep)

    # extract info
    ch_type = (
        'PUPIL' if (
                'pupil' in ch_parts or
                'area' in ch_parts or
                'pupil' in ch_name.lower() or
                'area' in ch_name.lower()) else
        'GAZE' if (
                'gaze' in ch_parts or
                'pos' in ch_parts or
                'y' in ch_parts or
                'x' in ch_parts or
                'gaze' in ch_name.lower() or
                'pos' in ch_name.lower() or
                'y' in ch_name.lower() or
                'x' in ch_name.lower()
        # potentially problematic, can be in "px" (no prob though, as this would be pos anyway)
        ) else
        'unknown')

    ch_unit = (
        'PIX' if (
                'pixel' in ch_name.lower() or
                'pix' in ch_name.lower()) or
                'px' in ch_name.lower() else
        'DEG' if (
                'deg' in ch_name.lower()) else
        'MM' if (
                'mm' in ch_name.lower()) else
        'AU' if (
                'au' in ch_name.lower()) else
        'unknown')

    ch_eye = (
        'L' if (
                'l' in ch_parts or
                'left' in ch_parts or
                'left' in ch_name.lower()) else
        'R' if (
                'r' in ch_parts or
                'right' in ch_parts or
                'right' in ch_name.lower()) else
        'unknown')

    ch_xy = (
        'Y' if ('y' in ch_parts or
                'y' in ch_name.lower()) else
        'X' if ('x' in ch_parts or  # potentially problematic, as x is in px.
                ('x' in ch_name.lower() and not 'px' in ch_name.lower()) or
                ('x' in ch_name.lower() and not 'pix' in ch_name.lower())) else
        'unknown'
    )
    return ch_type, ch_unit, ch_eye, ch_xy


def _set_fiff_channelinfo_eyetrack(info, channel_dict=None):
    """
    set the correct FIFF constants for eyetrack channels
    """
    # set correct channel type and location
    loc = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    for i_ch, ch_name in enumerate(info.ch_names):
        # only do it for eyetrack channels
        if info['chs'][i_ch]['kind'] != FIFF.FIFFV_EYETRACK_CH:
            continue

        # if no channel dict is provided, try finding out what happened
        # possible units:
        # pupil - arbitrary units, mm (diameter)
        # gaze position - px, deg, ...
        # eye - left, right, (binocular average)
        ch_type, ch_unit, ch_eye, ch_xy = _eyetrack_channelinfo_from_chname(ch_name)
        if type(channel_dict) == dict:  # if there is user input, overwrite guesses
            if ch_name not in channel_dict.keys():
                warn("'channel_dict' was provided, but contains no info "
                        "on channel '{}'".format(ch_name) )
            else:
                keys = channel_dict[ch_name].keys()
                ch_type = channel_dict[ch_name]['ch_type'] if 'ch_type' in keys else ch_type
                ch_unit = channel_dict[ch_name]['ch_unit'] if 'ch_unit' in keys else ch_unit
                ch_eye = channel_dict[ch_name]['ch_eye'] if 'ch_eye' in keys else ch_eye
                ch_xy = channel_dict[ch_name]['ch_xy'] if 'ch_xy' in keys else ch_xy

        # set coil type
        if ch_type not in ['PUPIL', 'GAZE']:
            warn("couldn't determine channel type for channel {}."
                    "defaults to 'gaze'.".format(ch_name))
        coil_type = (
            FIFF.FIFFV_COIL_EYETRACK_PUPIL if (ch_type == 'PUPIL') else
            FIFF.FIFFV_COIL_EYETRACK_POS)
        info['chs'][i_ch]['coil_type'] = coil_type

        # set unit
        if ch_unit not in ['PIX', 'AU', 'DEG', 'MM']:
            warn("couldn't determine unit for channel {}."
                    "defaults to 'arbitrary units'.".format(ch_name))
        unit = (FIFF.FIFF_UNIT_PX if (ch_unit == 'PIX') else
                FIFF.FIFF_UNIT_DEG if (ch_unit == 'DEG') else
                FIFF.FIFF_UNIT_MM if (ch_unit == 'MM') else
                FIFF.FIFF_UNIT_NONE)
        info['chs'][i_ch]['unit'] = unit

        # set eye and x/y coordinate
        if ch_eye not in ['L', 'R']:
            warn("couldn't determine eye for channel {}."
                    "defaults to 'NaN'.".format(ch_name))
        loc[3] = (-1 if (ch_eye == 'L') else
                  1 if (ch_eye == 'R') else
                  np.nan)

        if (ch_xy not in ['X', 'Y']) and (ch_type == 'GAZE'):
            warn("couldn't determine X/Y for channel {}."
                    "defaults to 'NaN'.".format(ch_name))
        loc[4] = (-1 if (ch_xy == 'X') else
                  1 if (ch_xy == 'Y') else
                  np.nan)
        info['chs'][i_ch]['loc'] = loc.copy()

    return info
