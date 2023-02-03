# Authors: Dominik Welke <dominik.welke@mailbox.org>
#
# License: BSD-3-Clause


import numpy as np

from ..io.constants import FIFF


# specific function to set eyetrack channels
def set_channel_types_eyetrack(inst, mapping):
    """Define sensor type for eyetrack channels.

    This function can set all specific information.
    Note: inst.set_channel_types() with eyetrack_pos or eyetrack_pupil
    achieves the same, but cannot set specific unit, eye or x/y component
    (for Gaze channel).

    Parameters
    ----------
    inst : Instance of Raw, Epochs, or Evoked
        The data instance.
    mapping : dict
        A dictionary mapping a channel to a list/tuple including
        sensor type, unit, eye, [and x/y component] (all as str),  e.g.,
        ``{'l_x': ('eyetrack_pos', 'deg', 'left', 'x')}``.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        The instance, modified in place.
    """
    ch_names = inst.info['ch_names']

    # allowed
    valid_types = ['eyetrack_pos', 'eyetrack_pupil']  # ch_type
    valid_units = {'px': ['px', 'pixel'],
                   'deg': ['deg', 'degree', 'degrees'],
                   'mm': ['mm', 'diameter'],
                   'au': [None, 'none', 'au', 'area']}
    valid_units['all'] = [item for sublist in valid_units.values()
                          for item in sublist]
    valid_eye = {'l': ['left', 'l'],
                 'r': ['right', 'r']}
    valid_eye['all'] = [item for sublist in valid_eye.values()
                        for item in sublist]
    valid_xy = {'x': ['x', 'h', 'horizontal'],
                'y': ['y', 'v', 'vertical']}
    valid_xy['all'] = [item for sublist in valid_xy.values()
                       for item in sublist]

    # loop over channels
    for ch_name, ch_desc in mapping.items():
        if ch_name not in ch_names:
            raise ValueError("This channel name (%s) doesn't exist in "
                             "info." % ch_name)
        c_ind = ch_names.index(ch_name)

        # set ch_type and unit
        ch_type = ch_desc[0].lower()
        if ch_type not in valid_types:
            raise ValueError(
                "ch_type must be one of {}. "
                "Got '{}' instead.".format(valid_types, ch_type))
        if ch_type == 'eyetrack_pos':
            coil_type = FIFF.FIFFV_COIL_EYETRACK_POS
        elif ch_type == 'eyetrack_pupil':
            coil_type = FIFF.FIFFV_COIL_EYETRACK_PUPIL
        inst.info['chs'][c_ind]['coil_type'] = coil_type
        inst.info['chs'][c_ind]['kind'] = FIFF.FIFFV_EYETRACK_CH

        ch_unit = None if (ch_desc[1] is None) else ch_desc[1].lower()
        if ch_unit not in valid_units['all']:
            raise ValueError(
                "unit must be one of {}. Got '{}' instead.".format(
                    valid_units['all'], ch_unit))
        if ch_unit in valid_units['px']:
            unit_new = FIFF.FIFF_UNIT_PX
        elif ch_unit in valid_units['deg']:
            unit_new = FIFF.FIFF_UNIT_DEG
        elif ch_unit in valid_units['mm']:
            unit_new = FIFF.FIFF_UNIT_MM
        elif ch_unit in valid_units['au']:
            unit_new = FIFF.FIFF_UNIT_NONE
        inst.info['chs'][c_ind]['unit'] = unit_new

        # set eye (and x/y-component)
        loc = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        ch_eye = ch_desc[2].lower()
        if ch_eye not in valid_eye['all']:
            raise ValueError(
                "eye must be one of {}. Got '{}' instead.".format(
                    valid_eye['all'], ch_eye))
        if ch_eye in valid_eye['l']:
            loc[3] = -1
        elif ch_eye in valid_eye['r']:
            loc[3] = 1

        if ch_type == 'eyetrack_pos':
            ch_xy = ch_desc[3].lower()
            if ch_xy not in valid_xy['all']:
                raise ValueError(
                    "x/y must be one of {}. Got '{}' instead.".format(
                        valid_xy['all'], ch_xy))
            if ch_xy in valid_xy['x']:
                loc[4] = -1
            elif ch_xy in valid_xy['y']:
                loc[4] = 1

        inst.info['chs'][c_ind]['loc'] = loc

    return inst
