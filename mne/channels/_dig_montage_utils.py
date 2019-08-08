import numpy as np

from ..transforms import apply_trans, get_ras_to_neuromag_trans


def _transform_to_head_call(data):
    """Transform digitizer points to Neuromag head coordinates.

    Parameters
    ----------
    data : Bunch.
        replicates DigMontage structure
    """
    if data.coord_frame == 'head':  # nothing to do
        return data
    nasion, rpa, lpa = data.nasion, data.rpa, data.lpa
    if any(x is None for x in (nasion, rpa, lpa)):
        if data.elp is None or data.point_names is None:
            raise ValueError('ELP points and names must be specified for '
                             'transformation.')
        names = [name.lower() for name in data.point_names]

        # check that all needed points are present
        kinds = ('nasion', 'lpa', 'rpa')
        missing = [name for name in kinds if name not in names]
        if len(missing) > 0:
            raise ValueError('The points %s are missing, but are needed '
                             'to transform the points to the MNE '
                             'coordinate system. Either add the points, '
                             'or read the montage with transform=False.'
                             % str(missing))

        nasion, lpa, rpa = [data.elp[names.index(kind)] for kind in kinds]

        # remove fiducials from elp
        mask = np.ones(len(names), dtype=bool)
        for fid in ['nasion', 'lpa', 'rpa']:
            mask[names.index(fid)] = False
        data.elp = data.elp[mask]
        data.point_names = [p for pi, p in enumerate(data.point_names)
                            if mask[pi]]

    native_head_t = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    data.nasion, data.lpa, data.rpa = apply_trans(
        native_head_t, np.array([nasion, lpa, rpa]))
    if data.elp is not None:
        data.elp = apply_trans(native_head_t, data.elp)
    if data.hsp is not None:
        data.hsp = apply_trans(native_head_t, data.hsp)
    if data.dig_ch_pos is not None:
        for key, val in data.dig_ch_pos.items():
            data.dig_ch_pos[key] = apply_trans(native_head_t, val)
    data.coord_frame = 'head'

    return data
