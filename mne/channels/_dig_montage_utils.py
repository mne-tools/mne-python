# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

import xml.etree.ElementTree as ElementTree

import numpy as np

from ..transforms import apply_trans, get_ras_to_neuromag_trans

from ..io.constants import FIFF
from ..utils import _check_fname, Bunch, warn


def _fix_data_fiducials(data):
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

        data.nasion, data.lpa, data.rpa = [
            data.elp[names.index(kind)] for kind in kinds
        ]

        # remove fiducials from elp
        mask = np.ones(len(names), dtype=bool)
        for fid in ['nasion', 'lpa', 'rpa']:
            mask[names.index(fid)] = False
        data.elp = data.elp[mask]
        data.point_names = [p for pi, p in enumerate(data.point_names)
                            if mask[pi]]
    return data


def _transform_to_head_call(data):
    """Transform digitizer points to Neuromag head coordinates.

    Parameters
    ----------
    data : Bunch.
        replicates DigMontage old structure. Requires the following fields:
        ['nasion', 'lpa', 'rpa', 'hsp', 'hpi', 'elp', 'coord_frame',
         'dig_ch_pos']

    Returns
    -------
    data : Bunch.
        transformed version of input data.
    """
    if data.coord_frame == 'head':  # nothing to do
        return data
    nasion, rpa, lpa = data.nasion, data.rpa, data.lpa

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


_cardinal_ident_mapping = {
    FIFF.FIFFV_POINT_NASION: 'nasion',
    FIFF.FIFFV_POINT_LPA: 'lpa',
    FIFF.FIFFV_POINT_RPA: 'rpa',
}


def _read_dig_montage_egi(
        fname,
        _scaling,
        _all_data_kwargs_are_none,
):

    if not _all_data_kwargs_are_none:
        raise ValueError('hsp, hpi, elp, point_names, fif must all be '
                         'None if egi is not None')
    _check_fname(fname, overwrite='read', must_exist=True)

    root = ElementTree.parse(fname).getroot()
    ns = root.tag[root.tag.index('{'):root.tag.index('}') + 1]
    sensors = root.find('%ssensorLayout/%ssensors' % (ns, ns))
    fids = dict()
    dig_ch_pos = dict()

    fid_name_map = {'Nasion': 'nasion',
                    'Right periauricular point': 'rpa',
                    'Left periauricular point': 'lpa'}

    for s in sensors:
        name, number, kind = s[0].text, int(s[1].text), int(s[2].text)
        coordinates = np.array([float(s[3].text), float(s[4].text),
                                float(s[5].text)])

        coordinates *= _scaling

        # EEG Channels
        if kind == 0:
            dig_ch_pos['EEG %03d' % number] = coordinates
        # Reference
        elif kind == 1:
            dig_ch_pos['EEG %03d' %
                       (len(dig_ch_pos.keys()) + 1)] = coordinates
            # XXX: we should do something with this (ref and eeg get mixed)

        # Fiducials
        elif kind == 2:
            fid_name = fid_name_map[name]
            fids[fid_name] = coordinates
        # Unknown
        else:
            warn('Unknown sensor type %s detected. Skipping sensor...'
                 'Proceed with caution!' % kind)

    return Bunch(
        # EGI stuff
        nasion=fids['nasion'], lpa=fids['lpa'], rpa=fids['rpa'],
        dig_ch_pos=dig_ch_pos, coord_frame='unknown',
        # not EGI stuff
        hsp=None, hpi=None, elp=None, point_names=None,
    )


def _foo_get_data_from_dig(dig):
    # XXXX:
    # This does something really similar to _read_dig_montage_fif but:
    #   - does not check coord_frame
    #   - does not do any operation that implies assumptions with the names

    # Split up the dig points by category
    hsp, hpi, elp = list(), list(), list()
    fids, dig_ch_pos_location = dict(), list()

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            fids[_cardinal_ident_mapping[d['ident']]] = d['r']
        elif d['kind'] == FIFF.FIFFV_POINT_HPI:
            hpi.append(d['r'])
            elp.append(d['r'])
            # XXX: point_names.append('HPI%03d' % d['ident'])
        elif d['kind'] == FIFF.FIFFV_POINT_EXTRA:
            hsp.append(d['r'])
        elif d['kind'] == FIFF.FIFFV_POINT_EEG:
            # XXX: dig_ch_pos['EEG%03d' % d['ident']] = d['r']
            dig_ch_pos_location.append(d['r'])

    dig_coord_frames = set([d['coord_frame'] for d in dig])
    assert len(dig_coord_frames) == 1, 'Only single coordinate frame in dig is supported' # noqa # XXX

    return Bunch(
        nasion=fids.get('nasion', None),
        lpa=fids.get('lpa', None),
        rpa=fids.get('rpa', None),
        hsp=np.array(hsp) if len(hsp) else None,
        hpi=np.array(hpi) if len(hpi) else None,
        elp=np.array(elp) if len(elp) else None,
        dig_ch_pos_location=dig_ch_pos_location,
        coord_frame=dig_coord_frames.pop(),
    )


def _get_fid_coords(dig):
    fid_coords = dict()
    fid_coord_frames = []

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            fid_coords[_cardinal_ident_mapping[d['ident']]] = d['r']
            fid_coord_frames.append(d['coord_frame'])

    fid_coord_frames = set(fid_coord_frames)

    if len(fid_coord_frames) != 1:
        raise ValueError(
            'All fiducial points must be in the same coordinate system '
            '(got %s)' % len(fid_coord_frames)
        )

    coord_frame = fid_coord_frames.pop()

    if set(fid_coords.keys()) != set(_cardinal_ident_mapping.values()):
        raise ValueError("Some fiducial points are missing (got %s)." %
                         fid_coords.keys())

    return fid_coords, coord_frame


def _read_dig_montage_bvct(
        fname,
        unit,
        _all_data_kwargs_are_none,
):

    if not _all_data_kwargs_are_none:
        raise ValueError('hsp, hpi, elp, point_names, fif must all be '
                         'None if egi is not None')
    _check_fname(fname, overwrite='read', must_exist=True)

    # CapTrak is natively in mm
    scale = dict(mm=1e-3, cm=1e-2, auto=1e-3, m=1)
    if unit not in scale:
        raise ValueError("Unit needs to be one of %s, not %r" %
                         (sorted(scale.keys()), unit))
    if unit not in ['mm', 'auto']:
        warn('Using "{}" as unit for BVCT file. BVCT files are usually '
             'specified in "mm". This might lead to errors.'.format(unit),
             RuntimeWarning)

    return _parse_brainvision_dig_montage(fname, scale=scale[unit])


BACK_COMPAT = object()  # XXX: to remove in 0.20


def _parse_brainvision_dig_montage(fname, scale=BACK_COMPAT):
    BVCT_SCALE = 1e-3
    FID_NAME_MAP = {'Nasion': 'nasion', 'RPA': 'rpa', 'LPA': 'lpa'}

    root = ElementTree.parse(fname).getroot()
    sensors = root.find('CapTrakElectrodeList')

    fids, dig_ch_pos = dict(), dict()

    for s in sensors:
        name = s.find('Name').text

        # Need to prune "GND" and "REF": these are not included in the raw
        # data and will raise errors when we try to do raw.set_montage(...)
        # XXX eventually this should be stored in ch['loc'][3:6]
        # but we don't currently have such capabilities here
        if name in ['GND', 'REF']:
            continue

        is_fid = name in FID_NAME_MAP
        coordinates = np.array([float(s.find('X').text),
                                float(s.find('Y').text),
                                float(s.find('Z').text)])

        coordinates *= BVCT_SCALE if scale is BACK_COMPAT else scale

        # Fiducials
        if is_fid:
            fids[FID_NAME_MAP[name]] = coordinates
        # EEG Channels
        else:
            dig_ch_pos[name] = coordinates

    return Bunch(
        # BVCT stuff
        nasion=fids['nasion'], lpa=fids['lpa'], rpa=fids['rpa'],
        dig_ch_pos=dig_ch_pos, coord_frame='unknown',
        # not BVCT stuff
        hsp=None, hpi=None, elp=None, point_names=None,
    )
