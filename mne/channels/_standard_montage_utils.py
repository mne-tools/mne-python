# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)
import os.path as op
import numpy as np

from functools import partial

from .montage import read_montage
from .montage import make_dig_montage
from .montage import DigMontage

from .._digitization import Digitization

from ..transforms import (apply_trans, get_ras_to_neuromag_trans, _sph_to_cart,
                          _topo_to_sph, _frame_to_str, _str_to_frame,
                          Transform)

from . import __file__ as _CHANNELS_INIT_FILE

MONTAGE_PATH = op.join(op.dirname(_CHANNELS_INIT_FILE), 'data', 'montages')


def get_egi_256():
    fname = op.join(MONTAGE_PATH, 'EGI_256.csd')
    options = dict(
        comments='//',
        unpack=True,
        dtype={
            'names': ('Label', 'Theta', 'Phi', 'Radius', 'X', 'Y', 'Z',
                      'off sphere surface'),
            'formats': (object, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')
        }
    )

    ch_names, _, _, _, x, y, z, _ = np.loadtxt(fname, **options)
    pos = np.stack([x, y, z], axis=-1)

    # Fix pos to match Montage code
    # pos -= np.mean(pos, axis=0)
    pos = 0.085 * (pos / np.linalg.norm(pos, axis=1).mean())

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )


# HEAD_SIZE_ESTIMATION = 0.085  # in [m]

# HEAD_SIZE_ESTIMATION = 0.085  # in [m]
HEAD_SIZE_ESTIMATION = 1  # in [m]


def get_easycap(basename):
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(
        skiprows=1,
        unpack=True,
        dtype={'names': ('Site', 'Theta', 'Phi'),
               'formats': (object, 'i4', 'i4')},
    )
    ch_names, theta, phi = np.loadtxt(fname, **options)
    radious = np.full_like(phi, HEAD_SIZE_ESTIMATION)
    pos = _sph_to_cart(np.stack([radious, phi, theta], axis=-1,))

    pos *= 0.085  # XXX this should work out of the box with HEAD_SIZE

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )


def get_hydrocel(basename):
    fname = op.join(MONTAGE_PATH, basename)

    LPA_CH_NAME = 'FidT9'
    NASION_CH_NAME = 'FidNz'
    RPA_CH_NAME = 'FidT10'
    with open(fname, 'r') as f:
        lines = f.read().replace('\t', ' ').splitlines()

    ch_names_, pos = [], []
    for ii, line in enumerate(lines):
        line = line.strip().split()
        if len(line) > 0:  # skip empty lines
            if len(line) != 4:  # name, x, y, z
                raise ValueError("Malformed .sfp file in line " + str(ii))
            this_name, x, y, z = line
            ch_names_.append(this_name)
            pos.append([float(cord) for cord in (x, y, z)])
    pos = np.asarray(pos)

    ch_pos = dict(zip(ch_names_, pos))
    nasion = ch_pos.pop(NASION_CH_NAME).reshape(3, )
    lpa = ch_pos.pop(LPA_CH_NAME).reshape(3, )
    rpa = ch_pos.pop(RPA_CH_NAME).reshape(3, )

    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown',
    )


def get_biosemi(basename):
    fname = op.join(MONTAGE_PATH, basename)
    data = np.genfromtxt(fname, dtype='str', skip_header=1)
    ch_names_ = data[:, 0].tolist()
    az = np.deg2rad(data[:, 2].astype(float))
    pol = np.deg2rad(data[:, 1].astype(float))
    rad = np.ones(len(az))  # spherical head model
    rad *= 85.  # scale up to realistic head radius (8.5cm == 85mm)
    pos = _sph_to_cart(np.array([rad, az, pol]).T)

    ch_pos = dict(zip(ch_names_, pos))
    nasion = ch_pos.pop('Nz').reshape(3, )
    lpa = ch_pos.pop('LPA').reshape(3, )
    rpa = ch_pos.pop('RPA').reshape(3, )
    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown',
    )


def get_mgh_or_standard(basename):
    fname = op.join(MONTAGE_PATH, basename)

    # 10-5 system
    ch_names_, pos = [], []
    with open(fname) as fid:
        # Default units are meters
        for line in fid:
            if 'UnitPosition' in line:
                units = line.split()[1]
                scale_factor = dict(m=1., mm=1e-3)[units]
                break
        else:
            raise RuntimeError('Could not detect units in file %s' % fname)
        for line in fid:
            if 'Positions\n' in line:
                break
        pos = []
        for line in fid:
            if 'Labels\n' in line:
                break
            pos.append(list(map(float, line.split())))
        for line in fid:
            if not line or not set(line) - {' '}:
                break
            ch_names_.append(line.strip(' ').strip('\n'))
    pos = np.array(pos) * scale_factor

    ch_pos = dict(zip(ch_names_, pos))
    nasion = ch_pos.pop('Nz').reshape(3, )
    lpa = ch_pos.pop('LPA').reshape(3, )
    rpa = ch_pos.pop('RPA').reshape(3, )
    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown',
    )


def get_standard_1005():
    fname = op.join(MONTAGE_PATH, 'standard_1005.elc')

    ch_names_, pos = [], []
    with open(fname) as fid:
        # Default units are meters
        for line in fid:
            if 'UnitPosition' in line:
                units = line.split()[1]
                scale_factor = dict(m=1., mm=1e-3)[units]
                break
        else:
            raise RuntimeError('Could not detect units in file %s' % fname)
        for line in fid:
            if 'Positions\n' in line:
                break
        pos = []
        for line in fid:
            if 'Labels\n' in line:
                break
            pos.append(list(map(float, line.split())))
        for line in fid:
            if not line or not set(line) - {' '}:
                break
            ch_names_.append(line.strip(' ').strip('\n'))
    pos = np.array(pos) * scale_factor

    ch_pos = dict(zip(ch_names_, pos))
    nasion = ch_pos.pop('Nz').reshape(3, )
    lpa = ch_pos.pop('LPA').reshape(3, )
    rpa = ch_pos.pop('RPA').reshape(3, )
    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown',
    )

standard_montage_look_up_table = {
    'easycap-M1': partial(get_easycap, basename='easycap-M1.txt'),
    'easycap-M10': partial(get_easycap, basename='easycap-M1.txt'),

    'GSN-HydroCel-128': partial(get_hydrocel, basename='GSN-HydroCel-128.sfp'),
    'GSN-HydroCel-129': partial(get_hydrocel, basename='GSN-HydroCel-129.sfp'),
    'GSN-HydroCel-256': partial(get_hydrocel, basename='GSN-HydroCel-256.sfp'),
    'GSN-HydroCel-257': partial(get_hydrocel, basename='GSN-HydroCel-257.sfp'),
    'GSN-HydroCel-32': partial(get_hydrocel, basename='GSN-HydroCel-32.sfp'),
    'GSN-HydroCel-64_1.0': partial(get_hydrocel,
                                   basename='GSN-HydroCel-64_1.0.sfp'),
    'GSN-HydroCel-65_1.0': partial(get_hydrocel,
                                   basename='GSN-HydroCel-65_1.0.sfp'),

    'biosemi128': partial(get_biosemi, basename='biosemi128.txt'),
    'biosemi16': partial(get_biosemi, basename='biosemi16.txt'),
    'biosemi160': partial(get_biosemi, basename='biosemi160.txt'),
    'biosemi256': partial(get_biosemi, basename='biosemi256.txt'),
    'biosemi32': partial(get_biosemi, basename='biosemi32.txt'),
    'biosemi64': partial(get_biosemi, basename='biosemi64.txt'),

    'mgh60': partial(get_mgh_or_standard, basename='mgh60.elc'),
    'mgh70': partial(get_mgh_or_standard, basename='mgh70.elc'),
    'standard_1005': partial(get_mgh_or_standard,
                             basename='standard_1005.elc'),
    'standard_1020': partial(get_mgh_or_standard,
                             basename='standard_1020.elc'),
    'standard_alphabetic': partial(get_mgh_or_standard,
                                   basename='standard_alphabetic.elc'),
    'standard_postfixed': partial(get_mgh_or_standard,
                                  basename='standard_postfixed.elc'),
    'standard_prefixed': partial(get_mgh_or_standard,
                                 basename='standard_prefixed.elc'),
    'standard_primed': partial(get_mgh_or_standard,
                               basename='standard_primed.elc'),
}


def read_standard_montage(kind):
    """Read a generic (built-in) montage.

    Individualized (digitized) electrode positions should be read in using
    :func:`read_dig_montage`.  # XXXX

    Parameters
    ----------
    kind : str
        The name of the montage file without the file extension (e.g.
        kind='easycap-M10' for 'easycap-M10.txt'). See notes for valid kinds.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage

    Notes
    -----
    Valid ``kind`` arguments are:

    ===================   =====================================================
    Kind                  Description
    ===================   =====================================================
    standard_1005         Electrodes are named and positioned according to the
                          international 10-05 system (343+3 locations)
    standard_1020         Electrodes are named and positioned according to the
                          international 10-20 system (94+3 locations)
    standard_alphabetic   Electrodes are named with LETTER-NUMBER combinations
                          (A1, B2, F4, ...) (65+3 locations)
    standard_postfixed    Electrodes are named according to the international
                          10-20 system using postfixes for intermediate
                          positions (100+3 locations)
    standard_prefixed     Electrodes are named according to the international
                          10-20 system using prefixes for intermediate
                          positions (74+3 locations)
    standard_primed       Electrodes are named according to the international
                          10-20 system using prime marks (' and '') for
                          intermediate positions (100+3 locations)

    biosemi16             BioSemi cap with 16 electrodes (16+3 locations)
    biosemi32             BioSemi cap with 32 electrodes (32+3 locations)
    biosemi64             BioSemi cap with 64 electrodes (64+3 locations)
    biosemi128            BioSemi cap with 128 electrodes (128+3 locations)
    biosemi160            BioSemi cap with 160 electrodes (160+3 locations)
    biosemi256            BioSemi cap with 256 electrodes (256+3 locations)

    easycap-M1            EasyCap with 10-05 electrode names (74 locations)
    easycap-M10           EasyCap with numbered electrodes (61 locations)

    EGI_256               Geodesic Sensor Net (256 locations)

    GSN-HydroCel-32       HydroCel Geodesic Sensor Net and Cz (33+3 locations)
    GSN-HydroCel-64_1.0   HydroCel Geodesic Sensor Net (64+3 locations)
    GSN-HydroCel-65_1.0   HydroCel Geodesic Sensor Net and Cz (65+3 locations)
    GSN-HydroCel-128      HydroCel Geodesic Sensor Net (128+3 locations)
    GSN-HydroCel-129      HydroCel Geodesic Sensor Net and Cz (129+3 locations)
    GSN-HydroCel-256      HydroCel Geodesic Sensor Net (256+3 locations)
    GSN-HydroCel-257      HydroCel Geodesic Sensor Net and Cz (257+3 locations)

    mgh60                 The (older) 60-channel cap used at
                          MGH (60+3 locations)
    mgh70                 The (newer) 70-channel BrainVision cap used at
                          MGH (70+3 locations)
    ===================   =====================================================

    .. versionadded:: 0.9.0
    """
    if kind not in standard_montage_look_up_table:
        # XXX: this is the old message needs update
        raise ValueError('Could not find the montage. Please provide the '
                         'full path.')
    else:
        return standard_montage_look_up_table[kind]()
