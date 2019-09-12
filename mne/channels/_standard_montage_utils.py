# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
import os.path as op
import numpy as np

from functools import partial

from .montage import make_dig_montage
from ..transforms import _sph_to_cart
from . import __file__ as _CHANNELS_INIT_FILE

MONTAGE_PATH = op.join(op.dirname(_CHANNELS_INIT_FILE), 'data', 'montages')

HEAD_SIZE_DEFAULT = 0.085  # in [m]


def _split_eeg_fid(ch_pos, nz_str='Nz', lpa_str='LPA', rpa_str='RPA'):
    nasion = ch_pos.pop(nz_str).reshape(3, ) if nz_str in ch_pos else None
    lpa = ch_pos.pop(lpa_str).reshape(3, ) if lpa_str in ch_pos else None
    rpa = ch_pos.pop(rpa_str).reshape(3, ) if rpa_str in ch_pos else None

    return ch_pos, nasion, lpa, rpa


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
    pos = HEAD_SIZE_DEFAULT * (pos / np.linalg.norm(pos, axis=1).mean())

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )


def get_easycap(basename):
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(
        skiprows=1,
        unpack=True,
        dtype={'names': ('Site', 'Theta', 'Phi'),
               'formats': (object, 'i4', 'i4')},
    )
    ch_names, theta, phi = np.loadtxt(fname, **options)
    radii = np.full_like(phi, 1.)
    pos = _sph_to_cart(np.stack([radii, phi, theta], axis=-1,))

    # scale up to realistic head radius (8.5cm == 85mm):
    pos *= HEAD_SIZE_DEFAULT

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )


def get_hydrocel(basename):
    fname = op.join(MONTAGE_PATH, basename)

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

    ch_pos, nasion, lpa, rpa = _split_eeg_fid(
        ch_pos=dict(zip(ch_names_, pos)),
        nz_str='FidNz', lpa_str='FidT9', rpa_str='FidT10'
    )

    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown',
    )


def get_biosemi(basename):
    fname = op.join(MONTAGE_PATH, basename)
    data = np.genfromtxt(fname, dtype='str', skip_header=1)
    ch_names_ = data[:, 0].tolist()
    az = np.deg2rad(data[:, 2].astype(float))
    pol = np.deg2rad(data[:, 1].astype(float))
    rad = np.ones(len(az))  # spherical head model of radius 1
    pos = _sph_to_cart(np.array([rad, az, pol]).T)

    # scale up to realistic head radius (8.5cm == 85mm):
    pos *= HEAD_SIZE_DEFAULT

    ch_pos, nasion, lpa, rpa = _split_eeg_fid(
        ch_pos=dict(zip(ch_names_, pos)),
        nz_str='Nz', lpa_str='LPA', rpa_str='RPA'
    )

    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown'
    )


def get_mgh_or_standard(basename):
    fname = op.join(MONTAGE_PATH, basename)

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

    ch_pos, nasion, lpa, rpa = _split_eeg_fid(
        ch_pos=dict(zip(ch_names_, pos)),
        nz_str='Nz', lpa_str='LPA', rpa_str='RPA'
    )
    return make_dig_montage(
        ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='unknown',
    )


standard_montage_look_up_table = {
    'EGI_256': get_egi_256,

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
