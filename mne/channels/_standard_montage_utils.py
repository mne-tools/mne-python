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


def _egi_256():
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

    ch_names, _, _, _, xs, ys, zs, _ = np.loadtxt(fname, **options)
    pos = np.stack([xs, ys, zs], axis=-1)

    # Fix pos to match Montage code
    # pos -= np.mean(pos, axis=0)
    pos = HEAD_SIZE_DEFAULT * (pos / np.linalg.norm(pos, axis=1).mean())

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )


def _easycap(basename):
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(
        skiprows=1,
        unpack=True,
        dtype={'names': ('Site', 'Theta', 'Phi'),
               'formats': (object, 'i4', 'i4')},
    )
    ch_names, theta, phi = np.loadtxt(fname, **options)

    radii = np.full_like(phi, 1)  # XXX: HEAD_SIZE_DEFAULT should work
    pos = _sph_to_cart(np.stack(
        [radii, np.deg2rad(phi), np.deg2rad(theta)],
        axis=-1,
    ))

    # scale up to realistic head radius (8.5cm == 85mm):
    pos *= HEAD_SIZE_DEFAULT  # XXXX: this should be done through radii

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )


def _hydrocel(basename, fid_names=('FidNz', 'FidT9', 'FidT10')):
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(
        unpack=True,
        dtype={'names': ('ch_names', 'x', 'y', 'z'),
               'formats': (object, 'f4', 'f4', 'f4')},
    )
    ch_names, xs, ys, zs = np.loadtxt(fname, **options)

    pos = np.stack([xs, ys, zs], axis=-1) * 0.01
    ch_pos = dict(zip(ch_names, pos))
    _ = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='head')


def _biosemi(basename, fid_names=('Nz', 'LPA', 'RPA')):
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(
        skiprows=1,
        unpack=True,
        dtype={'names': ('Site', 'Theta', 'Phi'),
               'formats': (object, 'i4', 'i4')},
    )
    ch_names, theta, phi = np.loadtxt(fname, **options)

    radii = np.full_like(phi, 1)  # XXX: HEAD_SIZE_DEFAULT should work
    pos = _sph_to_cart(np.stack(
        [radii, np.deg2rad(phi), np.deg2rad(theta)],
        axis=-1,
    ))

    # scale up to realistic head radius (8.5cm == 85mm):
    pos *= HEAD_SIZE_DEFAULT  # XXXX: this should be done through radii

    ch_pos = dict(zip(ch_names, pos))
    _ = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='head')


def _mgh_or_standard(basename, fid_names=('Nz', 'LPA', 'RPA')):
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

    ch_pos = dict(zip(ch_names_, pos))
    _ = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='head')


standard_montage_look_up_table = {
    'EGI_256': _egi_256,

    'easycap-M1': partial(_easycap, basename='easycap-M1.txt'),
    'easycap-M10': partial(_easycap, basename='easycap-M1.txt'),

    'GSN-HydroCel-128': partial(_hydrocel, basename='GSN-HydroCel-128.sfp'),
    'GSN-HydroCel-129': partial(_hydrocel, basename='GSN-HydroCel-129.sfp'),
    'GSN-HydroCel-256': partial(_hydrocel, basename='GSN-HydroCel-256.sfp'),
    'GSN-HydroCel-257': partial(_hydrocel, basename='GSN-HydroCel-257.sfp'),
    'GSN-HydroCel-32': partial(_hydrocel, basename='GSN-HydroCel-32.sfp'),
    'GSN-HydroCel-64_1.0': partial(_hydrocel,
                                   basename='GSN-HydroCel-64_1.0.sfp'),
    'GSN-HydroCel-65_1.0': partial(_hydrocel,
                                   basename='GSN-HydroCel-65_1.0.sfp'),

    'biosemi128': partial(_biosemi, basename='biosemi128.txt'),
    'biosemi16': partial(_biosemi, basename='biosemi16.txt'),
    'biosemi160': partial(_biosemi, basename='biosemi160.txt'),
    'biosemi256': partial(_biosemi, basename='biosemi256.txt'),
    'biosemi32': partial(_biosemi, basename='biosemi32.txt'),
    'biosemi64': partial(_biosemi, basename='biosemi64.txt'),

    'mgh60': partial(_mgh_or_standard, basename='mgh60.elc'),
    'mgh70': partial(_mgh_or_standard, basename='mgh70.elc'),
    'standard_1005': partial(_mgh_or_standard,
                             basename='standard_1005.elc'),
    'standard_1020': partial(_mgh_or_standard,
                             basename='standard_1020.elc'),
    'standard_alphabetic': partial(_mgh_or_standard,
                                   basename='standard_alphabetic.elc'),
    'standard_postfixed': partial(_mgh_or_standard,
                                  basename='standard_postfixed.elc'),
    'standard_prefixed': partial(_mgh_or_standard,
                                 basename='standard_prefixed.elc'),
    'standard_primed': partial(_mgh_or_standard,
                               basename='standard_primed.elc'),
}
