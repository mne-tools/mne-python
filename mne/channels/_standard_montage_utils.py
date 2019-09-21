# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
from collections import OrderedDict
import os.path as op
import numpy as np

from functools import partial

from .montage import make_dig_montage
from ..transforms import _sph_to_cart
from . import __file__ as _CHANNELS_INIT_FILE

MONTAGE_PATH = op.join(op.dirname(_CHANNELS_INIT_FILE), 'data', 'montages')

_str = 'U100'


# In standard_1020, T9=LPA, T10=RPA, Nasion is the same as Iz with a
# sign-flipped Y value

def _egi_256(head_size):
    fname = op.join(MONTAGE_PATH, 'EGI_256.csd')
    # Label, Theta, Phi, Radius, X, Y, Z, off sphere surface
    options = dict(comments='//',
                   dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    ch_names, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname, **options)
    pos = np.stack([xs, ys, zs], axis=-1)

    # Fix pos to match Montage code
    pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

    # For this cap, the Nasion is the frontmost electrode,
    # LPA/RPA we approximate by putting 75% of the way (toward the front)
    # between the two electrodes that are halfway down the ear holes
    nasion = pos[ch_names.index('E31')]
    lpa = (0.75 * pos[ch_names.index('E67')] +
           0.25 * pos[ch_names.index('E94')])
    rpa = (0.75 * pos[ch_names.index('E219')] +
           0.25 * pos[ch_names.index('E190')])

    return make_dig_montage(
        ch_pos=OrderedDict(zip(ch_names, pos)),
        coord_frame='unknown', nasion=nasion, lpa=lpa, rpa=rpa,
    )


def _easycap(basename, head_size):
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(skip_header=1, dtype=(_str, 'i4', 'i4'))
    ch_names, theta, phi = _safe_np_loadtxt(fname, **options)

    radii = np.full(len(phi), head_size)
    pos = _sph_to_cart(np.stack(
        [radii, np.deg2rad(phi), np.deg2rad(theta)],
        axis=-1,
    ))
    nasion = np.concatenate([[0],  pos[ch_names.index('Fpz'), 1:]])
    nasion *= head_size / np.linalg.norm(nasion)
    lpa = np.mean([pos[ch_names.index('FT9')],
                   pos[ch_names.index('TP9')]], axis=0)
    lpa *= head_size / np.linalg.norm(lpa)  # on sphere
    rpa = np.mean([pos[ch_names.index('FT10')],
                   pos[ch_names.index('TP10')]], axis=0)
    rpa *= head_size / np.linalg.norm(rpa)

    return make_dig_montage(
        ch_pos=OrderedDict(zip(ch_names, pos)),
        coord_frame='unknown', nasion=nasion, lpa=lpa, rpa=rpa,
    )


def _hydrocel(basename, head_size):
    fid_names = ('FidNz', 'FidT9', 'FidT10')
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(dtype=(_str, 'f4', 'f4', 'f4'))
    ch_names, xs, ys, zs = _safe_np_loadtxt(fname, **options)

    pos = np.stack([xs, ys, zs], axis=-1)
    ch_pos = OrderedDict(zip(ch_names, pos))
    nasion, lpa, rpa = [ch_pos.pop(n) for n in fid_names]
    scale = head_size / np.median(np.linalg.norm(pos, axis=-1))
    for value in ch_pos.values():
        value *= scale
    nasion *= scale
    lpa *= scale
    rpa *= scale

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, rpa=rpa, lpa=lpa)


def _str_names(ch_names):
    return [str(ch_name) for ch_name in ch_names]


def _safe_np_loadtxt(fname, **kwargs):
    out = np.genfromtxt(fname, **kwargs)
    ch_names = _str_names(out['f0'])
    others = tuple(out['f%d' % ii] for ii in range(1, len(out.dtype.fields)))
    return (ch_names,) + others


def _biosemi(basename, head_size):
    fid_names = ('Nz', 'LPA', 'RPA')
    fname = op.join(MONTAGE_PATH, basename)
    options = dict(skip_header=1, dtype=(_str, 'i4', 'i4'))
    ch_names, theta, phi = _safe_np_loadtxt(fname, **options)

    radii = np.full(len(phi), head_size)
    pos = _sph_to_cart(np.stack(
        [radii, np.deg2rad(phi), np.deg2rad(theta)],
        axis=-1,
    ))

    ch_pos = OrderedDict(zip(ch_names, pos))
    nasion, lpa, rpa = [ch_pos.pop(n) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, lpa=lpa, rpa=rpa)


def _mgh_or_standard(basename, head_size):
    fid_names = ('Nz', 'LPA', 'RPA')
    fname = op.join(MONTAGE_PATH, basename)

    ch_names_, pos = [], []
    with open(fname) as fid:
        # Ignore units as we will scale later using the norms anyway
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

    pos = np.array(pos)
    ch_pos = OrderedDict(zip(ch_names_, pos))
    nasion, lpa, rpa = [ch_pos.pop(n) for n in fid_names]
    scale = head_size / np.median(np.linalg.norm(pos, axis=1))
    for value in ch_pos.values():
        value *= scale
    nasion *= scale
    lpa *= scale
    rpa *= scale

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, lpa=lpa, rpa=rpa)


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
