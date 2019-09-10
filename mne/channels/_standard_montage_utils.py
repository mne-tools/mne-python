# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)
import os.path as op
import numpy as np

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
def get_easycap_M1():
    """ Load easycap M1.

        data = np.genfromtxt(fname, dtype='str', skip_header=1)
        ch_names_ = data[:, 0].tolist()
        az = np.deg2rad(data[:, 2].astype(float))
        pol = np.deg2rad(data[:, 1].astype(float))
        rad = np.ones(len(az))  # spherical head model
        rad *= 85.  # scale up to realistic head radius (8.5cm == 85mm)
        pos = _sph_to_cart(np.array([rad, az, pol]).T)
    """
    fname = op.join(MONTAGE_PATH, 'easycap-M1.txt')
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


def get_easycap_M10():
    """ Load easycap M10.

    """
    fname = op.join(MONTAGE_PATH, 'easycap-M10.txt')
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

def get_hydrocel_128():
    fname = op.join(MONTAGE_PATH, 'GSN-HydroCel-128.sfp')

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


def get_hydrocel_129():
    fname = op.join(MONTAGE_PATH, 'GSN-HydroCel-129.sfp')

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


def get_biosemi128():
    fname = op.join(MONTAGE_PATH, 'biosemi128.txt')
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


def get_mgh60():
    fname = op.join(MONTAGE_PATH, 'mgh60.elc')

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







def read_standard_montage(kind):
    if kind == 'EGI_256':
        dig_montage_A = get_egi_256()

    elif kind == 'easycap_M1':
        dig_montage_A = get_easycap_M1()
    elif kind == 'easycap_M10':
        dig_montage_A = get_easycap_M10()
    elif kind == 'GSN-HydroCel-128':
        dig_montage_A = get_hydrocel_128()
    elif kind == 'GSN-HydroCel-129':
        dig_montage_A = get_hydrocel_129()
    elif kind == 'biosemi128':
        dig_montage_A = get_biosemi128()
    elif kind == 'mgh60':
        dig_montage_A = get_mgh60()
    elif kind == 'standard_1005':
        dig_montage_A = get_standard_1005()

    else:
        montage = read_montage(kind)  # XXX: reader needs to go out!
        dig_montage_A = make_dig_montage(
            ch_pos=dict(zip(montage.ch_names, montage.pos)),
            nasion=montage.nasion,
            lpa=montage.lpa,
            rpa=montage.rpa,
        )
        # dig_montage_B is to create RawArray(.., montage=montage)

    return dig_montage_A
