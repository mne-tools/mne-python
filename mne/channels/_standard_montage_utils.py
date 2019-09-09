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
    pos -= np.mean(pos, axis=0)
    pos = 0.085 * (pos / np.linalg.norm(pos, axis=1).mean())

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='unknown',
    )


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
    pos = _sph_to_cart(np.array([np.ones_like(phi), phi, theta]).T)

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='unknown',
    )


def read_standard_montage(kind):
    if kind == 'EGI_256':
        dig_montage_A = get_egi_256()

    elif kind == 'easycap_M1':
        dig_montage_A = get_easycap_M1()
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


