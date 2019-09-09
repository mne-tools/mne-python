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

from . import __file__ as _CHANNELS_INIT_FILE

MONTAGE_PATH = op.join(op.dirname(_CHANNELS_INIT_FILE), 'data', 'montages')


def get_egi_256():
    _SCALE = 1e-2
    fname = op.join(MONTAGE_PATH, 'EGI_256.csd')
    options = dict(
        comments='//',
        unpack=True,
        dtype={
            'names': ('Label', 'Theta', 'Phi', 'Radius', 'X', 'Y', 'Z',
                      'off sphere surface'),
            'formats': ('S10', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')
        }
    )

    labels, _, _, _, x, y, z, _ = np.loadtxt(fname, **options)
    ch_names = [str(l, encoding='utf-8') for l in labels]
    pos = np.stack([x, y, z], axis=-1) * _SCALE

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='unknown',
    )


def read_standard_montage(kind):
    if kind == 'EGI_256':
        dig_montage_A = get_egi_256()

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


