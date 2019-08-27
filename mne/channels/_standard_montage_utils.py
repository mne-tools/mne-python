# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from .montage import read_montage
from .montage import make_dig_montage
from .montage import DigMontage

from .._digitization import Digitization


def read_standard_montage(kind):
    montage = read_montage(kind)  # XXX: reader needs to go out!
    dig_montage_A = make_dig_montage(
        ch_pos=dict(zip(montage.ch_names, montage.pos)),
        nasion=montage.nasion,
        lpa=montage.lpa,
        rpa=montage.rpa,
    )
    # dig_montage_B is to create RawArray(.., montage=montage)

    return dig_montage_A
