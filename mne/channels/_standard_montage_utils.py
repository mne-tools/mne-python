# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from .montage import read_montage


def read_standard_montage(kind):
    return read_montage(kind)
