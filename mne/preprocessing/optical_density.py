# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from ..io import BaseRaw
from ..utils import _validate_type


def optical_density(raw):
    r"""Convert NIRS raw data to optical density.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance. Operates in place.

    """
    _validate_type(raw, BaseRaw, 'raw')
    return raw
