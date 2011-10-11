# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .fiff import fiff_open
from .fiff.proj import read_proj as fiff_read_proj


def read_proj(fname):
    """Read projections from a FIF file.

    Parameters
    ----------
    fname: string
        The name of file containing the projections vectors.

    Returns
    -------
    projs: list
        The list of projection vectors.
    """
    fid, tree, _ = fiff_open(fname)
    projs = fiff_read_proj(fid, tree)
    return projs
