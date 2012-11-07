# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
import numpy as np


def check_idx(idx):
    """Check idx parameter"""

    if not isinstance(idx, tuple) or len(idx) != 2:
        raise ValueError('idx must be a tuple of length 2')

    if len(idx[0]) != len(idx[1]):
        raise ValueError('Index arrays idx[0] and idx[1] must have same '
                         'length')

    return idx


def idx_seed_con(seeds, dest):
    """Generate idx parameter for seed based connectivity analysis.

    Parameters
    ----------
    seeds : array of int
        Seed indices.
    dest : array of int
        Indices of signals for which to compute connectivity.

    Returns
    -------
    idx : tuple of arrays
        The idx parameter used for connectivity computation.
    """

    n_seeds = len(seeds)
    n_dest = len(dest)

    n_con = n_seeds * n_dest

    idx = (np.concatenate([np.tile(i, n_dest) for i in seeds]),
           np.tile(dest, n_seeds))

    return idx
