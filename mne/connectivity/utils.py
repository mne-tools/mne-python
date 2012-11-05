# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)


def check_idx(idx):
    """Check idx parameter"""

    if not isinstance(idx, tuple) or len(idx) != 2:
        raise ValueError('idx must be a tuple of length 2')

    if len(idx[0]) != len(idx[1]):
        raise ValueError('Index arrays idx[0] and idx[1] must have same '
                         'length')

    return idx
