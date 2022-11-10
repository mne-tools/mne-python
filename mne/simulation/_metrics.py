# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD-3-Clause

import numpy as np


def _check_stc(stc1, stc2):
    """Check that stcs are compatible."""
    if stc1.data.shape != stc2.data.shape:
        raise ValueError('Data in stcs must have the same size')
    if np.all(stc1.times != stc2.times):
        raise ValueError('Times of two stcs must match.')
