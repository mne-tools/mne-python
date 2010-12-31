import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal

import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'subjects', 'sample', 'bem',
                                            'sample-5120-bem-sol.fif')

def test_io_bem_surfaces():
    """Testing reading of bem surfaces
    """
    surf = fiff.read_bem_surfaces(fname, add_geom=False)
    surf = fiff.read_bem_surfaces(fname, add_geom=True)
    print "Number of surfaces : %d" % len(surf)
