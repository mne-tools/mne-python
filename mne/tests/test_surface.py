import os.path as op

# from numpy.testing import assert_array_almost_equal

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'subjects', 'sample', 'bem',
                                        'sample-5120-5120-5120-bem-sol.fif')


def test_io_bem_surfaces():
    """Testing reading of bem surfaces
    """
    surf = mne.read_bem_surfaces(fname, add_geom=False)
    surf = mne.read_bem_surfaces(fname, add_geom=True)
    print "Number of surfaces : %d" % len(surf)
