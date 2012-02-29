import os.path as op

from numpy.testing import assert_array_almost_equal

from ..datasets import sample
from .. import read_bem_surfaces, write_bem_surface

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'subjects', 'sample', 'bem',
                                        'sample-5120-5120-5120-bem-sol.fif')


def test_io_bem_surfaces():
    """Testing reading of bem surfaces
    """
    surf = read_bem_surfaces(fname, add_geom=True)
    surf = read_bem_surfaces(fname, add_geom=False)
    print "Number of surfaces : %d" % len(surf)

    write_bem_surface('bem_surf.fif', surf[0])
    surf_read = read_bem_surfaces('bem_surf.fif', add_geom=False)

    for key in surf[0].keys():
        assert_array_almost_equal(surf[0][key], surf_read[0][key])
