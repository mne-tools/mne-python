import os.path as op
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nose.tools import assert_true, assert_raises

from mne.datasets import sample
from mne import read_bem_surfaces, write_bem_surface, read_surface, \
                write_surface, decimate_surface
from mne.utils import _TempDir, requires_tvtk

data_path = sample.data_path()
fname = op.join(data_path, 'subjects', 'sample', 'bem',
                'sample-5120-5120-5120-bem-sol.fif')

tempdir = _TempDir()


def test_io_bem_surfaces():
    """Test reading of bem surfaces
    """
    surf = read_bem_surfaces(fname, add_geom=True)
    surf = read_bem_surfaces(fname, add_geom=False)
    print "Number of surfaces : %d" % len(surf)

    write_bem_surface(op.join(tempdir, 'bem_surf.fif'), surf[0])
    surf_read = read_bem_surfaces(op.join(tempdir, 'bem_surf.fif'),
                                  add_geom=False)

    for key in surf[0].keys():
        assert_array_almost_equal(surf[0][key], surf_read[0][key])


def test_io_surface():
    """Test reading and writing of Freesurfer surface mesh files
    """
    fname = op.join(data_path, 'subjects', 'fsaverage', 'surf', 'lh.inflated')
    pts, tri = read_surface(fname)
    write_surface(op.join(tempdir, 'tmp'), pts, tri)
    c_pts, c_tri = read_surface(op.join(tempdir, 'tmp'))
    assert_array_equal(pts, c_pts)
    assert_array_equal(tri, c_tri)


@requires_tvtk
def test_decimate_surface():
    """Test triangular surface decimation
    """
    points = np.array([[-0.00686118, -0.10369860, 0.02615170],
                       [-0.00713948, -0.10370162, 0.02614874],
                       [-0.00686208, -0.10368247, 0.02588313],
                       [-0.00713987, -0.10368724, 0.02587745]])
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, 0]])
    for n_tri in [4, 3, 2]:  # quadric decimation creates even numberd output.
        _, this_tris = decimate_surface(points, tris, n_tri)
        assert_true(len(this_tris) == n_tri if not n_tri % 2 else 2)
    nirvana = 5
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, nirvana]])
    assert_raises(ValueError, decimate_surface, points, tris, n_tri)
