import os.path as op

from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne.datasets import sample
from mne import read_bem_surfaces, write_bem_surface, read_surface, \
                write_surface
from mne.utils import _TempDir

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
