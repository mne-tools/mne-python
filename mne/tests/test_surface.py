from __future__ import print_function
import os.path as op
import numpy as np
from nose.tools import assert_true, assert_raises
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose, assert_equal)

from mne.datasets import sample
from mne import (read_bem_surfaces, write_bem_surface, read_surface,
                 write_surface, decimate_surface)
from mne.surface import (_make_morph_map, read_morph_map, _compute_nearest,
                         fast_cross_3d, get_head_surf,
                         get_meg_helmet_surf)
from mne.utils import _TempDir, requires_tvtk
from mne.io import read_info
from mne.transforms import _get_mri_head_t_from_trans_file

data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname = op.join(subjects_dir, 'sample', 'bem',
                'sample-5120-5120-5120-bem-sol.fif')
tempdir = _TempDir()


def test_helmet():
    """Test loading helmet surfaces
    """
    base_dir = op.join(op.dirname(__file__), '..', 'io')
    fname_raw = op.join(base_dir, 'tests', 'data', 'test_raw.fif')
    fname_kit_raw = op.join(base_dir, 'kit', 'tests', 'data',
                            'test_bin_raw.fif')
    fname_bti_raw = op.join(base_dir, 'bti', 'tests', 'data',
                            'exported4D_linux_raw.fif')
    fname_ctf_raw = op.join(base_dir, 'tests', 'data', 'test_ctf_raw.fif')
    fname_trans = op.join(base_dir, 'tests', 'data',
                          'sample-audvis-raw-trans.txt')
    trans = _get_mri_head_t_from_trans_file(fname_trans)
    for fname in [fname_raw, fname_kit_raw, fname_bti_raw, fname_ctf_raw]:
        helmet = get_meg_helmet_surf(read_info(fname), trans)
        assert_equal(len(helmet['rr']), 304)  # they all have 304 verts
        assert_equal(len(helmet['rr']), len(helmet['nn']))


@sample.requires_sample_data
def test_head():
    """Test loading the head surface
    """
    surf_1 = get_head_surf('sample', subjects_dir=subjects_dir)
    surf_2 = get_head_surf('sample', 'head', subjects_dir=subjects_dir)
    assert_true(len(surf_1['rr']) < len(surf_2['rr']))  # BEM vs dense head


def test_huge_cross():
    """Test cross product with lots of elements
    """
    x = np.random.rand(100000, 3)
    y = np.random.rand(1, 3)
    z = np.cross(x, y)
    zz = fast_cross_3d(x, y)
    assert_array_equal(z, zz)


def test_compute_nearest():
    """Test nearest neighbor searches"""
    x = np.random.randn(500, 3)
    x /= np.sqrt(np.sum(x ** 2, axis=1))[:, None]
    nn_true = np.random.permutation(np.arange(500, dtype=np.int))[:20]
    y = x[nn_true]

    nn1 = _compute_nearest(x, y, use_balltree=False)
    nn2 = _compute_nearest(x, y, use_balltree=True)
    assert_array_equal(nn_true, nn1)
    assert_array_equal(nn_true, nn2)

    # test distance support
    nnn1 = _compute_nearest(x, y, use_balltree=False, return_dists=True)
    nnn2 = _compute_nearest(x, y, use_balltree=True, return_dists=True)
    assert_array_equal(nnn1[0], nn_true)
    assert_array_equal(nnn1[1], np.zeros_like(nn1))  # all dists should be 0
    assert_equal(len(nnn1), len(nnn2))
    for nn1, nn2 in zip(nnn1, nnn2):
        assert_array_equal(nn1, nn2)


@sample.requires_sample_data
def test_make_morph_maps():
    """Test reading and creating morph maps
    """
    mmap = read_morph_map('fsaverage', 'sample', subjects_dir=subjects_dir)
    mmap2 = _make_morph_map('fsaverage', 'sample', subjects_dir=subjects_dir)
    assert_equal(len(mmap), len(mmap2))
    for m1, m2 in zip(mmap, mmap2):
        # deal with sparse matrix stuff
        diff = (m1 - m2).data
        assert_allclose(diff, np.zeros_like(diff), atol=1e-3, rtol=0)


@sample.requires_sample_data
def test_io_bem_surfaces():
    """Test reading of bem surfaces
    """
    surf = read_bem_surfaces(fname, add_geom=True)
    surf = read_bem_surfaces(fname, add_geom=False)
    print("Number of surfaces : %d" % len(surf))

    write_bem_surface(op.join(tempdir, 'bem_surf.fif'), surf[0])
    surf_read = read_bem_surfaces(op.join(tempdir, 'bem_surf.fif'),
                                  add_geom=False)

    for key in surf[0].keys():
        assert_array_almost_equal(surf[0][key], surf_read[0][key])


@sample.requires_sample_data
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
    for n_tri in [4, 3, 2]:  # quadric decimation creates even numbered output.
        _, this_tris = decimate_surface(points, tris, n_tri)
        assert_true(len(this_tris) == n_tri if not n_tri % 2 else 2)
    nirvana = 5
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, nirvana]])
    assert_raises(ValueError, decimate_surface, points, tris, n_tri)
