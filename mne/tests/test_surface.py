from __future__ import print_function
import os
import os.path as op
import numpy as np
import warnings
from shutil import copyfile
from scipy import sparse
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_array_equal, assert_allclose, assert_equal

from mne.datasets import testing
from mne import read_surface, write_surface, decimate_surface
from mne.surface import (read_morph_map, _compute_nearest,
                         fast_cross_3d, get_head_surf, read_curvature,
                         get_meg_helmet_surf)
from mne.utils import (_TempDir, requires_mayavi, requires_tvtk,
                       run_tests_if_main, slow_test, object_diff)
from mne.io import read_info
from mne.transforms import _get_trans

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname = op.join(subjects_dir, 'sample', 'bem',
                'sample-1280-1280-1280-bem-sol.fif')

warnings.simplefilter('always')
rng = np.random.RandomState(0)


def test_helmet():
    """Test loading helmet surfaces."""
    base_dir = op.join(op.dirname(__file__), '..', 'io')
    fname_raw = op.join(base_dir, 'tests', 'data', 'test_raw.fif')
    fname_kit_raw = op.join(base_dir, 'kit', 'tests', 'data',
                            'test_bin_raw.fif')
    fname_bti_raw = op.join(base_dir, 'bti', 'tests', 'data',
                            'exported4D_linux_raw.fif')
    fname_ctf_raw = op.join(base_dir, 'tests', 'data', 'test_ctf_raw.fif')
    fname_trans = op.join(base_dir, 'tests', 'data',
                          'sample-audvis-raw-trans.txt')
    trans = _get_trans(fname_trans)[0]
    for fname in [fname_raw, fname_kit_raw, fname_bti_raw, fname_ctf_raw]:
        helmet = get_meg_helmet_surf(read_info(fname), trans)
        assert_equal(len(helmet['rr']), 304)  # they all have 304 verts
        assert_equal(len(helmet['rr']), len(helmet['nn']))


@testing.requires_testing_data
def test_head():
    """Test loading the head surface."""
    surf_1 = get_head_surf('sample', subjects_dir=subjects_dir)
    surf_2 = get_head_surf('sample', 'head', subjects_dir=subjects_dir)
    assert_true(len(surf_1['rr']) < len(surf_2['rr']))  # BEM vs dense head
    assert_raises(TypeError, get_head_surf, subject=None,
                  subjects_dir=subjects_dir)


def test_huge_cross():
    """Test cross product with lots of elements."""
    x = rng.rand(100000, 3)
    y = rng.rand(1, 3)
    z = np.cross(x, y)
    zz = fast_cross_3d(x, y)
    assert_array_equal(z, zz)


def test_compute_nearest():
    """Test nearest neighbor searches."""
    x = rng.randn(500, 3)
    x /= np.sqrt(np.sum(x ** 2, axis=1))[:, None]
    nn_true = rng.permutation(np.arange(500, dtype=np.int))[:20]
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


@slow_test
@testing.requires_testing_data
def test_make_morph_maps():
    """Test reading and creating morph maps."""
    # make a new fake subjects_dir
    tempdir = _TempDir()
    for subject in ('sample', 'sample_ds', 'fsaverage_ds'):
        os.mkdir(op.join(tempdir, subject))
        os.mkdir(op.join(tempdir, subject, 'surf'))
        for hemi in ['lh', 'rh']:
            args = [subject, 'surf', hemi + '.sphere.reg']
            copyfile(op.join(subjects_dir, *args),
                     op.join(tempdir, *args))

    # this should trigger the creation of morph-maps dir and create the map
    with warnings.catch_warnings(record=True):
        mmap = read_morph_map('fsaverage_ds', 'sample_ds', tempdir)
    mmap2 = read_morph_map('fsaverage_ds', 'sample_ds', subjects_dir)
    assert_equal(len(mmap), len(mmap2))
    for m1, m2 in zip(mmap, mmap2):
        # deal with sparse matrix stuff
        diff = (m1 - m2).data
        assert_allclose(diff, np.zeros_like(diff), atol=1e-3, rtol=0)

    # This will also trigger creation, but it's trivial
    with warnings.catch_warnings(record=True):
        mmap = read_morph_map('sample', 'sample', subjects_dir=tempdir)
    for mm in mmap:
        assert_true((mm - sparse.eye(mm.shape[0], mm.shape[0])).sum() == 0)


@testing.requires_testing_data
def test_io_surface():
    """Test reading and writing of Freesurfer surface mesh files."""
    tempdir = _TempDir()
    fname_quad = op.join(data_path, 'subjects', 'bert', 'surf',
                         'lh.inflated.nofix')
    fname_tri = op.join(data_path, 'subjects', 'fsaverage', 'surf',
                        'lh.inflated')
    for fname in (fname_quad, fname_tri):
        with warnings.catch_warnings(record=True) as w:
            pts, tri, vol_info = read_surface(fname, read_metadata=True)
        assert_true(all('No volume info' in str(ww.message) for ww in w))
        write_surface(op.join(tempdir, 'tmp'), pts, tri, volume_info=vol_info)
        with warnings.catch_warnings(record=True) as w:  # No vol info
            c_pts, c_tri, c_vol_info = read_surface(op.join(tempdir, 'tmp'),
                                                    read_metadata=True)
        assert_array_equal(pts, c_pts)
        assert_array_equal(tri, c_tri)
        assert_equal(object_diff(vol_info, c_vol_info), '')


@testing.requires_testing_data
def test_read_curv():
    """Test reading curvature data."""
    fname_curv = op.join(data_path, 'subjects', 'fsaverage', 'surf', 'lh.curv')
    fname_surf = op.join(data_path, 'subjects', 'fsaverage', 'surf',
                         'lh.inflated')
    bin_curv = read_curvature(fname_curv)
    rr = read_surface(fname_surf)[0]
    assert_true(len(bin_curv) == len(rr))
    assert_true(np.logical_or(bin_curv == 0, bin_curv == 1).all())


@requires_tvtk
@requires_mayavi
def test_decimate_surface():
    """Test triangular surface decimation."""
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


run_tests_if_main()
