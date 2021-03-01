import os
import os.path as op
from shutil import copyfile

import numpy as np
from scipy import sparse

import pytest
from numpy.testing import assert_array_equal, assert_allclose, assert_equal

from mne.datasets import testing
from mne import (read_surface, write_surface, decimate_surface, pick_types,
                 dig_mri_distances)
from mne.surface import (read_morph_map, _compute_nearest, _tessellate_sphere,
                         fast_cross_3d, get_head_surf, read_curvature,
                         get_meg_helmet_surf, _normal_orth, _read_patch)
from mne.utils import (_TempDir, requires_vtk, catch_logging,
                       run_tests_if_main, object_diff, requires_freesurfer)
from mne.io import read_info
from mne.io.constants import FIFF
from mne.transforms import _get_trans

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname = op.join(subjects_dir, 'sample', 'bem',
                'sample-1280-1280-1280-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')

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
    new_info = read_info(fname_raw)
    artemis_info = new_info.copy()
    for pick in pick_types(new_info, meg=True):
        new_info['chs'][pick]['coil_type'] = 9999
        artemis_info['chs'][pick]['coil_type'] = \
            FIFF.FIFFV_COIL_ARTEMIS123_GRAD
    for info, n, name in [(read_info(fname_raw), 304, '306m'),
                          (read_info(fname_kit_raw), 150, 'KIT'),  # Delaunay
                          (read_info(fname_bti_raw), 304, 'Magnes'),
                          (read_info(fname_ctf_raw), 342, 'CTF'),
                          (new_info, 102, 'unknown'),  # Delaunay
                          (artemis_info, 102, 'ARTEMIS123'),  # Delaunay
                          ]:
        with catch_logging() as log:
            helmet = get_meg_helmet_surf(info, trans, verbose=True)
        log = log.getvalue()
        assert name in log
        assert_equal(len(helmet['rr']), n)
        assert_equal(len(helmet['rr']), len(helmet['nn']))


@testing.requires_testing_data
def test_head():
    """Test loading the head surface."""
    surf_1 = get_head_surf('sample', subjects_dir=subjects_dir)
    surf_2 = get_head_surf('sample', 'head', subjects_dir=subjects_dir)
    assert len(surf_1['rr']) < len(surf_2['rr'])  # BEM vs dense head
    pytest.raises(TypeError, get_head_surf, subject=None,
                  subjects_dir=subjects_dir)


def test_fast_cross_3d():
    """Test cross product with lots of elements."""
    x = rng.rand(100000, 3)
    y = rng.rand(1, 3)
    z = np.cross(x, y)
    zz = fast_cross_3d(x, y)
    assert_array_equal(z, zz)
    # broadcasting and non-2D
    zz = fast_cross_3d(x[:, np.newaxis], y[0])
    assert_array_equal(z, zz[:, 0])


def test_compute_nearest():
    """Test nearest neighbor searches."""
    x = rng.randn(500, 3)
    x /= np.sqrt(np.sum(x ** 2, axis=1))[:, None]
    nn_true = rng.permutation(np.arange(500, dtype=np.int64))[:20]
    y = x[nn_true]

    nn1 = _compute_nearest(x, y, method='BallTree')
    nn2 = _compute_nearest(x, y, method='cKDTree')
    nn3 = _compute_nearest(x, y, method='cdist')
    assert_array_equal(nn_true, nn1)
    assert_array_equal(nn_true, nn2)
    assert_array_equal(nn_true, nn3)

    # test distance support
    nnn1 = _compute_nearest(x, y, method='BallTree', return_dists=True)
    nnn2 = _compute_nearest(x, y, method='cKDTree', return_dists=True)
    nnn3 = _compute_nearest(x, y, method='cdist', return_dists=True)
    assert_array_equal(nnn1[0], nn_true)
    assert_array_equal(nnn1[1], np.zeros_like(nn1))  # all dists should be 0
    assert_equal(len(nnn1), len(nnn2))
    for nn1, nn2, nn3 in zip(nnn1, nnn2, nnn3):
        assert_array_equal(nn1, nn2)
        assert_array_equal(nn1, nn3)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_make_morph_maps():
    """Test reading and creating morph maps."""
    # make a new fake subjects_dir
    tempdir = _TempDir()
    for subject in ('sample', 'sample_ds', 'fsaverage_ds'):
        os.mkdir(op.join(tempdir, subject))
        os.mkdir(op.join(tempdir, subject, 'surf'))
        regs = ('reg', 'left_right') if subject == 'fsaverage_ds' else ('reg',)
        for hemi in ['lh', 'rh']:
            for reg in regs:
                args = [subject, 'surf', hemi + '.sphere.' + reg]
                copyfile(op.join(subjects_dir, *args),
                         op.join(tempdir, *args))

    for subject_from, subject_to, xhemi in (
            ('fsaverage_ds', 'sample_ds', False),
            ('fsaverage_ds', 'fsaverage_ds', True)):
        # trigger the creation of morph-maps dir and create the map
        with catch_logging() as log:
            mmap = read_morph_map(subject_from, subject_to, tempdir,
                                  xhemi=xhemi, verbose=True)
        log = log.getvalue()
        assert 'does not exist' in log
        assert 'Creating' in log
        mmap2 = read_morph_map(subject_from, subject_to, subjects_dir,
                               xhemi=xhemi)
        assert_equal(len(mmap), len(mmap2))
        for m1, m2 in zip(mmap, mmap2):
            # deal with sparse matrix stuff
            diff = (m1 - m2).data
            assert_allclose(diff, np.zeros_like(diff), atol=1e-3, rtol=0)

    # This will also trigger creation, but it's trivial
    with pytest.warns(None):
        mmap = read_morph_map('sample', 'sample', subjects_dir=tempdir)
    for mm in mmap:
        assert (mm - sparse.eye(mm.shape[0], mm.shape[0])).sum() == 0


@testing.requires_testing_data
def test_io_surface():
    """Test reading and writing of Freesurfer surface mesh files."""
    tempdir = _TempDir()
    fname_quad = op.join(data_path, 'subjects', 'bert', 'surf',
                         'lh.inflated.nofix')
    fname_tri = op.join(data_path, 'subjects', 'sample', 'bem',
                        'inner_skull.surf')
    for fname in (fname_quad, fname_tri):
        with pytest.warns(None):  # no volume info
            pts, tri, vol_info = read_surface(fname, read_metadata=True)
        write_surface(op.join(tempdir, 'tmp'), pts, tri, volume_info=vol_info,
                      overwrite=True)
        with pytest.warns(None):  # no volume info
            c_pts, c_tri, c_vol_info = read_surface(op.join(tempdir, 'tmp'),
                                                    read_metadata=True)
        assert_array_equal(pts, c_pts)
        assert_array_equal(tri, c_tri)
        assert_equal(object_diff(vol_info, c_vol_info), '')
        if fname != fname_tri:  # don't bother testing wavefront for the bigger
            continue

        # Test writing/reading a Wavefront .obj file
        write_surface(op.join(tempdir, 'tmp.obj'), pts, tri, volume_info=None,
                      overwrite=True)
        c_pts, c_tri = read_surface(op.join(tempdir, 'tmp.obj'),
                                    read_metadata=False)
        assert_array_equal(pts, c_pts)
        assert_array_equal(tri, c_tri)

    # reading patches (just a smoke test, let the flatmap viz tests be more
    # complete)
    fname_patch = op.join(
        data_path, 'subjects', 'fsaverage', 'surf', 'rh.cortex.patch.flat')
    _read_patch(fname_patch)


@testing.requires_testing_data
def test_read_curv():
    """Test reading curvature data."""
    fname_curv = op.join(data_path, 'subjects', 'fsaverage', 'surf', 'lh.curv')
    fname_surf = op.join(data_path, 'subjects', 'fsaverage', 'surf',
                         'lh.inflated')
    bin_curv = read_curvature(fname_curv)
    rr = read_surface(fname_surf)[0]
    assert len(bin_curv) == len(rr)
    assert np.logical_or(bin_curv == 0, bin_curv == 1).all()


@requires_vtk
def test_decimate_surface_vtk():
    """Test triangular surface decimation."""
    points = np.array([[-0.00686118, -0.10369860, 0.02615170],
                       [-0.00713948, -0.10370162, 0.02614874],
                       [-0.00686208, -0.10368247, 0.02588313],
                       [-0.00713987, -0.10368724, 0.02587745]])
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, 0]])
    for n_tri in [4, 3, 2]:  # quadric decimation creates even numbered output.
        _, this_tris = decimate_surface(points, tris, n_tri)
        assert len(this_tris) == n_tri if not n_tri % 2 else 2
    with pytest.raises(ValueError, match='exceeds number of original'):
        decimate_surface(points, tris, len(tris) + 1)
    nirvana = 5
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, nirvana]])
    pytest.raises(ValueError, decimate_surface, points, tris, n_tri)


@requires_freesurfer('mris_sphere')
def test_decimate_surface_sphere():
    """Test sphere mode of decimation."""
    rr, tris = _tessellate_sphere(3)
    assert len(rr) == 66
    assert len(tris) == 128
    for kind, n_tri in [('ico', 20), ('oct', 32)]:
        with catch_logging() as log:
            _, tris_new = decimate_surface(
                rr, tris, n_tri, method='sphere', verbose=True)
        log = log.getvalue()
        assert 'Freesurfer' in log
        assert kind in log
        assert len(tris_new) == n_tri


@pytest.mark.parametrize('dig_kinds, exclude, count, bounds, outliers', [
    ('auto', False, 72, (0.001, 0.002), 0),
    (('eeg', 'extra', 'cardinal', 'hpi'), False, 146, (0.002, 0.003), 1),
    (('eeg', 'extra', 'cardinal', 'hpi'), True, 139, (0.001, 0.002), 0),
])
@testing.requires_testing_data
def test_dig_mri_distances(dig_kinds, exclude, count, bounds, outliers):
    """Test the trans obtained by coregistration."""
    info = read_info(fname_raw)
    dists = dig_mri_distances(info, fname_trans, 'sample', subjects_dir,
                              dig_kinds=dig_kinds, exclude_frontal=exclude)
    assert dists.shape == (count,)
    assert bounds[0] < np.mean(dists) < bounds[1]
    assert np.sum(dists > 0.03) == outliers


def test_normal_orth():
    """Test _normal_orth."""
    nns = np.eye(3)
    for nn in nns:
        ori = _normal_orth(nn)
        assert_allclose(ori[2], nn, atol=1e-12)


run_tests_if_main()
