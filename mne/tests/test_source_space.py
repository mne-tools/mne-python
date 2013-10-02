import os
import os.path as op
from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
import warnings

from mne.datasets import sample
from mne import (read_source_spaces, vertex_to_mni, write_source_spaces,
                 setup_source_space, setup_volume_source_space)
from mne.utils import (_TempDir, requires_fs_or_nibabel, requires_nibabel,
                       requires_freesurfer, run_subprocess,
                       requires_mne)
from mne.surface import _accumulate_normals, _triangle_neighbors

from scipy.spatial.distance import cdist

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-5120-bem.fif')
fname_mri = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')

tempdir = _TempDir()


@requires_mne
def test_discrete_source_space():
    """Test setting up (and reading/writing) discrete source spaces
    """
    src = read_source_spaces(fname)
    v = src[0]['vertno']

    # let's make a discrete version with the C code, and with ours
    temp_name = op.join(tempdir, 'temp-src.fif')
    try:
        # save
        temp_pos = op.join(tempdir, 'temp-pos.txt')
        np.savetxt(temp_pos, np.c_[src[0]['rr'][v], src[0]['nn'][v]])
        # let's try the spherical one (no bem or surf supplied)
        run_subprocess(['mne_volume_source_space', '--meters',
                        '--pos',  temp_pos, '--src', temp_name])
        src_c = read_source_spaces(temp_name)
        src_new = setup_volume_source_space('sample', None,
                                            pos=dict(rr=src[0]['rr'][v],
                                                     nn=src[0]['nn'][v]),
                                            subjects_dir=subjects_dir)
        _compare_source_spaces(src_c, src_new, mode='approx')
        assert_allclose(src[0]['rr'][v], src_new[0]['rr'],
                        rtol=1e-3, atol=1e-6)
        assert_allclose(src[0]['nn'][v], src_new[0]['nn'],
                        rtol=1e-3, atol=1e-6)

        # now do writing
        write_source_spaces(temp_name, src_c)
        src_c2 = read_source_spaces(temp_name)
        _compare_source_spaces(src_c, src_c2)
    finally:
        if op.isfile(temp_name):
            os.remove(temp_name)


@requires_mne
def test_volume_source_space():
    """Test setting up volume source spaces
    """
    fname_vol = op.join(data_path, 'subjects', 'sample', 'bem',
                        'volume-7mm-src.fif')
    src = read_source_spaces(fname_vol)
    temp_name = op.join(tempdir, 'temp-src.fif')
    try:
        # The one in the sample dataset (uses bem as bounds)
        src_new = setup_volume_source_space('sample', temp_name, pos=7.0,
                                            bem=fname_bem, mri=fname_mri,
                                            subjects_dir=subjects_dir)
        _compare_source_spaces(src, src_new, mode='approx')
        src_new = read_source_spaces(temp_name)
        _compare_source_spaces(src, src_new, mode='approx')

        # let's try the spherical one (no bem or surf supplied)
        run_subprocess(['mne_volume_source_space',
                        '--grid',  '15.0',
                        '--src', temp_name,
                        '--mri', fname_mri])
        src = read_source_spaces(temp_name)
        src_new = setup_volume_source_space('sample', temp_name, pos=15.0,
                                            mri=fname_mri,
                                            subjects_dir=subjects_dir)
        _compare_source_spaces(src, src_new, mode='approx')
    finally:
        if op.isfile(temp_name):
            os.remove(temp_name)


def test_triangle_neighbors():
    """Test efficient vertex neighboring triangles for surfaces"""
    this = read_source_spaces(fname)[0]
    this['neighbor_tri'] = [list() for _ in xrange(this['np'])]
    for p in xrange(this['ntri']):
        verts = this['tris'][p]
        this['neighbor_tri'][verts[0]].append(p)
        this['neighbor_tri'][verts[1]].append(p)
        this['neighbor_tri'][verts[2]].append(p)
    this['neighbor_tri'] = [np.array(nb, int) for nb in this['neighbor_tri']]

    neighbor_tri = _triangle_neighbors(this['tris'], this['np'])
    assert_true(np.array_equal(nt1, nt2)
                for nt1, nt2 in zip(neighbor_tri, this['neighbor_tri']))


def test_accumulate_normals():
    """Test efficient normal accumulation for surfaces"""
    # set up comparison
    rng = np.random.RandomState(0)
    n_pts = int(1.6e5)  # approx number in sample source space
    n_tris = int(3.2e5)
    # use all positive to make a worst-case for cumulative summation
    # (real "nn" vectors will have both positive and negative values)
    tris = (rng.rand(n_tris, 1) * (n_pts - 2)).astype(int)
    tris = np.c_[tris, tris + 1, tris + 2]
    tri_nn = rng.rand(n_tris, 3)
    this = dict(tris=tris, np=n_pts, ntri=n_tris, tri_nn=tri_nn)

    # cut-and-paste from original code in surface.py:
    #    Find neighboring triangles and accumulate vertex normals
    this['nn'] = np.zeros((this['np'], 3))
    for p in xrange(this['ntri']):
        # vertex normals
        verts = this['tris'][p]
        this['nn'][verts, :] += this['tri_nn'][p, :]
    nn = _accumulate_normals(this['tris'], this['tri_nn'], this['np'])

    # the moment of truth (or reckoning)
    assert_allclose(nn, this['nn'], rtol=1e-7, atol=1e-7)


def test_setup_source_space():
    """Test setting up ico, oct, and all source spaces
    """
    fname_all = op.join(data_path, 'subjects', 'sample', 'bem',
                        'sample-all-src.fif')
    fname_ico = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                        'fsaverage-ico-5-src.fif')
    # first lets test some input params
    assert_raises(ValueError, setup_source_space, 'sample', spacing='oct')
    assert_raises(ValueError, setup_source_space, 'sample', spacing='octo')
    assert_raises(ValueError, setup_source_space, 'sample', spacing='oct6e')
    assert_raises(ValueError, setup_source_space, 'sample', spacing='7emm')
    assert_raises(ValueError, setup_source_space, 'sample', spacing='alls')
    assert_raises(IOError, setup_source_space, 'sample', spacing='oct6',
                  subjects_dir=subjects_dir)

    # ico 5 (fsaverage) - write to temp file
    src = read_source_spaces(fname_ico)
    temp_name = op.join(tempdir, 'temp-src.fif')
    with warnings.catch_warnings(True):  # sklearn equiv neighbors
        src_new = setup_source_space('fsaverage', temp_name, spacing='ico5',
                                     subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')

    # oct-6 (sample) - auto filename + IO
    src = read_source_spaces(fname)
    temp_name = op.join(tempdir, 'temp-src.fif')
    with warnings.catch_warnings(True):  # sklearn equiv neighbors
        src_new = setup_source_space('sample', temp_name, spacing='oct6',
                                     subjects_dir=subjects_dir,
                                     overwrite=True)
    _compare_source_spaces(src, src_new, mode='approx')
    src_new = read_source_spaces(temp_name)
    _compare_source_spaces(src, src_new, mode='approx')

    # all source points - no file writing
    src = read_source_spaces(fname_all)
    src_new = setup_source_space('sample', None, spacing='all',
                                 subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')


def test_read_source_spaces():
    """Test reading of source space meshes
    """
    src = read_source_spaces(fname, add_geom=True)

    # 3D source space
    lh_points = src[0]['rr']
    lh_faces = src[0]['tris']
    lh_use_faces = src[0]['use_tris']
    rh_points = src[1]['rr']
    rh_faces = src[1]['tris']
    rh_use_faces = src[1]['use_tris']
    assert_true(lh_faces.min() == 0)
    assert_true(lh_faces.max() == lh_points.shape[0] - 1)
    assert_true(lh_use_faces.min() >= 0)
    assert_true(lh_use_faces.max() <= lh_points.shape[0] - 1)
    assert_true(rh_faces.min() == 0)
    assert_true(rh_faces.max() == rh_points.shape[0] - 1)
    assert_true(rh_use_faces.min() >= 0)
    assert_true(rh_use_faces.max() <= rh_points.shape[0] - 1)


def test_write_source_space():
    """Test writing and reading of source spaces
    """
    src0 = read_source_spaces(fname, add_geom=False)
    write_source_spaces(op.join(tempdir, 'tmp.fif'), src0)
    src1 = read_source_spaces(op.join(tempdir, 'tmp.fif'), add_geom=False)
    _compare_source_spaces(src0, src1)


def _compare_source_spaces(src0, src1, mode='exact'):
    for s0, s1 in zip(src0, src1):
        for name in ['nuse', 'ntri', 'np', 'type', 'id']:
            print name
            assert_true(s0[name] == s1[name])
        for name in ['subject_his_id']:
            if name in s0 or name in s1:
                print name
                assert_true(s0[name] == s1[name])
        for name in ['interpolator']:
            if name in s0 or name in s1:
                print name
                diffs = (s0['interpolator'] - s1['interpolator']).data
                assert_true(np.sqrt(np.mean(diffs ** 2)) < 0.05)  # 5%
        for name in ['nn', 'rr', 'nuse_tri', 'coord_frame', 'tris']:
            print name
            if s0[name] is None:
                assert_true(s1[name] is None)
            else:
                if mode == 'exact':
                    assert_array_equal(s0[name], s1[name])
                elif mode == 'approx':
                    assert_allclose(s0[name], s1[name], rtol=1e-3, atol=1e-4)
                else:
                    raise RuntimeError('unknown mode')
        if mode == 'exact':
            for name in ['inuse', 'vertno', 'use_tris']:
                assert_array_equal(s0[name], s1[name])
            # these fields will exist if patch info was added, these are
            # not tested in mode == 'approx'
            for name in ['nearest', 'nearest_dist']:
                print name
                if s0[name] is None:
                    assert_true(s1[name] is None)
                else:
                    assert_array_equal(s0[name], s1[name])
            for name in ['dist_limit']:
                print name
                assert_true(s0[name] == s1[name])
            for name in ['dist']:
                if s0[name] is not None:
                    assert_true(s1[name].shape == s0[name].shape)
                    assert_true(len((s0['dist'] - s1['dist']).data) == 0)
            for name in ['pinfo']:
                if s0[name] is not None:
                    assert_true(len(s0[name]) == len(s1[name]))
                    for p1, p2 in zip(s0[name], s1[name]):
                        assert_true(all(p1 == p2))
        elif mode == 'approx':
            # deal with vertno, inuse, and use_tris carefully
            assert_array_equal(s0['vertno'], np.where(s0['inuse'])[0])
            assert_array_equal(s1['vertno'], np.where(s1['inuse'])[0])
            assert_equal(len(s0['vertno']), len(s1['vertno']))
            agreement = np.mean(s0['inuse'] == s1['inuse'])
            assert_true(agreement > 0.99)
            if agreement < 1.0:
                # make sure mismatched vertno are within 1.5mm
                v0 = np.setdiff1d(s0['vertno'], s1['vertno'])
                v1 = np.setdiff1d(s1['vertno'], s0['vertno'])
                dists = cdist(s0['rr'][v0], s1['rr'][v1])
                assert_allclose(np.min(dists, axis=1), np.zeros(len(v0)),
                                atol=1.5e-3)
            if s0['use_tris'] is not None:  # for "spacing"
                assert_array_equal(s0['use_tris'].shape, s1['use_tris'].shape)
            else:
                assert_true(s1['use_tris'] is None)
            assert_true(np.mean(s0['use_tris'] == s1['use_tris']) > 0.99)
    # The above "if s0[name] is not None" can be removed once the sample
    # dataset is updated to have a source space with distance info
    for name in ['working_dir', 'command_line']:
        if mode == 'exact':
            assert_true(src0.info[name] == src1.info[name])
        elif mode == 'approx':
            assert_true(name in src0.info)
            assert_true(name in src1.info)


@requires_fs_or_nibabel
def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates
    """
    # obtained using "tksurfer (sample/fsaverage) (l/r)h white"
    vertices = [100960, 7620, 150549, 96761]
    coords_s = np.array([[-60.86, -11.18, -3.19], [-36.46, -93.18, -2.36],
                         [-38.00, 50.08, -10.61], [47.14, 8.01, 46.93]])
    coords_f = np.array([[-41.28, -40.04, 18.20], [-6.05, 49.74, -18.15],
                         [-61.71, -14.55, 20.52], [21.70, -60.84, 25.02]])
    hemis = [0, 0, 0, 1]
    for coords, subj in zip([coords_s, coords_f], ['sample', 'fsaverage']):
        coords_2 = vertex_to_mni(vertices, hemis, subj)
        # less than 1mm error
        assert_allclose(coords, coords_2, atol=1.0)


@requires_freesurfer
@requires_nibabel
def test_vertex_to_mni_fs_nibabel():
    """Test equivalence of vert_to_mni for nibabel and freesurfer
    """
    n_check = 1000
    for subject in ['sample', 'fsaverage']:
        vertices = np.random.randint(0, 100000, n_check)
        hemis = np.random.randint(0, 1, n_check)
        coords = vertex_to_mni(vertices, hemis, subject, mode='nibabel')
        coords_2 = vertex_to_mni(vertices, hemis, subject, mode='freesurfer')
        # less than 0.1 mm error
        assert_allclose(coords, coords_2, atol=0.1)
