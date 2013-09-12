import os.path as op
from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal

from mne.datasets import sample
from mne import read_source_spaces, vertex_to_mni, write_source_spaces, \
                setup_source_space
from mne.utils import _TempDir, requires_fs_or_nibabel, requires_nibabel, \
                      requires_freesurfer

data_path = sample.data_path()
fname = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')
fname_all = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-all-src.fif')
fname_spacing = op.join(data_path, 'subjects', 'sample', 'bem',
                        'sample-7-src.fif')
fname_ico = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                    'fsaverage-ico-5-src.fif')
fname_morph = op.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-fsaverage-ico-5-src.fif')
fname_nodist = op.join(data_path, 'subjects', 'sample', 'bem',
                       'sample-oct-6-orig-src.fif')
subjects_dir = op.join(data_path, 'subjects')

tempdir = _TempDir()


def test_setup_source_space():
    """Test setting up a source space
    """
    # first lets test some input params
    assert_raises(ValueError, setup_source_space, 'sample', oct=6, ico=6)
    assert_raises(IOError, setup_source_space, 'sample', oct=6,
                  subjects_dir=subjects_dir)

    # ico 5 fsaverage->sample morph - no file writing
    src = read_source_spaces(fname_morph)
    src_new = setup_source_space('fsaverage', False, ico=5, morph='sample',
                                 subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')

    # all source points - no file writing
    src = read_source_spaces(fname_all)
    src_new = setup_source_space('sample', False, use_all=True,
                                 subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')

    # spacing 7 (sample) - no file writing
    src = read_source_spaces(fname_spacing)
    src_new = setup_source_space('sample', False, spacing=7,
                                 subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')

    # ico 5 (fsaverage) - write to temp file
    src = read_source_spaces(fname_ico)
    temp_name = op.join(tempdir, 'temp-src.fif')
    src_new = setup_source_space('fsaverage', temp_name, ico=5,
                                 subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')


    # oct-6 (sample) - auto filename + IO
    src = read_source_spaces(fname)
    temp_name = op.join(tempdir, 'temp-src.fif')
    src_new = setup_source_space('sample', temp_name, oct=6,
                                 subjects_dir=subjects_dir, overwrite=True)
    _compare_source_spaces(src, src_new, mode='approx')
    src_new = read_source_spaces(temp_name)
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
        for name in ['nuse', 'dist_limit', 'ntri', 'np', 'type', 'id',
                     'subject_his_id']:
            print name
            assert_true(s0[name] == s1[name])
        for name in ['nn', 'rr', 'nuse_tri', 'coord_frame', 'tris', 'nearest',
                     'nearest_dist']:
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
        elif mode == 'approx':
            # deal with vertno, inuse, and use_tris carefully
            assert_array_equal(s0['vertno'], np.where(s0['inuse'])[0])
            assert_array_equal(s1['vertno'], np.where(s1['inuse'])[0])
            assert_equal(len(s0['vertno']), len(s1['vertno']))
            assert_true(np.mean(s0['inuse'] == s1['inuse']) > 0.99)
            if s0['use_tris'] is not None:  # for "spacing"
                assert_array_equal(s0['use_tris'].shape, s1['use_tris'].shape)
            else:
                assert_true(s1['use_tris'] is None)
            assert_true(np.mean(s0['use_tris'] == s1['use_tris']) > 0.99)
        for name in ['dist']:
            if s0[name] is not None:
                assert_true(s1[name].shape == s0[name].shape)
                assert_true(len((s0['dist'] - s1['dist']).data) == 0)
        for name in ['pinfo']:
            if s0[name] is not None:
                assert_true(len(s0[name]) == len(s1[name]))
                for p1, p2 in zip(s0[name], s1[name]):
                    assert_true(all(p1 == p2))
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
