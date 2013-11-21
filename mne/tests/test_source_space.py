import os.path as op
from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from mne.datasets import sample
from mne import read_source_spaces, vertex_to_mni, write_source_spaces
from mne.utils import _TempDir, requires_fs_or_nibabel, requires_nibabel, \
                      requires_freesurfer

data_path = sample.data_path()
fname = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')
fname_nodist = op.join(data_path, 'subjects', 'sample', 'bem',
                       'sample-oct-6-orig-src.fif')

tempdir = _TempDir()


def test_read_source_spaces():
    """Test reading of source space meshes
    """
    src = read_source_spaces(fname, add_geom=True)
    print src

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
    src0_old = read_source_spaces(fname, add_geom=False)
    write_source_spaces(op.join(tempdir, 'tmp.fif'), src0)
    src1 = read_source_spaces(op.join(tempdir, 'tmp.fif'), add_geom=False)
    for orig in [src0, src0_old]:
        for s0, s1 in zip(src0, src1):
            for name in ['nuse', 'dist_limit', 'ntri', 'np', 'type', 'id',
                         'subject_his_id']:
                assert_true(s0[name] == s1[name])
            for name in ['nn', 'rr', 'inuse', 'vertno', 'nuse_tri',
                         'coord_frame', 'use_tris', 'tris', 'nearest',
                         'nearest_dist']:
                assert_array_equal(s0[name], s1[name])
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
        assert_true(src0.info[name] == src1.info[name])


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
