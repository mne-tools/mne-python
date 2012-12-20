import os.path as op
from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_equal

from mne.datasets import sample
from mne import read_source_spaces, vertex_to_mni, write_source_spaces
from mne.utils import _TempDir

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')

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
    write_source_spaces(op.join(tempdir, 'tmp.fif'), src0)
    src1 = read_source_spaces(op.join(tempdir, 'tmp.fif'))
    for s0, s1 in zip(src0, src1):
        for name in ['nuse', 'dist_limit', 'ntri', 'np', 'type', 'id',
                     'subject_his_id']:
            assert_true(s0[name] == s1[name])
        for name in ['nn', 'rr', 'inuse', 'vertno', 'nuse_tri', 'coord_frame',
                     'use_tris', 'tris', 'nearest', 'nearest_dist']:
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


def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates
    """
    # these were random vertices pulled from "sample" in mne_analyze
    # but mne_analyze won't load the xfm file! So we must use fsaverage,
    # which is sily because the xfm is the identity matrix..
    # vertices = [109445, 82962, 137444]
    # coords = [[-33.3, 11.5, 50.7], [51.8, -15.4, 30.5], [37.6, 38.4, 57.1]]
    # hemi = [0, 1, 1]
    # coords_2 = vertex_to_mni(vertices, hemis, 'sample')
    vertices = [148611, 157229, 95466]
    coords = [[-55.7, -36.6, -9.6], [-48.5, -35.7, -1.1], [44.0, -34.9, -0.9]]
    hemis = [0, 0, 1]  # , 1]
    coords_2 = np.round(vertex_to_mni(vertices, hemis, 'fsaverage'), 1)
    for vi in range(len(vertices)):
        assert_true(coords[vi] == coords_2[vi].tolist())
