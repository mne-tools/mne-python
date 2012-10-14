import os.path as op
from nose.tools import assert_true
import numpy as np

from mne.datasets import sample
from mne import read_source_spaces, vertex_to_mni

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')


def test_read_source_spaces():
    """Testing reading of source space meshes
    """
    src = read_source_spaces(fname, add_geom=False)
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


def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates
    """
    # these were random vertices pulled from "sample" in mne_analyze
    # but mne_analyze won't load the xfm file! So we must use fsaverage,
    # which is sily because the xfm is the identity matrix..
    #vertices = [109445, 82962, 137444]
    #coords = [[-33.3, 11.5, 50.7], [51.8, -15.4, 30.5], [37.6, 38.4, 57.1]]
    #hemi = [0, 1, 1]
    #coords_2 = vertex_to_mni(vertices, hemis, 'sample')
    vertices = [148611, 157229, 95466]
    coords = [[-55.7, -36.6, -9.6], [-48.5, -35.7, -1.1], [44.0, -34.9, -0.9]]
    hemis = [0, 0, 1]  # , 1]
    coords_2 = np.round(vertex_to_mni(vertices, hemis, 'fsaverage'), 1)
    for vi in range(len(vertices)):
        assert_true(coords[vi] == coords_2[vi].tolist())