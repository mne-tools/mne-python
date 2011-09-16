import os.path as op
from nose.tools import assert_true

from ..datasets import sample
from .. import read_source_spaces

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-eeg-oct-6p-fwd.fif')


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
