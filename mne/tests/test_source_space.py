import os.path as op

# from numpy.testing import assert_array_almost_equal

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-eeg-oct-6-fwd.fif')

def test_read_source_spaces():
    """Testing reading of source space meshes
    """
    src = mne.read_source_spaces(fname, add_geom=False)
    src = mne.read_source_spaces(fname, add_geom=True)

    # 3D source space
    lh_points = src[0]['rr']
    lh_faces = src[0]['use_tris']
    rh_points = src[1]['rr']
    rh_faces = src[1]['use_tris']
    # XXX : test something
