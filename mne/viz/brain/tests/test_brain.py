import mne
import pytest
import os.path as path
from mne.viz import Brain, get_3d_backend


def test_brain_init(backends_3d):
    backend_name = get_3d_backend()
    data_path = mne.datasets.sample.data_path()
    hemi = 'both'
    surf = 'inflated'
    subject_id = 'sample'
    subjects_dir = path.join(data_path, 'subjects')

    pytest.raises(ValueError, Brain, subject_id=subject_id,
                  hemi="split", surf=surf)
    pytest.raises(ValueError, Brain, subject_id=subject_id,
                  hemi=hemi, surf=surf, figure=0)
    pytest.raises(ValueError, Brain, subject_id=subject_id,
                  hemi=hemi, surf=surf, interaction=0)
    pytest.raises(KeyError, Brain, subject_id=subject_id,
                  hemi="foo", surf=surf)

    brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)
    if backend_name != "mayavi":
        brain.show()
