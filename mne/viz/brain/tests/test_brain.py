import mne
import pytest
import numpy as np
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


def test_brain_add_data(backends_3d):
    backend_name = get_3d_backend()
    data_path = mne.datasets.sample.data_path()
    act_data = path.join(data_path, 'MEG/sample/sample_audvis-meg-eeg')

    stc = mne.read_source_estimate(act_data)

    hemi = 'lh'
    hemi_data = stc.data[:len(stc.vertices[0]), 10]
    hemi_vertices = stc.vertices[0]

    fmin = stc.data.min()
    fmax = stc.data.max()
    data_path = mne.datasets.sample.data_path()
    surf = 'inflated'
    subject_id = 'sample'
    subjects_dir = path.join(data_path, 'subjects')

    brain_data = Brain(subject_id, hemi, surf, size=300,
                       subjects_dir=subjects_dir)

    with pytest.raises(ValueError):
        brain_data.add_data(array=np.zeros(3))
        brain_data.add_data(array=hemi_data, thresh=0)
        brain_data.add_data(array=hemi_data, transparent=0)
        brain_data.add_data(array=hemi_data, remove_existing=0)
        brain_data.add_data(array=hemi_data, time_label_size=0)
        brain_data.add_data(array=hemi_data, scale_factor=0)
        brain_data.add_data(array=hemi_data, vector_alpha=0)
        brain_data.add_data(array=hemi_data, verbose=0)
        brain_data.add_data(hemi_data, min=fmin, hemi=hemi,
                            max=fmax, vertices=None)

    brain_data.add_data(hemi_data, min=fmin, hemi=hemi, max=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        colorbar=False)

    if backend_name != "mayavi":
        brain_data.show()
