import pytest
import numpy as np
import os.path as path
from mne import read_source_estimate
from mne.datasets import sample
from mne.viz import Brain, get_3d_backend
from mne.viz.brain.view import TimeViewer
from mne.viz.brain.colormap import _calculate_lut


def test_brain_init(backends_3d):
    """Test initialization of the Brain instance."""
    backend_name = get_3d_backend()
    data_path = sample.data_path()
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
    """Test adding data in Brain instance."""
    backend_name = get_3d_backend()
    data_path = sample.data_path()
    act_data = path.join(data_path, 'MEG/sample/sample_audvis-meg-eeg')

    stc = read_source_estimate(act_data)

    hemi = 'lh'
    hemi_data = stc.data[:len(stc.vertices[0]), 10]
    hemi_vertices = stc.vertices[0]

    fmin = stc.data.min()
    fmax = stc.data.max()
    surf = 'inflated'
    subject_id = 'sample'
    subjects_dir = path.join(data_path, 'subjects')

    brain_data = Brain(subject_id, hemi, surf, size=300,
                       subjects_dir=subjects_dir)

    with pytest.raises(ValueError):
        brain_data.add_data(array=np.zeros(3))
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, thresh=0)
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, transparent=0)
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, remove_existing=0)
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, time_label_size=0)
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, scale_factor=0)
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, vector_alpha=0)
    with pytest.raises(ValueError):
        brain_data.add_data(array=hemi_data, verbose=0)
    with pytest.raises(ValueError):
        brain_data.add_data(hemi_data, min=fmin, hemi=hemi,
                            max=fmax, vertices=None)

    brain_data.add_data(hemi_data, min=fmin, hemi=hemi, max=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        colorbar=False)

    if backend_name != "mayavi":
        brain_data.show()


def test_brain_colormap():
    """Test brain's colormap functions."""
    from matplotlib import cm
    colormap = "coolwarm"
    alpha = 1.0
    min = 0.0
    mid = 0.5
    max = 1.0
    center = None
    _calculate_lut(colormap, alpha=alpha, fmin=min,
                   fmid=mid, fmax=max, center=center)
    center = 0.0
    colormap = cm.get_cmap(colormap)
    _calculate_lut(colormap, alpha=alpha, fmin=min,
                   fmid=mid, fmax=max, center=center)


def test_brain_time_viewer(backends_3d):
    """Test of brain's time viewer."""
    backend_name = get_3d_backend()
    backend_name = get_3d_backend()
    data_path = sample.data_path()
    hemi = 'both'
    surf = 'inflated'
    subject_id = 'sample'
    subjects_dir = path.join(data_path, 'subjects')
    brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)

    with pytest.raises(KeyError):
        TimeViewer(brain)

    brain.data['time'] = None

    with pytest.raises(ValueError):
        TimeViewer(brain)

    if backend_name != "ipyvolume":
        pytest.skip()

    brain.data['time'] = np.zeros(1)
    brain.data['time_idx'] = np.zeros(1).astype(np.int)
    brain.data['time_label'] = "0"

    time_viewer = TimeViewer(brain)
    time_viewer.show()
