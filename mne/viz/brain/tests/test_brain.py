# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#
# License: Simplified BSD

import pytest
import numpy as np
import os.path as path
from mne import read_source_estimate
from mne.datasets import sample
from mne.viz import Brain, get_3d_backend
from mne.viz.backends._utils import backends3D
from mne.viz.brain.view import TimeViewer, ColorBar
from mne.viz.brain.colormap import _calculate_lut

data_path = sample.data_path()
subject_id = 'sample'
subjects_dir = path.join(data_path, 'subjects')
surf = 'inflated'


def test_brain_init(backends_3d):
    """Test initialization of the Brain instance."""
    backend_name = get_3d_backend()
    hemi = 'both'

    with pytest.raises(ValueError, match='hemi'):
        Brain(subject_id=subject_id, hemi="split", surf=surf)
    with pytest.raises(ValueError, match='figure'):
        Brain(subject_id=subject_id, hemi=hemi, surf=surf, figure=0)
    with pytest.raises(ValueError, match='interaction'):
        Brain(subject_id=subject_id, hemi=hemi, surf=surf, interaction=0)
    with pytest.raises(KeyError):
        Brain(subject_id=subject_id, hemi="foo", surf=surf)

    brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)
    if backend_name != backends3D.mayavi:
        brain.show()


def test_brain_add_data(backends_3d):
    """Test adding data in Brain instance."""
    backend_name = get_3d_backend()
    act_data = path.join(data_path, 'MEG/sample/sample_audvis-meg-eeg')

    stc = read_source_estimate(act_data)

    hemi = 'lh'
    hemi_data = stc.data[:len(stc.vertices[0]), 10]
    hemi_vertices = stc.vertices[0]
    fmin = stc.data.min()
    fmax = stc.data.max()

    brain_data = Brain(subject_id, hemi, surf, size=300,
                       subjects_dir=subjects_dir)

    with pytest.raises(ValueError):
        brain_data.add_data(array=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi,
                            fmax=fmax, vertices=None)

    brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi, fmax=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        colorbar=False)

    if backend_name != backends3D.mayavi:
        brain_data.show()


def test_brain_colormap():
    """Test brain's colormap functions."""
    from matplotlib import cm
    colormap = "coolwarm"
    alpha = 1.0
    fmin = 0.0
    fmid = 0.5
    fmax = 1.0
    center = None
    _calculate_lut(colormap, alpha=alpha, fmin=fmin,
                   fmid=fmid, fmax=fmax, center=center)
    center = 0.0
    colormap = cm.get_cmap(colormap)
    _calculate_lut(colormap, alpha=alpha, fmin=fmin,
                   fmid=fmid, fmax=fmax, center=center)


def test_brain_time_viewer(backends_3d):
    """Test of brain's time viewer."""
    backend_name = get_3d_backend()
    hemi = 'both'
    brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)

    with pytest.raises(KeyError):
        TimeViewer(brain)

    brain.data['time'] = None

    with pytest.raises(ValueError):
        TimeViewer(brain)

    # XXX: only the ipyvolume backend is supported for now
    if backend_name != backends3D.ipyvolume:
        pytest.skip('This feature is not available on {} yet.'
                    .format(backend_name))

    brain.data['time'] = np.zeros(1)
    brain.data['time_idx'] = np.zeros(1).astype(np.int)
    brain.data['time_label'] = "0"

    time_viewer = TimeViewer(brain)
    time_viewer.show()


def test_brain_colorbar(backends_3d):
    """Test of brain's colorbar."""
    from matplotlib import cm

    backend_name = get_3d_backend()
    hemi = 'both'
    brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)

    with pytest.raises(KeyError):
        ColorBar(brain)

    # XXX: only the ipyvolume backend is supported for now
    if backend_name != backends3D.ipyvolume:
        pytest.skip('This feature is not available on {} yet.'
                    .format(backend_name))

    brain.data['center'] = None
    brain.data['fmin'] = 0.0
    brain.data['fmid'] = 0.5
    brain.data['fmax'] = 1.0
    brain.data['lut'] = cm.get_cmap("coolwarm")

    color_bar = ColorBar(brain)
    color_bar.show()
