# -*- coding: utf-8 -*-
#
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
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
from mne.datasets import testing
from mne.viz._brain import _Brain
from mne.viz._brain.colormap import calculate_lut

from matplotlib import cm

data_path = testing.data_path(download=False)
subject_id = 'sample'
subjects_dir = path.join(data_path, 'subjects')
fname_stc = path.join(data_path, 'MEG/sample/sample_audvis_trunc-meg')
fname_label = path.join(data_path, 'MEG/sample/labels/Vis-lh.label')
surf = 'inflated'


@testing.requires_testing_data
def test_brain_init(renderer):
    """Test initialization of the _Brain instance."""
    hemi = 'both'

    with pytest.raises(ValueError, match='size'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, size=0.5)
    with pytest.raises(TypeError, match='figure'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, figure='foo')
    with pytest.raises(ValueError, match='interaction'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, interaction=0)
    with pytest.raises(KeyError):
        _Brain(subject_id=subject_id, hemi='foo', surf=surf)

    _Brain(subject_id, hemi, surf, size=(300, 300),
           subjects_dir=subjects_dir)


@testing.requires_testing_data
def test_brain_screenshot(renderer):
    """Test screenshot of a _Brain instance."""
    brain = _Brain(subject_id, hemi='both', size=600,
                   surf=surf, subjects_dir=subjects_dir)
    img = brain.screenshot(mode='rgb')
    assert(img.shape == (600, 600, 3))


@testing.requires_testing_data
def test_brain_add_data(renderer):
    """Test adding data in _Brain instance."""
    stc = read_source_estimate(fname_stc)

    hemi = 'lh'
    hemi_data = stc.data[:len(stc.vertices[0]), 10]
    hemi_vertices = stc.vertices[0]
    fmin = stc.data.min()
    fmax = stc.data.max()

    brain_data = _Brain(subject_id, hemi, surf, size=300,
                        subjects_dir=subjects_dir)

    with pytest.raises(ValueError, match='thresh'):
        brain_data.add_data(hemi_data, thresh=-1)
    with pytest.raises(ValueError, match='remove_existing'):
        brain_data.add_data(hemi_data, remove_existing=-1)
    with pytest.raises(ValueError, match='time_label_size'):
        brain_data.add_data(hemi_data, time_label_size=-1)
    with pytest.raises(ValueError, match='scale_factor'):
        brain_data.add_data(hemi_data, scale_factor=-1)
    with pytest.raises(ValueError, match='vector_alpha'):
        brain_data.add_data(hemi_data, vector_alpha=-1)
    with pytest.raises(ValueError):
        brain_data.add_data(array=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi,
                            fmax=fmax, vertices=None)

    brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi, fmax=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        colorbar=False, time=None)
    brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi, fmax=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        initial_time=0., colorbar=True, time=None)


@testing.requires_testing_data
def test_brain_add_label(renderer):
    """Test adding data in _Brain instance."""
    from mne.label import read_label
    brain = _Brain(subject_id, hemi='lh', size=500,
                   surf=surf, subjects_dir=subjects_dir)
    label = read_label(fname_label)
    brain.add_label(fname_label)
    brain.add_label(label, scalar_thresh=0.)


@testing.requires_testing_data
def test_brain_add_foci(renderer):
    """Test adding foci in _Brain instance."""
    brain = _Brain(subject_id, hemi='lh', size=500,
                   surf=surf, subjects_dir=subjects_dir)
    brain.add_foci([0], coords_as_verts=True,
                   hemi='lh', color='blue')


@testing.requires_testing_data
def test_brain_add_text(renderer):
    """Test adding text in _Brain instance."""
    brain = _Brain(subject_id, hemi='lh', size=250,
                   surf=surf, subjects_dir=subjects_dir)
    brain.add_text(x=0, y=0, text='foo')


def test_brain_colormap():
    """Test brain's colormap functions."""
    colormap = "coolwarm"
    alpha = 1.0
    fmin = 0.0
    fmid = 0.5
    fmax = 1.0
    center = None
    calculate_lut(colormap, alpha=alpha, fmin=fmin,
                  fmid=fmid, fmax=fmax, center=center)
    center = 0.0
    colormap = cm.get_cmap(colormap)
    calculate_lut(colormap, alpha=alpha, fmin=fmin,
                  fmid=fmid, fmax=fmax, center=center)
