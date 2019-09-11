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
from mne.viz._brain.colormap import _calculate_lut

from matplotlib import cm

data_path = testing.data_path(download=False)
subject_id = 'sample'
subjects_dir = path.join(data_path, 'subjects')
fname_stc = path.join(data_path, 'MEG/sample/sample_audvis_trunc-meg')
surf = 'inflated'


@testing.requires_testing_data
def test_brain_init(renderer):
    """Test initialization of the _Brain instance."""
    backend_name = renderer.get_3d_backend()
    hemi = 'both'

    with pytest.raises(ValueError, match='hemi'):
        _Brain(subject_id=subject_id, hemi="split", surf=surf)
    with pytest.raises(ValueError, match='figure'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, figure=0)
    with pytest.raises(ValueError, match='interaction'):
        _Brain(subject_id=subject_id, hemi=hemi, surf=surf, interaction=0)
    with pytest.raises(KeyError):
        _Brain(subject_id=subject_id, hemi="foo", surf=surf)

    brain = _Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)
    if backend_name != 'mayavi':
        brain.show()


@testing.requires_testing_data
def test_brain_add_data(renderer):
    """Test adding data in _Brain instance."""
    backend_name = renderer.get_3d_backend()
    stc = read_source_estimate(fname_stc)

    hemi = 'lh'
    hemi_data = stc.data[:len(stc.vertices[0]), 10]
    hemi_vertices = stc.vertices[0]
    fmin = stc.data.min()
    fmax = stc.data.max()

    brain_data = _Brain(subject_id, hemi, surf, size=300,
                        subjects_dir=subjects_dir)

    with pytest.raises(ValueError):
        brain_data.add_data(array=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi,
                            fmax=fmax, vertices=None)

    brain_data.add_data(hemi_data, fmin=fmin, hemi=hemi, fmax=fmax,
                        colormap='hot', vertices=hemi_vertices,
                        colorbar=False)

    if backend_name != 'mayavi':
        brain_data.show()


def test_brain_colormap():
    """Test brain's colormap functions."""
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
