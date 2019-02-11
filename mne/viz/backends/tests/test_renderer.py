# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import pytest
import os.path as op
import numpy as np

from mne import SourceEstimate
from mne.datasets import testing
from mne.source_space import read_source_spaces
from mne.utils import requires_mayavi
from mne.viz import plot_sparse_source_estimates
from mne.viz.backends.renderer import (set_3d_backend,
                                       get_3d_backend)

data_dir = testing.data_path(download=False)
src_fname = op.join(data_dir, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')


@testing.requires_testing_data
@requires_mayavi
def test_3d_backend():
    """Test 3d backend degenerate scenarios and default plot."""
    pytest.raises(ValueError, set_3d_backend, "unknown_backend")
    pytest.raises(TypeError, set_3d_backend, 1)

    assert get_3d_backend() == "mayavi"

    # smoke test
    set_3d_backend('mayavi')
    set_3d_backend('mayavi')

    # example plot
    n_time = 5
    sample_src = read_source_spaces(src_fname)
    vertices = sample_src[0]['vertno']
    inds = [111, 333]
    stc_data = np.zeros((len(inds), n_time))
    stc_data[0, 1] = 1.
    stc_data[1, 4] = 2.
    vertices = [vertices[inds], np.empty(0, dtype=np.int)]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_sparse_source_estimates(sample_src, stc, bgcolor=(1, 1, 1),
                                 opacity=0.5, high_resolution=False)
