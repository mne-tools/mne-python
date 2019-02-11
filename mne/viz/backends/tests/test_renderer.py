# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import mne
import pytest
import os.path as op
from mne.datasets import sample, testing
from mne.viz.backends.renderer import (set_3d_backend,
                                       get_3d_backend)


@testing.requires_testing_data
def test_3d_backend():
    """Test 3d backend degenerate scenarios and default plot."""
    pytest.raises(ValueError, set_3d_backend, "unknown_backend")
    pytest.raises(TypeError, set_3d_backend, 1)

    assert get_3d_backend() == "mayavi"

    # smoke test
    set_3d_backend('mayavi')
    set_3d_backend('mayavi')

    # example plot
    data_path = sample.data_path(download=False)
    raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
    subjects_dir = op.join(data_path, 'subjects')
    subject = 'sample'
    trans = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
    info = mne.io.read_info(raw_fname)

    mne.viz.plot_alignment(info, trans, subject=subject, dig=True,
                           meg=['helmet', 'sensors'],
                           subjects_dir=subjects_dir,
                           surfaces='head-dense')
