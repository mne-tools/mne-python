# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import mne
import pytest
from mne.datasets import sample
from mne.viz.backends.renderer import (set_3d_backend,
                                       get_3d_backend)


def test_3d_backend():
    pytest.raises(ValueError, set_3d_backend, "unknown_backend")
    pytest.raises(TypeError, set_3d_backend, 1)

    assert get_3d_backend() == "mayavi"

    # smoke test
    set_3d_backend('mayavi')
    set_3d_backend('mayavi')

    # example plot
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
    subjects_dir = data_path + '/subjects'
    subject = 'sample'
    trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
    info = mne.io.read_info(raw_fname)

    mne.viz.plot_alignment(info, trans, subject=subject, dig=True,
                           meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
                           surfaces='head-dense')
