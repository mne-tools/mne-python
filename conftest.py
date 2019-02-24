# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import pytest
# For some unknown reason, on Travis-xenial there are segfaults caused on
# the line pytest -> pdb.Pdb.__init__ -> "import readline". Forcing an
# import here seems to prevent them (!?). This suggests a potential problem
# with some other library stepping on memory where it shouldn't. It only
# seems to happen on the Linux runs that install Mayavi. Anectodally,
# @larsoner has had problems a couple of years ago where a mayavi import
# seemed to corrupt SciPy linalg function results (!), likely due to the
# associated VTK import, so this could be another manifestation of that.
import readline  # noqa

import numpy as np
import mne
from mne.datasets import testing

test_path = testing.data_path(download=False)
s_path = op.join(test_path, 'MEG', 'sample')
fname_data = op.join(s_path, 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
# turn anything that uses testing data into an auto-skipper by
# setting params=[testing_param]
testing_param = pytest.mark.skipif(testing._testing._skip_testing_data(),
                                   reason='Requires testing dataset')('x')


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    matplotlib.use('agg')  # don't pop up windows
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
    try:
        from traits.etsconfig.api import ETSConfig
    except Exception:
        pass
    else:
        ETSConfig.toolkit = 'qt4'
    try:
        with warnings.catch_warnings(record=True):  # traits
            from mayavi import mlab
    except Exception:
        pass
    else:
        mlab.options.backend = 'test'


@pytest.fixture(scope='function', params=[testing_param])
def evoked():
    """Get evoked data."""
    evoked = mne.read_evokeds(fname_data, condition='Left Auditory',
                              baseline=(None, 0))
    evoked.crop(0, 0.2)
    return evoked


@pytest.fixture(scope='function', params=[testing_param])
def noise_cov():
    return mne.read_cov(fname_cov)


@pytest.fixture(scope='function')
def bias_params(evoked, noise_cov):
    """Provide inputs for bias functions."""
    # Identity input
    evoked.pick_types(meg=True, eeg=True, exclude=())
    evoked = mne.EvokedArray(np.eye(len(evoked.data)), evoked.info)
    # restrict to limited set of verts (small src here) and one hemi for speed
    fwd_orig = mne.read_forward_solution(fname_fwd)
    vertices = [fwd_orig['src'][0]['vertno'].copy(), []]
    stc = mne.SourceEstimate(np.zeros((sum(len(v) for v in vertices), 1)),
                             vertices, 0., 1.)
    fwd_orig = mne.forward.restrict_forward_to_stc(fwd_orig, stc)
    return evoked, fwd_orig, noise_cov
