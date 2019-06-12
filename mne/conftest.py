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
try:
    import readline  # noqa
except Exception:
    pass

import numpy as np
import mne
from mne.datasets import testing

test_path = testing.data_path(download=False)
s_path = op.join(test_path, 'MEG', 'sample')
fname_evoked = op.join(s_path, 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ('slowtest', 'ultraslowtest'):
        config.addinivalue_line('markers', marker)

    # Fixtures
    for fixture in ('matplotlib_config',):
        config.addinivalue_line('usefixtures', fixture)

    # Warnings
    # - Once SciPy updates not to have non-integer and non-tuple errors (1.2.0)
    #   we should remove them from here.
    # - This list should also be considered alongside reset_warnings in
    #   doc/conf.py.
    warning_lines = """
    error::
    ignore::ImportWarning
    ignore:the matrix subclass:PendingDeprecationWarning
    ignore:numpy.dtype size changed:RuntimeWarning
    ignore:.*HasTraits.trait_.*:DeprecationWarning
    ignore:.*takes no parameters:DeprecationWarning
    ignore:joblib not installed:RuntimeWarning
    ignore:Using a non-tuple sequence for multidimensional indexing:FutureWarning
    ignore:using a non-integer number instead of an integer will result in an error:DeprecationWarning
    ignore:Importing from numpy.testing.decorators is deprecated:DeprecationWarning
    ignore:np.loads is deprecated, use pickle.loads instead:DeprecationWarning
    ignore:The oldnumeric module will be dropped:DeprecationWarning
    ignore:Collection picker None could not be converted to float:UserWarning
    ignore:covariance is not positive-semidefinite:RuntimeWarning
    ignore:Can only plot ICA components:RuntimeWarning
    ignore:Matplotlib is building the font cache using fc-list:UserWarning
    ignore:Using or importing the ABCs from 'collections':DeprecationWarning
    ignore:`formatargspec` is deprecated:DeprecationWarning
    # This is only necessary until sklearn updates their wheels for NumPy 1.16
    ignore:numpy.ufunc size changed:RuntimeWarning
    ignore:.*mne-realtime.*:DeprecationWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    # "force" should not really be necessary but should not hurt
    with warnings.catch_warnings(record=True):
        matplotlib.use('agg', force=True)  # don't pop up windows
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


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def evoked():
    """Get evoked data."""
    evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                              baseline=(None, 0))
    evoked.crop(0, 0.2)
    return evoked


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def noise_cov():
    """Get a noise cov from the testing dataset."""
    return mne.read_cov(fname_cov)


@pytest.fixture(scope='function')
def bias_params_free(evoked, noise_cov):
    """Provide inputs for free bias functions."""
    fwd = mne.read_forward_solution(fname_fwd)
    return _bias_params(evoked, noise_cov, fwd)


@pytest.fixture(scope='function')
def bias_params_fixed(evoked, noise_cov):
    """Provide inputs for fixed bias functions."""
    fwd = mne.read_forward_solution(fname_fwd)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
    return _bias_params(evoked, noise_cov, fwd)


def _bias_params(evoked, noise_cov, fwd):
    evoked.pick_types(meg=True, eeg=True, exclude=())
    # restrict to limited set of verts (small src here) and one hemi for speed
    vertices = [fwd['src'][0]['vertno'].copy(), []]
    stc = mne.SourceEstimate(np.zeros((sum(len(v) for v in vertices), 1)),
                             vertices, 0., 1.)
    fwd = mne.forward.restrict_forward_to_stc(fwd, stc)
    assert fwd['sol']['row_names'] == noise_cov['names']
    assert noise_cov['names'] == evoked.ch_names
    evoked = mne.EvokedArray(fwd['sol']['data'].copy(), evoked.info)
    data_cov = noise_cov.copy()
    data_cov['data'] = np.dot(fwd['sol']['data'], fwd['sol']['data'].T)
    assert data_cov['data'].shape[0] == len(noise_cov['names'])
    want = np.arange(fwd['sol']['data'].shape[1])
    if not mne.forward.is_fixed_orient(fwd):
        want //= 3
    return evoked, fwd, noise_cov, data_cov, want


@pytest.fixture(scope="module", params=[
    "mayavi",
    "pyvista",
])
def backend_name(request):
    """Get the backend name."""
    yield request.param


@pytest.yield_fixture
def backends_3d(backend_name):
    """Yield the 3D backends."""
    from mne.viz.backends.renderer import _use_test_3d_backend
    from mne.viz.backends.tests._utils import has_mayavi, has_pyvista
    if backend_name == 'mayavi':
        if not has_mayavi():
            pytest.skip("Test skipped, requires mayavi.")
    elif backend_name == 'pyvista':
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")
    with _use_test_3d_backend(backend_name):
        yield
