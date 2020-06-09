# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from distutils.version import LooseVersion
import gc
import os
import os.path as op
import shutil
import sys
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
from mne.fixes import _fn35

test_path = testing.data_path(download=False)
s_path = op.join(test_path, 'MEG', 'sample')
fname_evoked = op.join(s_path, 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
subjects_dir = op.join(test_path, 'subjects')


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ('slowtest', 'ultraslowtest'):
        config.addinivalue_line('markers', marker)

    # Fixtures
    for fixture in ('matplotlib_config', 'fix_pytest_tmpdir_35'):
        config.addinivalue_line('usefixtures', fixture)

    # Warnings
    # - Once SciPy updates not to have non-integer and non-tuple errors (1.2.0)
    #   we should remove them from here.
    # - This list should also be considered alongside reset_warnings in
    #   doc/conf.py.
    warning_lines = r"""
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
    ignore:.*imp.*:DeprecationWarning
    ignore:Exception creating Regex for oneOf.*:SyntaxWarning
    ignore:scipy\.gradient is deprecated.*:DeprecationWarning
    ignore:sklearn\.externals\.joblib is deprecated.*:FutureWarning
    ignore:The sklearn.*module.*deprecated.*:FutureWarning
    ignore:.*trait.*handler.*deprecated.*:DeprecationWarning
    ignore:.*rich_compare.*metadata.*deprecated.*:DeprecationWarning
    ignore:.*In future, it will be an error for 'np.bool_'.*:DeprecationWarning
    ignore:.*Converting `np\.character` to a dtype is deprecated.*:DeprecationWarning
    ignore:.*sphinx\.util\.smartypants is deprecated.*:
    ignore:.*pandas\.util\.testing is deprecated.*:
    ignore:.*tostring.*is deprecated.*:DeprecationWarning
    always:.*get_data.* is deprecated in favor of.*:DeprecationWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


# Have to be careful with autouse=True, but this is just an int comparison
# so it shouldn't really add appreciable overhead
@pytest.fixture(autouse=True)
def check_verbose(request):
    """Set to the default logging level to ensure it's tested properly."""
    starting_level = mne.utils.logger.level
    yield
    # ensures that no tests break the global state
    try:
        assert mne.utils.logger.level == starting_level
    except AssertionError:
        pytest.fail('.'.join([request.module.__name__,
                              request.function.__name__]) +
                    ' modifies logger.level')


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    from matplotlib import cbook
    # "force" should not really be necessary but should not hurt
    kwargs = dict()
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings('ignore')
        matplotlib.use('agg', force=True, **kwargs)  # don't pop up windows
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

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None):
            args = ()
            if LooseVersion(matplotlib.__version__) >= LooseVersion('2.1'):
                args += (exception_handler,)
            super(CallbackRegistryReraise, self).__init__(*args)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture()
def check_gui_ci():
    """Skip tests that are not reliable on CIs."""
    osx = (os.getenv('TRAVIS', 'false').lower() == 'true' and
           sys.platform == 'darwin')
    win = os.getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true'
    if win or osx:
        pytest.skip('Skipping GUI tests on Travis OSX and Azure Windows')


def _replace(mod, key):
    orig = getattr(mod, key)

    def func(x, *args, **kwargs):
        return orig(_fn35(x), *args, **kwargs)

    setattr(mod, key, func)


@pytest.fixture(scope='session')
def fix_pytest_tmpdir_35():
    """Deal with tmpdir being a LocalPath, which bombs on 3.5."""
    if sys.version_info >= (3, 6):
        return

    for key in ('stat', 'mkdir', 'makedirs', 'access'):
        _replace(os, key)
    for key in ('split', 'splitext', 'realpath', 'join', 'basename'):
        _replace(op, key)


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
def renderer(backend_name, garbage_collect):
    """Yield the 3D backends."""
    from mne.viz.backends.renderer import _use_test_3d_backend
    _check_skip_backend(backend_name)
    with _use_test_3d_backend(backend_name):
        from mne.viz.backends import renderer
        yield renderer
        renderer.backend._close_all()


@pytest.yield_fixture
def garbage_collect():
    """Garbage collect on exit."""
    yield
    gc.collect()


@pytest.fixture(scope="module", params=[
    "pyvista",
    "mayavi",
])
def backend_name_interactive(request):
    """Get the backend name."""
    yield request.param


@pytest.yield_fixture
def renderer_interactive(backend_name_interactive):
    """Yield the 3D backends."""
    from mne.viz.backends.renderer import _use_test_3d_backend
    _check_skip_backend(backend_name_interactive)
    with _use_test_3d_backend(backend_name_interactive, interactive=True):
        from mne.viz.backends import renderer
        yield renderer
        renderer.backend._close_all()


def _check_skip_backend(name):
    from mne.viz.backends.tests._utils import has_mayavi, has_pyvista
    if name == 'mayavi':
        if not has_mayavi():
            pytest.skip("Test skipped, requires mayavi.")
    elif name == 'pyvista':
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def subjects_dir_tmp(tmpdir):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ('sample', 'fsaverage'):
        shutil.copytree(op.join(subjects_dir, key), str(tmpdir.join(key)))
    return str(tmpdir)
