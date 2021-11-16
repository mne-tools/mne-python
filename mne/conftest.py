# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from contextlib import contextmanager
from distutils.version import LooseVersion
import inspect
from textwrap import dedent
import gc
import os
import os.path as op
from pathlib import Path
import shutil
import sys
import warnings
import pytest

import numpy as np

import mne
from mne import read_events, pick_types, Epochs
from mne.channels import read_layout
from mne.datasets import testing
from mne.fixes import has_numba
from mne.io import read_raw_fif, read_raw_ctf
from mne.stats import cluster_level
from mne.utils import (_pl, _assert_no_instances, numerics, Bunch,
                       _check_pyqt5_version)

# data from sample dataset
from mne.viz._figure import use_browser_backend

test_path = testing.data_path(download=False)
s_path = op.join(test_path, 'MEG', 'sample')
fname_evoked = op.join(s_path, 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_full = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
bem_path = op.join(test_path, 'subjects', 'sample', 'bem')
fname_bem = op.join(bem_path, 'sample-1280-bem.fif')
fname_aseg = op.join(test_path, 'subjects', 'sample', 'mri', 'aseg.mgz')
subjects_dir = op.join(test_path, 'subjects')
fname_src = op.join(bem_path, 'sample-oct-4-src.fif')
fname_trans = op.join(s_path, 'sample_audvis_trunc-trans.fif')

ctf_dir = op.join(test_path, 'CTF')
fname_ctf_continuous = op.join(ctf_dir, 'testdata_ctf.ds')

# data from mne.io.tests.data
base_dir = op.join(op.dirname(__file__), 'io', 'tests', 'data')
fname_raw_io = op.join(base_dir, 'test_raw.fif')
fname_event_io = op.join(base_dir, 'test-eve.fif')
fname_cov_io = op.join(base_dir, 'test-cov.fif')
fname_evoked_io = op.join(base_dir, 'test-ave.fif')
event_id, tmin, tmax = 1, -0.1, 1.0
vv_layout = read_layout('Vectorview-all')

collect_ignore = ['export/_eeglab.py', 'export/_edf.py']


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ('slowtest', 'ultraslowtest', 'pgtest'):
        config.addinivalue_line('markers', marker)

    # Fixtures
    for fixture in ('matplotlib_config', 'close_all', 'check_verbose'):
        config.addinivalue_line('usefixtures', fixture)

    # Warnings
    # - Once SciPy updates not to have non-integer and non-tuple errors (1.2.0)
    #   we should remove them from here.
    # - This list should also be considered alongside reset_warnings in
    #   doc/conf.py.
    warning_lines = r"""
    error::
    ignore:.*deprecated and ignored since IPython.*:DeprecationWarning
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
    ignore:.*`np.bool` is a deprecated alias.*:DeprecationWarning
    ignore:.*`np.int` is a deprecated alias.*:DeprecationWarning
    ignore:.*`np.float` is a deprecated alias.*:DeprecationWarning
    ignore:.*`np.object` is a deprecated alias.*:DeprecationWarning
    ignore:.*`np.long` is a deprecated alias:DeprecationWarning
    ignore:.*Converting `np\.character` to a dtype is deprecated.*:DeprecationWarning
    ignore:.*sphinx\.util\.smartypants is deprecated.*:
    ignore:.*pandas\.util\.testing is deprecated.*:
    ignore:.*tostring.*is deprecated.*:DeprecationWarning
    ignore:.*QDesktopWidget\.availableGeometry.*:DeprecationWarning
    ignore:Unable to enable faulthandler.*:UserWarning
    ignore:Fetchers from the nilearn.*:FutureWarning
    ignore:SelectableGroups dict interface is deprecated\. Use select\.:DeprecationWarning
    ignore:Call to deprecated class vtk.*:DeprecationWarning
    ignore:Call to deprecated method.*Deprecated since.*:DeprecationWarning
    always:.*get_data.* is deprecated in favor of.*:DeprecationWarning
    ignore:.*rcParams is deprecated.*global_theme.*:DeprecationWarning
    ignore:.*distutils\.sysconfig module is deprecated.*:DeprecationWarning
    ignore:.*moved to a new package \(mne-connectivity\).*:DeprecationWarning
    ignore:.*numpy\.dual is deprecated.*:DeprecationWarning
    ignore:.*`np.typeDict` is a deprecated.*:DeprecationWarning
    ignore:.*Creating an ndarray from ragged.*:numpy.VisibleDeprecationWarning
    ignore:^Please use.*scipy\..*:DeprecationWarning
    ignore:.*Passing a schema to Validator.*:DeprecationWarning
    ignore:.*Found the following unknown channel type.*:RuntimeWarning
    ignore:.*in an Any trait will be shared.*:DeprecationWarning
    ignore:.*Mayavi 3D backend is deprecated.*:DeprecationWarning
    ignore:.*np\.MachAr.*:DeprecationWarning
    always::ResourceWarning
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


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture(autouse=True)
def add_mne(doctest_namespace):
    """Add mne to the namespace."""
    doctest_namespace["mne"] = mne


@pytest.fixture(scope='function')
def verbose_debug():
    """Run a test with debug verbosity."""
    with mne.utils.use_log_level('debug'):
        yield


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    from matplotlib import cbook
    # Allow for easy interactive debugging with a call like:
    #
    #     $ MNE_MPL_TESTING_BACKEND=Qt5Agg pytest mne/viz/tests/test_raw.py -k annotation -x --pdb  # noqa: E501
    #
    try:
        want = os.environ['MNE_MPL_TESTING_BACKEND']
    except KeyError:
        want = 'agg'  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings('ignore')
        matplotlib.use(want, force=True)
    import matplotlib.pyplot as plt
    assert plt.get_backend() == want
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
        try:
            ETSConfig.toolkit = 'qt4'
        except Exception:
            pass  # 'null' might be the only option in some configs

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None):
            args = ()
            if LooseVersion(matplotlib.__version__) >= LooseVersion('2.1'):
                args += (exception_handler,)
            super(CallbackRegistryReraise, self).__init__(*args)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture(scope='session')
def ci_macos():
    """Determine if running on MacOS CI."""
    return (os.getenv('CI', 'false').lower() == 'true' and
            sys.platform == 'darwin')


@pytest.fixture(scope='session')
def azure_windows():
    """Determine if running on Azure Windows."""
    return (os.getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true' and
            sys.platform.startswith('win'))


@pytest.fixture()
def check_gui_ci(ci_macos, azure_windows):
    """Skip tests that are not reliable on CIs."""
    if azure_windows or ci_macos:
        pytest.skip('Skipping GUI tests on MacOS CIs and Azure Windows')


@pytest.fixture(scope='function')
def raw_orig():
    """Get raw data without any change to it from mne.io.tests.data."""
    raw = read_raw_fif(fname_raw_io, preload=True)
    return raw


@pytest.fixture(scope='function')
def raw():
    """
    Get raw data and pick channels to reduce load for testing.

    (from mne.io.tests.data)
    """
    raw = read_raw_fif(fname_raw_io, preload=True)
    # Throws a warning about a changed unit.
    with pytest.warns(RuntimeWarning, match='unit'):
        raw.set_channel_types({raw.ch_names[0]: 'ias'})
    raw.pick_channels(raw.ch_names[:9])
    raw.info.normalize_proj()  # Fix projectors after subselection
    return raw


@pytest.fixture(scope='function')
def raw_ctf():
    """Get ctf raw data from mne.io.tests.data."""
    raw_ctf = read_raw_ctf(fname_ctf_continuous, preload=True)
    return raw_ctf


@pytest.fixture(scope='function')
def events():
    """Get events from mne.io.tests.data."""
    return read_events(fname_event_io)


def _get_epochs(stop=5, meg=True, eeg=False, n_chan=20):
    """Get epochs."""
    raw = read_raw_fif(fname_raw_io)
    events = read_events(fname_event_io)
    picks = pick_types(raw.info, meg=meg, eeg=eeg, stim=False,
                       ecg=False, eog=False, exclude='bads')
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs = Epochs(raw, events[:stop], event_id, tmin, tmax, picks=picks,
                        proj=False, preload=False)
    epochs.info.normalize_proj()  # avoid warnings
    return epochs


@pytest.fixture()
def epochs():
    """
    Get minimal, pre-loaded epochs data suitable for most tests.

    (from mne.io.tests.data)
    """
    return _get_epochs().load_data()


@pytest.fixture()
def epochs_unloaded():
    """Get minimal, unloaded epochs data from mne.io.tests.data."""
    return _get_epochs()


@pytest.fixture()
def epochs_full():
    """Get full, preloaded epochs from mne.io.tests.data."""
    return _get_epochs(None).load_data()


@pytest.fixture(scope='session', params=[testing._pytest_param()])
def _evoked():
    # This one is session scoped, so be sure not to modify it (use evoked
    # instead)
    evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                              baseline=(None, 0))
    evoked.crop(0, 0.2)
    return evoked


@pytest.fixture()
def evoked(_evoked):
    """Get evoked data."""
    return _evoked.copy()


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def noise_cov():
    """Get a noise cov from the testing dataset."""
    return mne.read_cov(fname_cov)


@pytest.fixture
def noise_cov_io():
    """Get noise-covariance (from mne.io.tests.data)."""
    return mne.read_cov(fname_cov_io)


@pytest.fixture(scope='function')
def bias_params_free(evoked, noise_cov):
    """Provide inputs for free bias functions."""
    fwd = mne.read_forward_solution(fname_fwd)
    return _bias_params(evoked, noise_cov, fwd)


@pytest.fixture(scope='function')
def bias_params_fixed(evoked, noise_cov):
    """Provide inputs for fixed bias functions."""
    fwd = mne.read_forward_solution(fname_fwd)
    mne.convert_forward_solution(
        fwd, force_fixed=True, surf_ori=True, copy=False)
    return _bias_params(evoked, noise_cov, fwd)


def _bias_params(evoked, noise_cov, fwd):
    evoked.pick_types(meg=True, eeg=True, exclude=())
    # restrict to limited set of verts (small src here) and one hemi for speed
    vertices = [fwd['src'][0]['vertno'].copy(), []]
    stc = mne.SourceEstimate(
        np.zeros((sum(len(v) for v in vertices), 1)), vertices, 0, 1)
    fwd = mne.forward.restrict_forward_to_stc(fwd, stc)
    assert fwd['sol']['row_names'] == noise_cov['names']
    assert noise_cov['names'] == evoked.ch_names
    evoked = mne.EvokedArray(fwd['sol']['data'].copy(), evoked.info)
    data_cov = noise_cov.copy()
    data = fwd['sol']['data'] @ fwd['sol']['data'].T
    data *= 1e-14  # 100 nAm at each source, effectively (1e-18 would be 1 nAm)
    # This is rank-deficient, so let's make it actually positive semidefinite
    # by regularizing a tiny bit
    data.flat[::data.shape[0] + 1] += mne.make_ad_hoc_cov(evoked.info)['data']
    # Do our projection
    proj, _, _ = mne.io.proj.make_projector(
        data_cov['projs'], data_cov['names'])
    data = proj @ data @ proj.T
    data_cov['data'][:] = data
    assert data_cov['data'].shape[0] == len(noise_cov['names'])
    want = np.arange(fwd['sol']['data'].shape[1])
    if not mne.forward.is_fixed_orient(fwd):
        want //= 3
    return evoked, fwd, noise_cov, data_cov, want


@pytest.fixture
def garbage_collect():
    """Garbage collect on exit."""
    yield
    gc.collect()


@pytest.fixture
def mpl_backend(garbage_collect):
    """Use for epochs/ica when not implemented with pyqtgraph yet."""
    with use_browser_backend('matplotlib') as backend:
        yield backend
        backend._close_all()


def _check_pyqtgraph():
    try:
        import PyQt5  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip('PyQt5 is not installed but needed for pyqtgraph!')
    try:
        assert LooseVersion(_check_pyqt5_version()) >= LooseVersion('5.12')
    except AssertionError:
        pytest.skip(f'PyQt5 has version {_check_pyqt5_version()}'
                    f'but pyqtgraph needs >= 5.12!')
    try:
        import mne_qt_browser  # noqa: F401
    except Exception:
        pytest.skip('Requires mne_qt_browser')


@pytest.mark.pgtest
@pytest.fixture
def pg_backend(garbage_collect):
    """Use for pyqtgraph-specific test-functions."""
    _check_pyqtgraph()
    with use_browser_backend('pyqtgraph') as backend:
        yield backend
        backend._close_all()


@pytest.fixture(params=[
    'matplotlib',
    pytest.param('pyqtgraph', marks=pytest.mark.pgtest),
])
def browser_backend(request, garbage_collect):
    """Parametrizes the name of the browser backend."""
    backend_name = request.param
    if backend_name == 'pyqtgraph':
        _check_pyqtgraph()
    with use_browser_backend(backend_name) as backend:
        yield backend
        backend._close_all()


@pytest.fixture(params=["mayavi", "pyvistaqt"])
def renderer(request, garbage_collect):
    """Yield the 3D backends."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["pyvistaqt"])
def renderer_pyvistaqt(request, garbage_collect):
    """Yield the PyVista backend."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["mayavi"])
def renderer_mayavi(request, garbage_collect):
    """Yield the mayavi backend."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["notebook"])
def renderer_notebook(request):
    """Yield the 3D notebook renderer."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(scope="module", params=["pyvistaqt"])
def renderer_interactive_pyvistaqt(request):
    """Yield the interactive PyVista backend."""
    with _use_backend(request.param, interactive=True) as renderer:
        yield renderer


@pytest.fixture(scope="module", params=["pyvistaqt", "mayavi"])
def renderer_interactive(request):
    """Yield the interactive 3D backends."""
    with _use_backend(request.param, interactive=True) as renderer:
        if renderer._get_3d_backend() == 'mayavi':
            with warnings.catch_warnings(record=True):
                try:
                    from surfer import Brain  # noqa: F401 analysis:ignore
                except Exception:
                    pytest.skip('Requires PySurfer')
        yield renderer


@contextmanager
def _use_backend(backend_name, interactive):
    from mne.viz.backends.renderer import _use_test_3d_backend
    _check_skip_backend(backend_name)
    with _use_test_3d_backend(backend_name, interactive=interactive):
        from mne.viz.backends import renderer
        try:
            yield renderer
        finally:
            renderer.backend._close_all()


def _check_skip_backend(name):
    from mne.viz.backends.tests._utils import (has_mayavi, has_pyvista,
                                               has_pyqt5, has_imageio_ffmpeg,
                                               has_pyvistaqt)
    if name in ('pyvistaqt', 'notebook'):
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")
        if not has_imageio_ffmpeg():
            pytest.skip("Test skipped, requires imageio-ffmpeg")
    if name in ('pyvistaqt', 'mayavi') and not has_pyqt5():
        pytest.skip("Test skipped, requires PyQt5.")
    if name == 'mayavi' and not has_mayavi():
        pytest.skip("Test skipped, requires mayavi.")
    if name == 'pyvistaqt' and not has_pyvistaqt():
        pytest.skip("Test skipped, requires pyvistaqt")


@pytest.fixture(scope='session')
def pixel_ratio():
    """Get the pixel ratio."""
    from mne.viz.backends.tests._utils import (has_mayavi, has_pyvista,
                                               has_pyqt5)
    if not (has_mayavi() or has_pyvista()) or not has_pyqt5():
        return 1.
    from PyQt5.QtWidgets import QApplication, QMainWindow
    _ = QApplication.instance() or QApplication([])
    window = QMainWindow()
    ratio = float(window.devicePixelRatio())
    window.close()
    return ratio


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def subjects_dir_tmp(tmp_path):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ('sample', 'fsaverage'):
        shutil.copytree(op.join(subjects_dir, key), str(tmp_path / key))
    return str(tmp_path)


# Scoping these as session will make things faster, but need to make sure
# not to modify them in-place in the tests, so keep them private
@pytest.fixture(scope='session', params=[testing._pytest_param()])
def _evoked_cov_sphere(_evoked):
    """Compute a small evoked/cov/sphere combo for use with forwards."""
    evoked = _evoked.copy().pick_types(meg=True)
    evoked.pick_channels(evoked.ch_names[::4])
    assert len(evoked.ch_names) == 77
    cov = mne.read_cov(fname_cov)
    sphere = mne.make_sphere_model('auto', 'auto', evoked.info)
    return evoked, cov, sphere


@pytest.fixture(scope='session')
def _fwd_surf(_evoked_cov_sphere):
    """Compute a forward for a surface source space."""
    evoked, cov, sphere = _evoked_cov_sphere
    src_surf = mne.read_source_spaces(fname_src)
    return mne.make_forward_solution(
        evoked.info, fname_trans, src_surf, sphere, mindist=5.0)


@pytest.fixture(scope='session')
def _fwd_subvolume(_evoked_cov_sphere):
    """Compute a forward for a surface source space."""
    pytest.importorskip('nibabel')
    evoked, cov, sphere = _evoked_cov_sphere
    volume_labels = ['Left-Cerebellum-Cortex', 'right-Cerebellum-Cortex']
    with pytest.raises(ValueError,
                       match=r"Did you mean one of \['Right-Cere"):
        mne.setup_volume_source_space(
            'sample', pos=20., volume_label=volume_labels,
            subjects_dir=subjects_dir)
    volume_labels[1] = 'R' + volume_labels[1][1:]
    src_vol = mne.setup_volume_source_space(
        'sample', pos=20., volume_label=volume_labels,
        subjects_dir=subjects_dir, add_interpolator=False)
    return mne.make_forward_solution(
        evoked.info, fname_trans, src_vol, sphere, mindist=5.0)


@pytest.fixture(scope='session')
def _all_src_types_fwd(_fwd_surf, _fwd_subvolume):
    """Create all three forward types (surf, vol, mixed)."""
    fwds = dict(surface=_fwd_surf, volume=_fwd_subvolume)
    with pytest.raises(RuntimeError,
                       match='Invalid source space with kinds'):
        fwds['volume']['src'] + fwds['surface']['src']

    # mixed (4)
    fwd = fwds['surface'].copy()
    f2 = fwds['volume']
    for keys, axis in [(('source_rr',), 0),
                       (('source_nn',), 0),
                       (('sol', 'data'), 1),
                       (('_orig_sol',), 1)]:
        a, b = fwd, f2
        key = keys[0]
        if len(keys) > 1:
            a, b = a[key], b[key]
            key = keys[1]
        a[key] = np.concatenate([a[key], b[key]], axis=axis)
    fwd['sol']['ncol'] = fwd['sol']['data'].shape[1]
    fwd['nsource'] = fwd['sol']['ncol'] // 3
    fwd['src'] = fwd['src'] + f2['src']
    fwds['mixed'] = fwd

    return fwds


@pytest.fixture(scope='session')
def _all_src_types_inv_evoked(_evoked_cov_sphere, _all_src_types_fwd):
    """Compute inverses for all source types."""
    evoked, cov, _ = _evoked_cov_sphere
    invs = dict()
    for kind, fwd in _all_src_types_fwd.items():
        assert fwd['src'].kind == kind
        with pytest.warns(RuntimeWarning, match='has magnitude'):
            invs[kind] = mne.minimum_norm.make_inverse_operator(
                evoked.info, fwd, cov)
    return invs, evoked


@pytest.fixture(scope='function')
def all_src_types_inv_evoked(_all_src_types_inv_evoked):
    """All source types of inverses, allowing for possible modification."""
    invs, evoked = _all_src_types_inv_evoked
    invs = {key: val.copy() for key, val in invs.items()}
    evoked = evoked.copy()
    return invs, evoked


@pytest.fixture(scope='function')
def mixed_fwd_cov_evoked(_evoked_cov_sphere, _all_src_types_fwd):
    """Compute inverses for all source types."""
    evoked, cov, _ = _evoked_cov_sphere
    return _all_src_types_fwd['mixed'].copy(), cov.copy(), evoked.copy()


@pytest.fixture(scope='session')
@pytest.mark.slowtest
@pytest.mark.parametrize(params=[testing._pytest_param()])
def src_volume_labels():
    """Create a 7mm source space with labels."""
    pytest.importorskip('nibabel')
    volume_labels = mne.get_volume_labels_from_aseg(fname_aseg)
    with pytest.warns(RuntimeWarning, match='Found no usable.*Left-vessel.*'):
        src = mne.setup_volume_source_space(
            'sample', 7., mri='aseg.mgz', volume_label=volume_labels,
            add_interpolator=False, bem=fname_bem,
            subjects_dir=subjects_dir)
    lut, _ = mne.read_freesurfer_lut()
    assert len(volume_labels) == 46
    assert volume_labels[0] == 'Unknown'
    assert lut['Unknown'] == 0  # it will be excluded during label gen
    return src, tuple(volume_labels), lut


def _fail(*args, **kwargs):
    __tracebackhide__ = True
    raise AssertionError('Test should not download')


@pytest.fixture(scope='function')
def download_is_error(monkeypatch):
    """Prevent downloading by raising an error when it's attempted."""
    import pooch
    monkeypatch.setattr(pooch, 'retrieve', _fail)


@pytest.fixture()
def brain_gc(request):
    """Ensure that brain can be properly garbage collected."""
    keys = (
        'renderer_interactive',
        'renderer_interactive_pyvistaqt',
        'renderer_interactive_pysurfer',
        'renderer',
        'renderer_pyvistaqt',
        'renderer_notebook',
    )
    assert set(request.fixturenames) & set(keys) != set()
    for key in keys:
        if key in request.fixturenames:
            is_pv = \
                request.getfixturevalue(key)._get_3d_backend() == 'pyvistaqt'
            close_func = request.getfixturevalue(key).backend._close_all
            break
    if not is_pv:
        yield
        return
    import pyvista
    if LooseVersion(pyvista.__version__) <= LooseVersion('0.26.1'):
        yield
        return
    from mne.viz import Brain
    ignore = set(id(o) for o in gc.get_objects())
    yield
    close_func()
    # no need to warn if the test itself failed, pytest-harvest helps us here
    try:
        outcome = request.node.harvest_rep_call
    except Exception:
        outcome = 'failed'
    if outcome != 'passed':
        return
    _assert_no_instances(Brain, 'after')
    # We only check VTK for PyVista -- Mayavi/PySurfer is not as strict
    objs = gc.get_objects()
    bad = list()
    for o in objs:
        try:
            name = o.__class__.__name__
        except Exception:  # old Python, probably
            pass
        else:
            if name.startswith('vtk') and id(o) not in ignore:
                bad.append(name)
        del o
    del objs, ignore, Brain
    assert len(bad) == 0, 'VTK objects linger:\n' + '\n'.join(bad)


def pytest_sessionfinish(session, exitstatus):
    """Handle the end of the session."""
    n = session.config.option.durations
    if n is None:
        return
    print('\n')
    try:
        import pytest_harvest
    except ImportError:
        print('Module-level timings require pytest-harvest')
        return
    from py.io import TerminalWriter
    # get the number to print
    res = pytest_harvest.get_session_synthesis_dct(session)
    files = dict()
    for key, val in res.items():
        parts = Path(key.split(':')[0]).parts
        # split mne/tests/test_whatever.py into separate categories since these
        # are essentially submodule-level tests. Keeping just [:3] works,
        # except for mne/viz where we want level-4 granulatity
        split_submodules = (('mne', 'viz'), ('mne', 'preprocessing'))
        parts = parts[:4 if parts[:2] in split_submodules else 3]
        if not parts[-1].endswith('.py'):
            parts = parts + ('',)
        file_key = '/'.join(parts)
        files[file_key] = files.get(file_key, 0) + val['pytest_duration_s']
    files = sorted(list(files.items()), key=lambda x: x[1])[::-1]
    # print
    files = files[:n]
    if len(files):
        writer = TerminalWriter()
        writer.line()  # newline
        writer.sep('=', f'slowest {n} test module{_pl(n)}')
        names, timings = zip(*files)
        timings = [f'{timing:0.2f}s total' for timing in timings]
        rjust = max(len(timing) for timing in timings)
        timings = [timing.rjust(rjust) for timing in timings]
        for name, timing in zip(names, timings):
            writer.line(f'{timing.ljust(15)}{name}')


@pytest.fixture(scope="function", params=('Numba', 'NumPy'))
def numba_conditional(monkeypatch, request):
    """Test both code paths on machines that have Numba."""
    assert request.param in ('Numba', 'NumPy')
    if request.param == 'NumPy' and has_numba:
        monkeypatch.setattr(
            cluster_level, '_get_buddies', cluster_level._get_buddies_fallback)
        monkeypatch.setattr(
            cluster_level, '_get_selves', cluster_level._get_selves_fallback)
        monkeypatch.setattr(
            cluster_level, '_where_first', cluster_level._where_first_fallback)
        monkeypatch.setattr(
            numerics, '_arange_div', numerics._arange_div_fallback)
    if request.param == 'Numba' and not has_numba:
        pytest.skip('Numba not installed')
    yield request.param


# Create one nbclient and reuse it
@pytest.fixture(scope='session')
def _nbclient():
    try:
        import nbformat
        from jupyter_client import AsyncKernelManager
        from nbclient import NotebookClient
        from ipywidgets import Button  # noqa
        import ipyvtklink  # noqa
    except Exception as exc:
        return pytest.skip(f'Skipping Notebook test: {exc}')
    km = AsyncKernelManager(config=None)
    nb = nbformat.reads("""
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata":{},
   "outputs": [],
   "source":[]
  }
 ],
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version":3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}""", as_version=4)
    client = NotebookClient(nb, km=km)
    yield client
    client._cleanup_kernel()


@pytest.fixture(scope='function')
def nbexec(_nbclient):
    """Execute Python code in a notebook."""
    # Adapted/simplified from nbclient/client.py (BSD-3-Clause)
    _nbclient._cleanup_kernel()

    def execute(code, reset=False):
        _nbclient.reset_execution_trackers()
        with _nbclient.setup_kernel():
            assert _nbclient.kc is not None
            cell = Bunch(cell_type='code', metadata={}, source=dedent(code))
            _nbclient.execute_cell(cell, 0, execution_count=0)
            _nbclient.set_widgets_metadata()

    yield execute


def pytest_runtest_call(item):
    """Run notebook code written in Python."""
    if 'nbexec' in getattr(item, 'fixturenames', ()):
        nbexec = item.funcargs['nbexec']
        code = inspect.getsource(getattr(item.module, item.name.split('[')[0]))
        code = code.splitlines()
        ci = 0
        for ci, c in enumerate(code):
            if c.startswith('    '):  # actual content
                break
        code = '\n'.join(code[ci:])

        def run(nbexec=nbexec, code=code):
            nbexec(code)

        item.runtest = run
    return
