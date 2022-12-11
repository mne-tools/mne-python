# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from contextlib import contextmanager
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
from unittest import mock

import numpy as np

import mne
from mne import read_events, pick_types, Epochs
from mne.channels import read_layout
from mne.coreg import create_default_subject
from mne.datasets import testing
from mne.fixes import has_numba, _compare_version
from mne.io import read_raw_fif, read_raw_ctf, read_raw_nirx, read_raw_snirf
from mne.stats import cluster_level
from mne.utils import (_pl, _assert_no_instances, numerics, Bunch,
                       _check_qt_version, _TempDir, check_version)

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

nirx_path = test_path / 'NIRx'
snirf_path = test_path / 'SNIRF'
nirsport2 = nirx_path / 'nirsport_v2' / 'aurora_recording _w_short_and_acc'
nirsport2_snirf = (
    snirf_path / 'NIRx' / 'NIRSport2' / '1.0.3' /
    '2021-05-05_001.snirf')
nirsport2_2021_9 = nirx_path / 'nirsport_v2' / 'aurora_2021_9'
nirsport2_20219_snirf = (
    snirf_path / 'NIRx' / 'NIRSport2' / '2021.9' /
    '2021-10-01_002.snirf')

# data from mne.io.tests.data
base_dir = op.join(op.dirname(__file__), 'io', 'tests', 'data')
fname_raw_io = op.join(base_dir, 'test_raw.fif')
fname_event_io = op.join(base_dir, 'test-eve.fif')
fname_cov_io = op.join(base_dir, 'test-cov.fif')
fname_evoked_io = op.join(base_dir, 'test-ave.fif')
event_id, tmin, tmax = 1, -0.1, 1.0
vv_layout = read_layout('Vectorview-all')

collect_ignore = [
    'export/_brainvision.py',
    'export/_eeglab.py',
    'export/_edf.py']


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ('slowtest', 'ultraslowtest', 'pgtest'):
        config.addinivalue_line('markers', marker)

    # Fixtures
    for fixture in ('matplotlib_config', 'close_all', 'check_verbose',
                    'qt_config', 'protect_config'):
        config.addinivalue_line('usefixtures', fixture)

    # pytest-qt uses PYTEST_QT_API, but let's make it respect qtpy's QT_API
    # if present
    if os.getenv('PYTEST_QT_API') is None and os.getenv('QT_API') is not None:
        os.environ['PYTEST_QT_API'] = os.environ['QT_API']

    # Warnings
    # - Once SciPy updates not to have non-integer and non-tuple errors (1.2.0)
    #   we should remove them from here.
    # - This list should also be considered alongside reset_warnings in
    #   doc/conf.py.
    if os.getenv('MNE_IGNORE_WARNINGS_IN_TESTS', '') != 'true':
        first_kind = 'error'
    else:
        first_kind = 'always'
    warning_lines = r"""
    {0}::
    # matplotlib->traitlets (notebook)
    ignore:Passing unrecognized arguments to super.*:DeprecationWarning
    # notebook tests
    ignore:There is no current event loop:DeprecationWarning
    ignore:unclosed <socket\.socket:ResourceWarning
    ignore:unclosed event loop <:ResourceWarning
    # ignore if joblib is missing
    ignore:joblib not installed.*:RuntimeWarning
    # TODO: This is indicative of a problem
    ignore:.*Matplotlib is currently using agg.*:
    # qdarkstyle
    ignore:.*Setting theme=.*:RuntimeWarning
    # scikit-learn using this arg
    ignore:.*The 'sym_pos' keyword is deprecated.*:DeprecationWarning
    # Should be removable by 2022/07/08, SciPy savemat issue
    ignore:.*elementwise comparison failed; returning scalar in.*:FutureWarning
    # numba with NumPy dev
    ignore:`np.MachAr` is deprecated.*:DeprecationWarning
    # matplotlib 3.6 and pyvista/nilearn
    ignore:.*cmap function will be deprecated.*:
    # joblib hasn't updated to avoid distutils
    ignore:.*distutils package is deprecated.*:DeprecationWarning
    # nbclient
    ignore:Passing a schema to Validator\.iter_errors is deprecated.*:
    ignore:Unclosed context <zmq.asyncio.Context.*:ResourceWarning
    ignore:Jupyter is migrating its paths.*:DeprecationWarning
    ignore:Widget\..* is deprecated\.:DeprecationWarning
    # hopefully temporary https://github.com/matplotlib/matplotlib/pull/24455#issuecomment-1319318629
    ignore:The circles attribute was deprecated in Matplotlib.*:
    # PySide6
    ignore:Enum value .* is marked as deprecated:DeprecationWarning
    ignore:Function.*is marked as deprecated, please check the documentation.*:DeprecationWarning
    """.format(first_kind)  # noqa: E501
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
def qt_config():
    """Configure the Qt backend for viz tests."""
    os.environ['_MNE_BROWSER_NO_BLOCK'] = 'true'


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
        plt.rcParams['figure.raise_window'] = False
    except KeyError:  # MPL < 3.3
        pass

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super(CallbackRegistryReraise, self).__init__(exception_handler)

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


# Skip functions or modules for mne-qt-browser < 0.2.0
pre_2_0_skip_modules = ['mne.viz.tests.test_epochs',
                        'mne.viz.tests.test_ica']
pre_2_0_skip_funcs = ['test_plot_raw_white',
                      'test_plot_raw_selection']


def _check_pyqtgraph(request):
    # Check Qt
    qt_version, api = _check_qt_version(return_api=True)
    if (not qt_version) or _compare_version(qt_version, '<', '5.12'):
        pytest.skip(f'Qt API {api} has version {qt_version} '
                    f'but pyqtgraph needs >= 5.12!')
    try:
        import mne_qt_browser  # noqa: F401
        # Check mne-qt-browser version
        lower_2_0 = _compare_version(mne_qt_browser.__version__, '<', '0.2.0')
        m_name = request.function.__module__
        f_name = request.function.__name__
        if lower_2_0 and m_name in pre_2_0_skip_modules:
            pytest.skip(f'Test-Module "{m_name}" was skipped for'
                        f' mne-qt-browser < 0.2.0')
        elif lower_2_0 and f_name in pre_2_0_skip_funcs:
            pytest.skip(f'Test "{f_name}" was skipped for '
                        f'mne-qt-browser < 0.2.0')
    except Exception:
        pytest.skip('Requires mne_qt_browser')
    else:
        ver = mne_qt_browser.__version__
        if api != 'PyQt5' and _compare_version(ver, '<=', '0.2.6'):
            pytest.skip(f'mne_qt_browser {ver} requires PyQt5, API is {api}')


@pytest.fixture
def pg_backend(request, garbage_collect):
    """Use for pyqtgraph-specific test-functions."""
    _check_pyqtgraph(request)
    from mne_qt_browser._pg_figure import MNEQtBrowser
    with use_browser_backend('qt') as backend:
        backend._close_all()
        yield backend
        backend._close_all()
        # This shouldn't be necessary, but let's make sure nothing is stale
        import mne_qt_browser
        mne_qt_browser._browser_instances.clear()
        if check_version('mne_qt_browser', min_version='0.4'):
            _assert_no_instances(
                MNEQtBrowser, f'Closure of {request.node.name}')


@pytest.fixture(params=[
    'matplotlib',
    pytest.param('qt', marks=pytest.mark.pgtest),
])
def browser_backend(request, garbage_collect, monkeypatch):
    """Parametrizes the name of the browser backend."""
    backend_name = request.param
    if backend_name == 'qt':
        _check_pyqtgraph(request)
    with use_browser_backend(backend_name) as backend:
        backend._close_all()
        monkeypatch.setenv('MNE_BROWSE_RAW_SIZE', '10,10')
        yield backend
        backend._close_all()
        if backend_name == 'qt':
            # This shouldn't be necessary, but let's make sure nothing is stale
            import mne_qt_browser
            mne_qt_browser._browser_instances.clear()


@pytest.fixture(params=["pyvistaqt"])
def renderer(request, options_3d, garbage_collect):
    """Yield the 3D backends."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["pyvistaqt"])
def renderer_pyvistaqt(request, options_3d, garbage_collect):
    """Yield the PyVista backend."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["notebook"])
def renderer_notebook(request, options_3d):
    """Yield the 3D notebook renderer."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(scope="module", params=["pyvistaqt"])
def renderer_interactive_pyvistaqt(request, options_3d):
    """Yield the interactive PyVista backend."""
    with _use_backend(request.param, interactive=True) as renderer:
        yield renderer


@pytest.fixture(scope="module", params=["pyvistaqt"])
def renderer_interactive(request, options_3d):
    """Yield the interactive 3D backends."""
    with _use_backend(request.param, interactive=True) as renderer:
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
    from mne.viz.backends.tests._utils import (has_pyvista,
                                               has_imageio_ffmpeg,
                                               has_pyvistaqt)
    from mne.viz.backends._utils import _notebook_vtk_works
    if name in ('pyvistaqt', 'notebook'):
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")
        if not has_imageio_ffmpeg():
            pytest.skip("Test skipped, requires imageio-ffmpeg")
    if name == 'pyvistaqt' and not _check_qt_version():
        pytest.skip("Test skipped, requires Qt.")
    if name == 'pyvistaqt' and not has_pyvistaqt():
        pytest.skip("Test skipped, requires pyvistaqt")
    if name == 'notebook' and not _notebook_vtk_works():
        pytest.skip("Test skipped, requires working notebook vtk")


@pytest.fixture(scope='session')
def pixel_ratio():
    """Get the pixel ratio."""
    from mne.viz.backends.tests._utils import has_pyvista
    # _check_qt_version will init an app for us, so no need for us to do it
    if not has_pyvista() or not _check_qt_version():
        return 1.
    from qtpy.QtWidgets import QMainWindow
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


@pytest.fixture(params=[testing._pytest_param()])
def subjects_dir_tmp_few(tmp_path):
    """Copy fewer files to a tmp_path."""
    subjects_path = tmp_path / 'subjects'
    os.mkdir(subjects_path)
    # add fsaverage
    create_default_subject(subjects_dir=subjects_path, fs_home=test_path,
                           verbose=True)
    # add sample (with few files)
    sample_path = subjects_path / 'sample'
    os.makedirs(sample_path / 'bem')
    for dirname in ('mri', 'surf'):
        shutil.copytree(
            test_path / 'subjects' / 'sample' / dirname, sample_path / dirname)
    return subjects_path


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


@pytest.fixture
def fwd_volume_small(_fwd_subvolume):
    """Provide a small volumetric source space."""
    return _fwd_subvolume.copy()


@pytest.fixture(scope='session')
def _all_src_types_fwd(_fwd_surf, _fwd_subvolume):
    """Create all three forward types (surf, vol, mixed)."""
    fwds = dict(
        surface=_fwd_surf.copy(),
        volume=_fwd_subvolume.copy())
    with pytest.raises(RuntimeError,
                       match='Invalid source space with kinds'):
        fwds['volume']['src'] + fwds['surface']['src']

    # mixed (4)
    fwd = fwds['surface'].copy()
    f2 = fwds['volume'].copy()
    del _fwd_surf, _fwd_subvolume
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
        with pytest.warns(RuntimeWarning, match='has been reduced'):
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


# We can't use monkeypatch because its scope (function-level) conflicts with
# the requests fixture (module-level), so we live with a module-scoped version
# that uses mock
@pytest.fixture(scope='module')
def options_3d():
    """Disable advanced 3d rendering."""
    with mock.patch.dict(
        os.environ, {
            "MNE_3D_OPTION_ANTIALIAS": "false",
            "MNE_3D_OPTION_DEPTH_PEELING": "false",
            "MNE_3D_OPTION_SMOOTH_SHADING": "false",
        }
    ):
        yield


@pytest.fixture(scope='session')
def protect_config():
    """Protect ~/.mne."""
    temp = _TempDir()
    with mock.patch.dict(os.environ, {"_MNE_FAKE_HOME_DIR": temp}):
        yield


@pytest.fixture()
def brain_gc(request):
    """Ensure that brain can be properly garbage collected."""
    keys = (
        'renderer_interactive',
        'renderer_interactive_pyvistaqt',
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
    # Check VTK
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


_files = list()


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
    _files[:] = files[:n]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the module-level timings."""
    writer = terminalreporter
    n = len(_files)
    if n:
        writer.line('')  # newline
        writer.write_sep('=', f'slowest {n} test module{_pl(n)}')
        names, timings = zip(*_files)
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
    try:
        client._cleanup_kernel()
    except Exception:
        pass


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


@pytest.mark.filterwarnings('ignore:.*Extraction of measurement.*:')
@pytest.fixture(params=(
    [nirsport2, nirsport2_snirf, testing._pytest_param()],
    [nirsport2_2021_9, nirsport2_20219_snirf, testing._pytest_param()],
))
def nirx_snirf(request):
    """Return a (raw_nirx, raw_snirf) matched pair."""
    pytest.importorskip('h5py')
    skipper = request.param[2].marks[0].mark
    if skipper.args[0]:  # will skip
        pytest.skip(skipper.kwargs['reason'])
    return (read_raw_nirx(request.param[0], preload=True),
            read_raw_snirf(request.param[1], preload=True))
