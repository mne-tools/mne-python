# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import gc
import inspect
import os
import os.path as op
import platform
import re
import shutil
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from unittest import mock

import numpy as np
import pytest
from pytest import StashKey

import mne
from mne import Epochs, pick_types, read_events
from mne.channels import read_layout
from mne.coreg import create_default_subject
from mne.datasets import testing
from mne.fixes import _compare_version, has_numba
from mne.io import read_raw_ctf, read_raw_fif, read_raw_nirx, read_raw_snirf
from mne.stats import cluster_level
from mne.utils import (
    Bunch,
    _assert_no_instances,
    _check_qt_version,
    _pl,
    _record_warnings,
    _TempDir,
    check_version,
    numerics,
)

# data from sample dataset
from mne.viz._figure import use_browser_backend
from mne.viz.backends._utils import _init_mne_qtapp

test_path = testing.data_path(download=False)
s_path = op.join(test_path, "MEG", "sample")
fname_evoked = op.join(s_path, "sample_audvis_trunc-ave.fif")
fname_cov = op.join(s_path, "sample_audvis_trunc-cov.fif")
fname_fwd = op.join(s_path, "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif")
fname_fwd_full = op.join(s_path, "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif")
bem_path = op.join(test_path, "subjects", "sample", "bem")
fname_bem = op.join(bem_path, "sample-1280-bem.fif")
fname_aseg = op.join(test_path, "subjects", "sample", "mri", "aseg.mgz")
subjects_dir = op.join(test_path, "subjects")
fname_src = op.join(bem_path, "sample-oct-4-src.fif")
fname_trans = op.join(s_path, "sample_audvis_trunc-trans.fif")

ctf_dir = op.join(test_path, "CTF")
fname_ctf_continuous = op.join(ctf_dir, "testdata_ctf.ds")

nirx_path = test_path / "NIRx"
snirf_path = test_path / "SNIRF"
nirsport2 = nirx_path / "nirsport_v2" / "aurora_recording _w_short_and_acc"
nirsport2_snirf = snirf_path / "NIRx" / "NIRSport2" / "1.0.3" / "2021-05-05_001.snirf"
nirsport2_2021_9 = nirx_path / "nirsport_v2" / "aurora_2021_9"
nirsport2_20219_snirf = (
    snirf_path / "NIRx" / "NIRSport2" / "2021.9" / "2021-10-01_002.snirf"
)

# data from mne.io.tests.data
base_dir = op.join(op.dirname(__file__), "io", "tests", "data")
fname_raw_io = op.join(base_dir, "test_raw.fif")
fname_event_io = op.join(base_dir, "test-eve.fif")
fname_cov_io = op.join(base_dir, "test-cov.fif")
fname_evoked_io = op.join(base_dir, "test-ave.fif")
event_id, tmin, tmax = 1, -0.1, 1.0
vv_layout = read_layout("Vectorview-all")

collect_ignore = ["export/_brainvision.py", "export/_eeglab.py", "export/_edf.py"]


def pytest_configure(config: pytest.Config):
    """Configure pytest options."""
    # Markers
    # can be queried with `pytest --markers` for example
    for marker in (
        "slowtest: mark a test as slow",
        "ultraslowtest: mark a test as ultraslow or to be run rarely",
        "pgtest: mark a test as relevant for mne-qt-browser",
        "pvtest: mark a test as relevant for pyvistaqt",
        "allow_unclosed: allow unclosed pyvistaqt instances",
    ):
        config.addinivalue_line("markers", marker)

    # Fixtures
    for fixture in (
        "matplotlib_config",
        "qt_config",
        "protect_config",
    ):
        config.addinivalue_line("usefixtures", fixture)

    # pytest-qt uses PYTEST_QT_API, but let's make it respect qtpy's QT_API
    # if present
    if os.getenv("PYTEST_QT_API") is None and os.getenv("QT_API") is not None:
        os.environ["PYTEST_QT_API"] = os.environ["QT_API"]

    # suppress:
    # Debugger warning: It seems that frozen modules are being used, which may
    # make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
    # to python to disable frozen modules.
    if os.getenv("PYDEVD_DISABLE_FILE_VALIDATION") is None:
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    # https://numba.readthedocs.io/en/latest/reference/deprecation.html#deprecation-of-old-style-numba-captured-errors  # noqa: E501
    if "NUMBA_CAPTURED_ERRORS" not in os.environ:
        os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"

    # Warnings
    # - Once SciPy updates not to have non-integer and non-tuple errors (1.2.0)
    #   we should remove them from here.
    # - This list should also be considered alongside reset_warnings in
    #   doc/conf.py.
    if os.getenv("MNE_IGNORE_WARNINGS_IN_TESTS", "") not in ("true", "1"):
        first_kind = "error"
    else:
        first_kind = "always"
    warning_lines = f"    {first_kind}::"
    warning_lines += r"""
    # matplotlib->traitlets (notebook)
    ignore:Passing unrecognized arguments to super.*:DeprecationWarning
    # notebook tests
    ignore:There is no current event loop:DeprecationWarning
    ignore:unclosed <socket\.socket:ResourceWarning
    ignore:unclosed event loop <:ResourceWarning
    # ignore if joblib is missing
    ignore:joblib not installed.*:RuntimeWarning
    # qdarkstyle
    ignore:.*Setting theme=.*:RuntimeWarning
    # nbclient
    ignore:Passing a schema to Validator\.iter_errors is deprecated.*:
    ignore:Unclosed context <zmq.asyncio.Context.*:ResourceWarning
    ignore:Jupyter is migrating its paths.*:DeprecationWarning
    ignore:datetime\.datetime\.utcnow\(\) is deprecated.*:DeprecationWarning
    ignore:Widget\..* is deprecated\.:DeprecationWarning
    ignore:.*is deprecated in pyzmq.*:DeprecationWarning
    ignore:The `ipykernel.comm.Comm` class has been deprecated.*:DeprecationWarning
    ignore:Proactor event loop does not implement:RuntimeWarning
    # PySide6
    ignore:Enum value .* is marked as deprecated:DeprecationWarning
    ignore:Function.*is marked as deprecated, please check the documentation.*:DeprecationWarning
    ignore:Failed to disconnect.*:RuntimeWarning
    # pkg_resources usage bug
    ignore:Implementing implicit namespace packages.*:DeprecationWarning
    ignore:Deprecated call to `pkg_resources.*:DeprecationWarning
    ignore:pkg_resources is deprecated as an API.*:DeprecationWarning
    # numpy distutils used by SciPy
    ignore:(\n|.)*numpy\.distutils` is deprecated since NumPy(\n|.)*:DeprecationWarning
    ignore:datetime\.utcfromtimestamp.*is deprecated:DeprecationWarning
    ignore:The numpy\.array_api submodule is still experimental.*:UserWarning
    # tqdm (Fedora)
    ignore:.*'tqdm_asyncio' object has no attribute 'last_print_t':pytest.PytestUnraisableExceptionWarning
    # Windows CIs using MESA get this
    ignore:Mesa version 10\.2\.4 is too old for translucent.*:RuntimeWarning
    # Matplotlib->tz
    ignore:datetime.datetime.utcfromtimestamp.*:DeprecationWarning
    # joblib
    ignore:ast\.Num is deprecated.*:DeprecationWarning
    ignore:Attribute n is deprecated and will be removed in Python 3\.14.*:DeprecationWarning
    # numpydoc
    ignore:ast\.NameConstant is deprecated and will be removed in Python 3\.14.*:DeprecationWarning
    # pooch
    ignore:Python 3\.14 will, by default, filter extracted tar archives.*:DeprecationWarning
    # pandas
    ignore:\n*Pyarrow will become a required dependency of pandas.*:DeprecationWarning
    ignore:np\.find_common_type is deprecated.*:DeprecationWarning
    ignore:Python binding for RankQuantileOptions.*:
    # pyvista <-> NumPy 2.0
    ignore:__array_wrap__ must accept context and return_scalar arguments.*:DeprecationWarning
    # pyvista <-> VTK dev
    ignore:Call to deprecated method GetInputAsDataSet.*:DeprecationWarning
    # nibabel <-> NumPy 2.0
    ignore:__array__ implementation doesn't accept a copy.*:DeprecationWarning
    # quantities via neo
    ignore:The 'copy' argument in Quantity is deprecated.*:
    # debugpy uses deprecated matplotlib API
    ignore:The (non_)?interactive_bk attribute was deprecated.*:
    # SWIG (via OpenMEEG)
    ignore:.*builtin type swigvarlink has no.*:DeprecationWarning
    # eeglabio
    ignore:numpy\.core\.records is deprecated.*:DeprecationWarning
    ignore:Starting field name with a underscore.*:
    # joblib
    ignore:process .* is multi-threaded, use of fork/exec.*:DeprecationWarning
    # sklearn
    ignore:Python binding for RankQuantileOptions.*:RuntimeWarning
    ignore:.*The `disp` and `iprint` options of the L-BFGS-B solver.*:DeprecationWarning
    """  # noqa: E501
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


def pytest_collection_modifyitems(items: list[pytest.Item]):
    """Add slowtest marker automatically to anything marked ultraslow."""
    for item in items:
        if len(list(item.iter_markers("ultraslowtest"))):
            item.add_marker(pytest.mark.slowtest)


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
        pytest.fail(
            ".".join([request.module.__name__, request.function.__name__])
            + " modifies logger.level"
        )


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def add_mne(doctest_namespace):
    """Add mne to the namespace."""
    doctest_namespace["mne"] = mne


@pytest.fixture(scope="function")
def verbose_debug():
    """Run a test with debug verbosity."""
    with mne.utils.use_log_level("debug"):
        yield


@pytest.fixture(scope="session")
def qt_config():
    """Configure the Qt backend for viz tests."""
    os.environ["_MNE_BROWSER_NO_BLOCK"] = "true"
    if "_MNE_BROWSER_BACK" not in os.environ:
        os.environ["_MNE_BROWSER_BACK"] = "true"


@pytest.fixture(scope="session")
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    from matplotlib import cbook

    # Allow for easy interactive debugging with a call like:
    #
    #     $ MNE_MPL_TESTING_BACKEND=Qt5Agg pytest mne/viz/tests/test_raw.py -k annotation -x --pdb  # noqa: E501
    #
    try:
        want = os.environ["MNE_MPL_TESTING_BACKEND"]
    except KeyError:
        want = "agg"  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings("ignore")
        matplotlib.use(want, force=True)
    import matplotlib.pyplot as plt

    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.raise_window"] = False

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super().__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture(scope="session")
def azure_windows():
    """Determine if running on Azure Windows."""
    return (
        os.getenv("AZURE_CI_WINDOWS", "false").lower() == "true"
        and platform.system() == "Windows"
    )


@pytest.fixture(scope="function")
def raw_orig():
    """Get raw data without any change to it from mne.io.tests.data."""
    raw = read_raw_fif(fname_raw_io, preload=True)
    return raw


@pytest.fixture(scope="function")
def raw():
    """
    Get raw data and pick channels to reduce load for testing.

    (from mne.io.tests.data)
    """
    raw = read_raw_fif(fname_raw_io, preload=True)
    # Throws a warning about a changed unit.
    with pytest.warns(RuntimeWarning, match="unit"):
        raw.set_channel_types({raw.ch_names[0]: "ias"})
    raw.pick(raw.ch_names[:9])
    raw.info.normalize_proj()  # Fix projectors after subselection
    return raw


@pytest.fixture(scope="function")
def raw_ctf():
    """Get ctf raw data from mne.io.tests.data."""
    raw_ctf = read_raw_ctf(fname_ctf_continuous, preload=True)
    return raw_ctf


@pytest.fixture(scope="function")
def raw_spectrum(raw):
    """Get raw with power spectral density computed from mne.io.tests.data."""
    return raw.compute_psd()


@pytest.fixture(scope="function")
def events():
    """Get events from mne.io.tests.data."""
    return read_events(fname_event_io)


def _get_epochs(stop=5, meg=True, eeg=False, n_chan=20):
    """Get epochs."""
    raw = read_raw_fif(fname_raw_io)
    events = read_events(fname_event_io)
    picks = pick_types(
        raw.info, meg=meg, eeg=eeg, stim=False, ecg=False, eog=False, exclude="bads"
    )
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    with pytest.warns(RuntimeWarning, match="projection"):
        epochs = Epochs(
            raw,
            events[:stop],
            event_id,
            tmin,
            tmax,
            picks=picks,
            proj=False,
            preload=False,
        )
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


@pytest.fixture()
def epochs_spectrum():
    """Get epochs with power spectral density computed from mne.io.tests.data."""
    return _get_epochs().load_data().compute_psd()


@pytest.fixture()
def epochs_tfr():
    """Get an EpochsTFR computed from mne.io.tests.data."""
    epochs = _get_epochs().load_data()
    return epochs.compute_tfr(method="morlet", freqs=np.linspace(20, 40, num=5))


@pytest.fixture()
def average_tfr(epochs_tfr):
    """Get an AverageTFR computed by averaging an EpochsTFR (this is small & fast)."""
    return epochs_tfr.average()


@pytest.fixture()
def full_average_tfr(full_evoked):
    """Get an AverageTFR computed from Evoked.

    This is slower than the `average_tfr` fixture, but a few TFR.plot_* tests need it.
    """
    return full_evoked.compute_tfr(method="morlet", freqs=np.linspace(20, 40, num=5))


@pytest.fixture()
def raw_tfr(raw):
    """Get a RawTFR computed from mne.io.tests.data."""
    return raw.compute_tfr(method="morlet", freqs=np.linspace(20, 40, num=5))


@pytest.fixture()
def epochs_empty():
    """Get empty epochs from mne.io.tests.data."""
    epochs = _get_epochs(meg=True, eeg=True).load_data()
    with pytest.warns(RuntimeWarning, match="were dropped"):
        epochs.drop_bad(reject={"mag": 1e-20})

    return epochs


@pytest.fixture(scope="session", params=[testing._pytest_param()])
def _full_evoked():
    # This is session scoped, so be sure not to modify its return value (use
    # `full_evoked` fixture instead)
    return mne.read_evokeds(fname_evoked, condition="Left Auditory", baseline=(None, 0))


@pytest.fixture(scope="session", params=[testing._pytest_param()])
def _evoked(_full_evoked):
    # This is session scoped, so be sure not to modify its return value (use `evoked`
    # fixture instead)
    return _full_evoked.copy().crop(0, 0.2)


@pytest.fixture()
def evoked(_evoked):
    """Get truncated evoked data."""
    return _evoked.copy()


@pytest.fixture()
def full_evoked(_full_evoked):
    """Get full-duration evoked data (needed for, e.g., testing TFR)."""
    return _full_evoked.copy()


@pytest.fixture(scope="function", params=[testing._pytest_param()])
def noise_cov():
    """Get a noise cov from the testing dataset."""
    return mne.read_cov(fname_cov)


@pytest.fixture
def noise_cov_io():
    """Get noise-covariance (from mne.io.tests.data)."""
    return mne.read_cov(fname_cov_io)


@pytest.fixture(scope="function")
def bias_params_free(evoked, noise_cov):
    """Provide inputs for free bias functions."""
    fwd = mne.read_forward_solution(fname_fwd)
    return _bias_params(evoked, noise_cov, fwd)


@pytest.fixture(scope="function")
def bias_params_fixed(evoked, noise_cov):
    """Provide inputs for fixed bias functions."""
    fwd = mne.read_forward_solution(fname_fwd)
    mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, copy=False)
    return _bias_params(evoked, noise_cov, fwd)


def _bias_params(evoked, noise_cov, fwd):
    evoked.pick(picks=["meg", "eeg"])
    # restrict to limited set of verts (small src here) and one hemi for speed
    vertices = [fwd["src"][0]["vertno"].copy(), []]
    stc = mne.SourceEstimate(
        np.zeros((sum(len(v) for v in vertices), 1)), vertices, 0, 1
    )
    fwd = mne.forward.restrict_forward_to_stc(fwd, stc)
    assert fwd["sol"]["row_names"] == noise_cov["names"]
    assert noise_cov["names"] == evoked.ch_names
    evoked = mne.EvokedArray(fwd["sol"]["data"].copy(), evoked.info)
    data_cov = noise_cov.copy()
    data = fwd["sol"]["data"] @ fwd["sol"]["data"].T
    data *= 1e-14  # 100 nAm at each source, effectively (1e-18 would be 1 nAm)
    # This is rank-deficient, so let's make it actually positive semidefinite
    # by regularizing a tiny bit
    data.flat[:: data.shape[0] + 1] += mne.make_ad_hoc_cov(evoked.info)["data"]
    # Do our projection
    proj, _, _ = mne._fiff.proj.make_projector(data_cov["projs"], data_cov["names"])
    data = proj @ data @ proj.T
    data_cov["data"][:] = data
    assert data_cov["data"].shape[0] == len(noise_cov["names"])
    want = np.arange(fwd["sol"]["data"].shape[1])
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
    with use_browser_backend("matplotlib") as backend:
        yield backend
        backend._close_all()


# Skip functions or modules for mne-qt-browser < 0.2.0
pre_2_0_skip_modules = ["mne.viz.tests.test_epochs", "mne.viz.tests.test_ica"]
pre_2_0_skip_funcs = ["test_plot_raw_white", "test_plot_raw_selection"]


def _check_pyqtgraph(request):
    # Check Qt
    qt_version, api = _check_qt_version(return_api=True)
    if (not qt_version) or _compare_version(qt_version, "<", "5.12"):
        pytest.skip(
            f"Qt API {api} has version {qt_version} but pyqtgraph needs >= 5.12!"
        )
    try:
        import mne_qt_browser  # noqa: F401

        # Check mne-qt-browser version
        lower_2_0 = _compare_version(mne_qt_browser.__version__, "<", "0.2.0")
        m_name = request.function.__module__
        f_name = request.function.__name__
        if lower_2_0 and m_name in pre_2_0_skip_modules:
            pytest.skip(
                f'Test-Module "{m_name}" was skipped for mne-qt-browser < 0.2.0'
            )
        elif lower_2_0 and f_name in pre_2_0_skip_funcs:
            pytest.skip(f'Test "{f_name}" was skipped for mne-qt-browser < 0.2.0')
    except Exception:
        pytest.skip("Requires mne_qt_browser")
    else:
        ver = mne_qt_browser.__version__
        if api != "PyQt5" and _compare_version(ver, "<=", "0.2.6"):
            pytest.skip(f"mne_qt_browser {ver} requires PyQt5, API is {api}")


@pytest.fixture
def pg_backend(request, garbage_collect):
    """Use for pyqtgraph-specific test-functions."""
    _check_pyqtgraph(request)
    from mne_qt_browser._pg_figure import MNEQtBrowser

    with use_browser_backend("qt") as backend:
        backend._close_all()
        yield backend
        backend._close_all()
        # This shouldn't be necessary, but let's make sure nothing is stale
        import mne_qt_browser

        mne_qt_browser._browser_instances.clear()
        if not _test_passed(request):
            return
        _assert_no_instances(MNEQtBrowser, f"Closure of {request.node.name}")


@pytest.fixture(
    params=[
        "matplotlib",
        pytest.param("qt", marks=pytest.mark.pgtest),
    ]
)
def browser_backend(request, garbage_collect, monkeypatch):
    """Parametrizes the name of the browser backend."""
    backend_name = request.param
    if backend_name == "qt":
        _check_pyqtgraph(request)
    with use_browser_backend(backend_name) as backend:
        backend._close_all()
        monkeypatch.setenv("MNE_BROWSE_RAW_SIZE", "10,10")
        yield backend
        backend._close_all()
        if backend_name == "qt":
            # This shouldn't be necessary, but let's make sure nothing is stale
            import mne_qt_browser

            mne_qt_browser._browser_instances.clear()


@pytest.fixture(params=[pytest.param("pyvistaqt", marks=pytest.mark.pvtest)])
def renderer(request, options_3d, garbage_collect):
    """Yield the 3D backends."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=[pytest.param("pyvistaqt", marks=pytest.mark.pvtest)])
def renderer_pyvistaqt(request, options_3d, garbage_collect):
    """Yield the PyVista backend."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=[pytest.param("notebook", marks=pytest.mark.pvtest)])
def renderer_notebook(request, options_3d):
    """Yield the 3D notebook renderer."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=[pytest.param("pyvistaqt", marks=pytest.mark.pvtest)])
def renderer_interactive_pyvistaqt(request, options_3d, qt_windows_closed):
    """Yield the interactive PyVista backend."""
    with _use_backend(request.param, interactive=True) as renderer:
        yield renderer


@pytest.fixture(params=[pytest.param("pyvistaqt", marks=pytest.mark.pvtest)])
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
    from mne.viz.backends._utils import _notebook_vtk_works

    pytest.importorskip("pyvista")
    pytest.importorskip("imageio_ffmpeg")
    if name == "pyvistaqt":
        pytest.importorskip("pyvistaqt")
        if not _check_qt_version():
            pytest.skip("Test skipped, requires Qt.")
    else:
        assert name == "notebook", name
        pytest.importorskip("jupyter")
        pytest.importorskip("ipympl")
        pytest.importorskip("trame")
        pytest.importorskip("trame_vtk")
        pytest.importorskip("trame_vuetify")
        if not _notebook_vtk_works():
            pytest.skip("Test skipped, requires working notebook vtk")


@pytest.fixture(scope="session")
def pixel_ratio():
    """Get the pixel ratio."""
    # _check_qt_version will init an app for us, so no need for us to do it
    if not check_version("pyvista", "0.32") or not _check_qt_version():
        return 1.0
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMainWindow

    app = _init_mne_qtapp()
    app.processEvents()
    window = QMainWindow()
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    ratio = float(window.devicePixelRatio())
    window.close()
    return ratio


@pytest.fixture(scope="function", params=[testing._pytest_param()])
def subjects_dir_tmp(tmp_path):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ("sample", "fsaverage"):
        shutil.copytree(op.join(subjects_dir, key), str(tmp_path / key))
    return str(tmp_path)


@pytest.fixture(params=[testing._pytest_param()])
def subjects_dir_tmp_few(tmp_path):
    """Copy fewer files to a tmp_path."""
    subjects_path = tmp_path / "subjects"
    os.mkdir(subjects_path)
    # add fsaverage
    create_default_subject(subjects_dir=subjects_path, fs_home=test_path, verbose=True)
    # add sample (with few files)
    sample_path = subjects_path / "sample"
    os.makedirs(sample_path / "bem")
    for dirname in ("mri", "surf"):
        shutil.copytree(
            test_path / "subjects" / "sample" / dirname, sample_path / dirname
        )
    return subjects_path


# Scoping these as session will make things faster, but need to make sure
# not to modify them in-place in the tests, so keep them private
@pytest.fixture(scope="session", params=[testing._pytest_param()])
def _evoked_cov_sphere(_evoked):
    """Compute a small evoked/cov/sphere combo for use with forwards."""
    evoked = _evoked.copy().pick(picks="meg")
    evoked.pick(evoked.ch_names[::4])
    assert len(evoked.ch_names) == 77
    cov = mne.read_cov(fname_cov)
    sphere = mne.make_sphere_model("auto", "auto", evoked.info)
    return evoked, cov, sphere


@pytest.fixture(scope="session")
def _fwd_surf(_evoked_cov_sphere):
    """Compute a forward for a surface source space."""
    evoked, cov, sphere = _evoked_cov_sphere
    src_surf = mne.read_source_spaces(fname_src)
    return mne.make_forward_solution(
        evoked.info, fname_trans, src_surf, sphere, mindist=5.0
    )


@pytest.fixture(scope="session")
def _fwd_subvolume(_evoked_cov_sphere):
    """Compute a forward for a surface source space."""
    pytest.importorskip("nibabel")
    evoked, cov, sphere = _evoked_cov_sphere
    volume_labels = ["Left-Cerebellum-Cortex", "right-Cerebellum-Cortex"]
    with pytest.raises(ValueError, match=r"Did you mean one of \['Right-Cere"):
        mne.setup_volume_source_space(
            "sample", pos=20.0, volume_label=volume_labels, subjects_dir=subjects_dir
        )
    volume_labels[1] = "R" + volume_labels[1][1:]
    src_vol = mne.setup_volume_source_space(
        "sample",
        pos=20.0,
        volume_label=volume_labels,
        subjects_dir=subjects_dir,
        add_interpolator=False,
    )
    return mne.make_forward_solution(
        evoked.info, fname_trans, src_vol, sphere, mindist=5.0
    )


@pytest.fixture
def fwd_volume_small(_fwd_subvolume):
    """Provide a small volumetric source space."""
    return _fwd_subvolume.copy()


@pytest.fixture(scope="session")
def _all_src_types_fwd(_fwd_surf, _fwd_subvolume):
    """Create all three forward types (surf, vol, mixed)."""
    fwds = dict(surface=_fwd_surf.copy(), volume=_fwd_subvolume.copy())
    with pytest.raises(RuntimeError, match="Invalid source space with kinds"):
        fwds["volume"]["src"] + fwds["surface"]["src"]

    # mixed (4)
    fwd = fwds["surface"].copy()
    f2 = fwds["volume"].copy()
    del _fwd_surf, _fwd_subvolume
    for keys, axis in [
        (("source_rr",), 0),
        (("source_nn",), 0),
        (("sol", "data"), 1),
        (("_orig_sol",), 1),
    ]:
        a, b = fwd, f2
        key = keys[0]
        if len(keys) > 1:
            a, b = a[key], b[key]
            key = keys[1]
        a[key] = np.concatenate([a[key], b[key]], axis=axis)
    fwd["sol"]["ncol"] = fwd["sol"]["data"].shape[1]
    fwd["nsource"] = fwd["sol"]["ncol"] // 3
    fwd["src"] = fwd["src"] + f2["src"]
    fwds["mixed"] = fwd

    return fwds


@pytest.fixture(scope="session")
def _all_src_types_inv_evoked(_evoked_cov_sphere, _all_src_types_fwd):
    """Compute inverses for all source types."""
    evoked, cov, _ = _evoked_cov_sphere
    invs = dict()
    for kind, fwd in _all_src_types_fwd.items():
        assert fwd["src"].kind == kind
        with pytest.warns(RuntimeWarning, match="has been reduced"):
            invs[kind] = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov)
    return invs, evoked


@pytest.fixture(scope="function")
def all_src_types_inv_evoked(_all_src_types_inv_evoked):
    """All source types of inverses, allowing for possible modification."""
    invs, evoked = _all_src_types_inv_evoked
    invs = {key: val.copy() for key, val in invs.items()}
    evoked = evoked.copy()
    return invs, evoked


@pytest.fixture(scope="function")
def mixed_fwd_cov_evoked(_evoked_cov_sphere, _all_src_types_fwd):
    """Compute inverses for all source types."""
    evoked, cov, _ = _evoked_cov_sphere
    return _all_src_types_fwd["mixed"].copy(), cov.copy(), evoked.copy()


@pytest.fixture(scope="session")
def src_volume_labels():
    """Create a 7mm source space with labels."""
    pytest.importorskip("nibabel")
    volume_labels = mne.get_volume_labels_from_aseg(fname_aseg)
    with (
        _record_warnings(),
        pytest.warns(RuntimeWarning, match="Found no usable.*t-vessel.*"),
    ):
        src = mne.setup_volume_source_space(
            "sample",
            7.0,
            mri="aseg.mgz",
            volume_label=volume_labels,
            add_interpolator=False,
            bem=fname_bem,
            subjects_dir=subjects_dir,
        )
    lut, _ = mne.read_freesurfer_lut()
    assert len(volume_labels) == 46
    assert volume_labels[0] == "Unknown"
    assert lut["Unknown"] == 0  # it will be excluded during label gen
    return src, tuple(volume_labels), lut


def _fail(*args, **kwargs):
    __tracebackhide__ = True
    raise AssertionError("Test should not download")


@pytest.fixture(scope="function")
def download_is_error(monkeypatch):
    """Prevent downloading by raising an error when it's attempted."""
    import pooch

    monkeypatch.setattr(pooch, "retrieve", _fail)
    yield


@pytest.fixture()
def fake_retrieve(monkeypatch, download_is_error):
    """Monkeypatch pooch.retrieve to avoid downloading (just touch files)."""
    import pooch

    my_func = _FakeFetch()
    monkeypatch.setattr(pooch, "retrieve", my_func)
    monkeypatch.setattr(pooch, "create", my_func)
    yield my_func


class _FakeFetch:
    def __init__(self):
        self.call_args_list = list()

    @property
    def call_count(self):
        return len(self.call_args_list)

    # Wrapper for pooch.retrieve(...) and pooch.create(...)
    def __call__(self, *args, **kwargs):
        assert "path" in kwargs
        if "fname" in kwargs:  # pooch.retrieve(...)
            self.call_args_list.append((args, kwargs))
            path = Path(kwargs["path"], kwargs["fname"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
            return path
        else:  # pooch.create(...) has been called
            self.path = kwargs["path"]
            return self

    # Wrappers for Pooch instances (e.g., in eegbci we pooch.create)
    def fetch(self, fname):
        self(path=self.path, fname=fname)

    def load_registry(self, registry):
        assert Path(registry).exists(), registry


# We can't use monkeypatch because its scope (function-level) conflicts with
# the requests fixture (module-level), so we live with a module-scoped version
# that uses mock
@pytest.fixture(scope="module")
def options_3d():
    """Disable advanced 3d rendering."""
    with mock.patch.dict(
        os.environ,
        {
            "MNE_3D_OPTION_ANTIALIAS": "false",
            "MNE_3D_OPTION_DEPTH_PEELING": "false",
            "MNE_3D_OPTION_SMOOTH_SHADING": "false",
        },
    ):
        yield


@pytest.fixture(scope="session")
def protect_config():
    """Protect ~/.mne."""
    temp = _TempDir()
    with mock.patch.dict(os.environ, {"_MNE_FAKE_HOME_DIR": temp}):
        yield


def _test_passed(request):
    if _phase_report_key not in request.node.stash:
        return True
    report = request.node.stash[_phase_report_key]
    return "call" in report and report["call"].outcome == "passed"


@pytest.fixture()
def brain_gc(request):
    """Ensure that brain can be properly garbage collected."""
    keys = (
        "renderer_interactive",
        "renderer_interactive_pyvistaqt",
        "renderer",
        "renderer_pyvistaqt",
        "renderer_notebook",
    )
    assert set(request.fixturenames) & set(keys) != set()
    for key in keys:
        if key in request.fixturenames:
            is_pv = request.getfixturevalue(key)._get_3d_backend() == "pyvistaqt"
            close_func = request.getfixturevalue(key).backend._close_all
            break
    if not is_pv:
        yield
        return
    from mne.viz import Brain

    ignore = set(id(o) for o in gc.get_objects())
    yield
    close_func()
    if not _test_passed(request):
        return
    _assert_no_instances(Brain, "after")
    # Check VTK
    objs = gc.get_objects()
    bad = list()
    for o in objs:
        try:
            name = o.__class__.__name__
        except Exception:  # old Python, probably
            pass
        else:
            if name.startswith("vtk") and id(o) not in ignore:
                bad.append(name)
        del o
    del objs, ignore, Brain
    assert len(bad) == 0, "VTK objects linger:\n" + "\n".join(bad)


_files = list()


def pytest_sessionfinish(session, exitstatus):
    """Handle the end of the session."""
    n = session.config.option.durations
    if n is None:
        return
    print("\n")
    # get the number to print
    files = defaultdict(lambda: 0.0)
    for item in session.items:
        if _phase_report_key not in item.stash:
            continue
        report = item.stash[_phase_report_key]
        dur = sum(x.duration for x in report.values())
        parts = Path(item.nodeid.split(":")[0]).parts
        # split mne/tests/test_whatever.py into separate categories since these
        # are essentially submodule-level tests. Keeping just [:3] works,
        # except for mne/viz where we want level-4 granulatity
        split_submodules = (("mne", "viz"), ("mne", "preprocessing"))
        parts = parts[: 4 if parts[:2] in split_submodules else 3]
        if not parts[-1].endswith(".py"):
            parts = parts + ("",)
        file_key = "/".join(parts)
        files[file_key] += dur
    files = sorted(list(files.items()), key=lambda x: x[1])[::-1]
    # print
    _files[:] = files[:n]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the module-level timings."""
    writer = terminalreporter
    n = len(_files)
    if n:
        writer.line("")  # newline
        writer.write_sep("=", f"slowest {n} test module{_pl(n)}")
        names, timings = zip(*_files)
        timings = [f"{timing:0.2f}s total" for timing in timings]
        rjust = max(len(timing) for timing in timings)
        timings = [timing.rjust(rjust) for timing in timings]
        for name, timing in zip(names, timings):
            writer.line(f"{timing.ljust(15)}{name}")


def pytest_report_header(config, startdir=None):
    """Add information to the pytest run header."""
    return f"MNE {mne.__version__} -- {Path(mne.__file__).parent}"


@pytest.fixture(scope="function", params=("Numba", "NumPy"))
def numba_conditional(monkeypatch, request):
    """Test both code paths on machines that have Numba."""
    assert request.param in ("Numba", "NumPy")
    if request.param == "NumPy" and has_numba:
        monkeypatch.setattr(
            cluster_level, "_get_buddies", cluster_level._get_buddies_fallback
        )
        monkeypatch.setattr(
            cluster_level, "_get_selves", cluster_level._get_selves_fallback
        )
        monkeypatch.setattr(
            cluster_level, "_where_first", cluster_level._where_first_fallback
        )
        monkeypatch.setattr(numerics, "_arange_div", numerics._arange_div_fallback)
    if request.param == "Numba" and not has_numba:
        pytest.skip("Numba not installed")
    yield request.param


# Create one nbclient and reuse it
@pytest.fixture(scope="session")
def _nbclient():
    try:
        import nbformat
        import trame  # noqa
        from ipywidgets import Button  # noqa
        from jupyter_client import AsyncKernelManager
        from nbclient import NotebookClient
    except Exception as exc:
        return pytest.skip(f"Skipping Notebook test: {exc}")
    km = AsyncKernelManager(config=None)
    nb = nbformat.reads(
        """
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
}""",
        as_version=4,
    )
    client = NotebookClient(nb, km=km)
    yield client
    try:
        client._cleanup_kernel()
    except Exception:
        pass


@pytest.fixture(scope="function")
def nbexec(_nbclient):
    """Execute Python code in a notebook."""
    # Adapted/simplified from nbclient/client.py (BSD-3-Clause)
    from nbclient.exceptions import CellExecutionError

    _nbclient._cleanup_kernel()

    def execute(code, reset=False):
        _nbclient.reset_execution_trackers()
        with _nbclient.setup_kernel():
            assert _nbclient.kc is not None
            cell = Bunch(cell_type="code", metadata={}, source=dedent(code), outputs=[])
            try:
                _nbclient.execute_cell(cell, 0, execution_count=0)
            except CellExecutionError:  # pragma: no cover
                for kind in ("stdout", "stderr"):
                    print(
                        "\n".join(
                            o["text"] for o in cell.outputs if o.get("name", "") == kind
                        ),
                        file=getattr(sys, kind),
                    )
                raise
            _nbclient.set_widgets_metadata()

    yield execute


def pytest_runtest_call(item):
    """Run notebook code written in Python."""
    if "nbexec" in getattr(item, "fixturenames", ()):
        nbexec = item.funcargs["nbexec"]
        code = inspect.getsource(getattr(item.module, item.name.split("[")[0]))
        code = code.splitlines()
        ci = 0
        for ci, c in enumerate(code):
            if c.startswith("    "):  # actual content
                break
        code = "\n".join(code[ci:])

        def run(nbexec=nbexec, code=code):
            nbexec(code)

        item.runtest = run
    return


@pytest.fixture(
    params=(
        [nirsport2, nirsport2_snirf, testing._pytest_param()],
        [nirsport2_2021_9, nirsport2_20219_snirf, testing._pytest_param()],
    )
)
def nirx_snirf(request):
    """Return a (raw_nirx, raw_snirf) matched pair."""
    pytest.importorskip("h5py")
    skipper = request.param[2].marks[0].mark
    if skipper.args[0]:  # will skip
        pytest.skip(skipper.kwargs["reason"])
    return (
        read_raw_nirx(request.param[0], preload=True),
        read_raw_snirf(request.param[1], preload=True),
    )


@pytest.fixture
def qt_windows_closed(request):
    """Ensure that no new Qt windows are open after a test."""
    _check_skip_backend("pyvistaqt")
    app = _init_mne_qtapp()

    app.processEvents()
    gc.collect()
    n_before = len(app.topLevelWidgets())
    marks = set(mark.name for mark in request.node.iter_markers())
    yield
    app.processEvents()
    gc.collect()
    if "allow_unclosed" in marks:
        return
    # Don't check when the test fails
    if not _test_passed(request):
        return
    widgets = app.topLevelWidgets()
    n_after = len(widgets)
    assert n_before == n_after, widgets[-4:]


# https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures  # noqa: E501
_phase_report_key = StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash the status of each item and turn unexpected skips into errors."""
    outcome = yield
    rep: pytest.TestReport = outcome.get_result()
    item.stash.setdefault(_phase_report_key, {})[rep.when] = rep
    if rep.outcome == "passed":  # only check for skips etc. if otherwise green
        _modify_report_skips(rep)
    return rep


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    """Turn unexpected skips during collection (e.g., module-level) into errors."""
    outcome = yield
    rep: pytest.CollectReport = outcome.get_result()
    _modify_report_skips(rep)
    return rep


# Default means "allow all skips". Can use something like "$." to mean
# "never match", i.e., "treat all skips as errors"
_valid_skips_re = re.compile(os.getenv("MNE_TEST_ALLOW_SKIP", ".*"))


# To turn unexpected skips into errors, we need to look both at the collection phase
# (for decorated tests) and the call phase (for things like `importorskip`
# within the test body). code adapted from pytest-error-for-skips
def _modify_report_skips(report: pytest.TestReport | pytest.CollectReport):
    if not report.skipped:
        return
    if isinstance(report.longrepr, tuple):
        file, lineno, reason = report.longrepr
    else:
        file, lineno, reason = "<unknown>", 1, str(report.longrepr)
    if _valid_skips_re.match(reason):
        return
    assert isinstance(report, pytest.TestReport | pytest.CollectReport), type(report)
    if file.endswith("doctest.py"):  # _python/doctest.py
        return
    # xfail tests aren't true "skips" but show up as skipped in reports
    if getattr(report, "keywords", {}).get("xfail", False):
        return
    # the above only catches marks, so we need to actually parse the report to catch
    # an xfail based on the traceback
    if " pytest.xfail( " in reason:
        return
    if reason.startswith("Skipped: "):
        reason = reason[9:]
    report.longrepr = f"{file}:{lineno}: UNEXPECTED SKIP: {reason}"
    # Make it show up as an error in the report
    report.outcome = "error" if isinstance(report, pytest.TestReport) else "failed"


@pytest.fixture(scope="function")
def eyetrack_cal():
    """Create a toy calibration instance."""
    screen_size = (0.4, 0.225)  # width, height in meters
    screen_resolution = (1920, 1080)
    screen_distance = 0.7  # meters
    onset = 0
    model = "HV9"
    eye = "R"
    avg_error = 0.5
    max_error = 1.0
    positions = np.zeros((9, 2))
    offsets = np.zeros((9,))
    gaze = np.zeros((9, 2))
    cal = mne.preprocessing.eyetracking.Calibration(
        screen_size=screen_size,
        screen_distance=screen_distance,
        screen_resolution=screen_resolution,
        eye=eye,
        model=model,
        positions=positions,
        offsets=offsets,
        gaze=gaze,
        onset=onset,
        avg_error=avg_error,
        max_error=max_error,
    )
    return cal


@pytest.fixture(scope="function")
def eyetrack_raw():
    """Create a toy raw instance with eyetracking channels."""
    # simulate a steady fixation at the center pixel of a 1920x1080 resolution screen
    shape = (1, 100)  # x or y, time
    data = np.vstack([np.full(shape, 960), np.full(shape, 540), np.full(shape, 0)])

    info = info = mne.create_info(
        ch_names=["xpos", "ypos", "pupil"], sfreq=100, ch_types="eyegaze"
    )
    more_info = dict(
        xpos=("eyegaze", "px", "right", "x"),
        ypos=("eyegaze", "px", "right", "y"),
        pupil=("pupil", "au", "right"),
    )
    raw = mne.io.RawArray(data, info)
    raw = mne.preprocessing.eyetracking.set_channel_types_eyetrack(raw, more_info)
    return raw
