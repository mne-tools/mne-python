# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from distutils.version import LooseVersion
import gc
import os
import os.path as op
from pathlib import Path
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
from mne.utils import _pl, _assert_no_instances

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
subjects_dir = op.join(test_path, 'subjects')
fname_cov = op.join(s_path, 'sample_audvis_trunc-cov.fif')
fname_trans = op.join(s_path, 'sample_audvis_trunc-trans.fif')


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
    always:.*get_data.* is deprecated in favor of.*:DeprecationWarning
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


@pytest.fixture(scope="module", params=[
    "mayavi",
    "pyvista",
])
def backend_name(request):
    """Get the backend name."""
    yield request.param


@pytest.fixture
def renderer(backend_name, garbage_collect):
    """Yield the 3D backends."""
    from mne.viz.backends.renderer import _use_test_3d_backend
    _check_skip_backend(backend_name)
    with _use_test_3d_backend(backend_name):
        from mne.viz.backends import renderer
        yield renderer
        renderer.backend._close_all()


@pytest.fixture
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


@pytest.fixture
def renderer_interactive(backend_name_interactive):
    """Yield the 3D backends."""
    from mne.viz.backends.renderer import _use_test_3d_backend
    _check_skip_backend(backend_name_interactive)
    with _use_test_3d_backend(backend_name_interactive, interactive=True):
        from mne.viz.backends import renderer
        yield renderer
        renderer.backend._close_all()


def _check_skip_backend(name):
    from mne.viz.backends.tests._utils import (has_mayavi, has_pyvista,
                                               has_pyqt5, has_imageio_ffmpeg)
    if name == 'mayavi':
        if not has_mayavi():
            pytest.skip("Test skipped, requires mayavi.")
    elif name == 'pyvista':
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")
        if not has_imageio_ffmpeg():
            pytest.skip("Test skipped, requires imageio-ffmpeg")
    if not has_pyqt5():
        pytest.skip("Test skipped, requires PyQt5.")


@pytest.fixture()
def renderer_notebook():
    """Verify that pytest_notebook is installed."""
    from mne.viz.backends import renderer
    with renderer._use_test_3d_backend('notebook'):
        yield renderer


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
def subjects_dir_tmp(tmpdir):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ('sample', 'fsaverage'):
        shutil.copytree(op.join(subjects_dir, key), str(tmpdir.join(key)))
    return str(tmpdir)


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
    raise AssertionError('Test should not download')


@pytest.fixture(scope='function')
def download_is_error(monkeypatch):
    """Prevent downloading by raising an error when it's attempted."""
    monkeypatch.setattr(mne.utils.fetching, '_get_http', _fail)


@pytest.fixture()
def brain_gc(request):
    """Ensure that brain can be properly garbage collected."""
    keys = ('renderer_interactive', 'renderer', 'renderer_notebook')
    assert set(request.fixturenames) & set(keys) != set()
    for key in keys:
        if key in request.fixturenames:
            is_pv = request.getfixturevalue(key)._get_3d_backend() == 'pyvista'
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
    _assert_no_instances(Brain, 'before')
    ignore = set(id(o) for o in gc.get_objects())
    yield
    close_func()
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
        parts = parts[:4 if parts[:2] == ('mne', 'viz') else 3]
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
