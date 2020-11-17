# -*- coding: utf-8 -*-
"""Testing functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from contextlib import contextmanager
from distutils.version import LooseVersion
from functools import partial, wraps
import os
import inspect
from io import StringIO
from shutil import rmtree
import sys
import tempfile
import traceback
from unittest import SkipTest
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy import linalg

from ._logging import warn, ClosingStringIO
from .numerics import object_diff


def nottest(f):
    """Mark a function as not a test (decorator)."""
    f.__test__ = False
    return f


def _explain_exception(start=-1, stop=None, prefix='> '):
    """Explain an exception."""
    # start=-1 means "only the most recent caller"
    etype, value, tb = sys.exc_info()
    string = traceback.format_list(traceback.extract_tb(tb)[start:stop])
    string = (''.join(string).split('\n') +
              traceback.format_exception_only(etype, value))
    string = ':\n' + prefix + ('\n' + prefix).join(string)
    return string


class _TempDir(str):
    """Create and auto-destroy temp dir.

    This is designed to be used with testing modules. Instances should be
    defined inside test functions. Instances defined at module level can not
    guarantee proper destruction of the temporary directory.

    When used at module level, the current use of the __del__() method for
    cleanup can fail because the rmtree function may be cleaned up before this
    object (an alternative could be using the atexit module instead).
    """

    def __new__(self):  # noqa: D105
        new = str.__new__(self, tempfile.mkdtemp(prefix='tmp_mne_tempdir_'))
        return new

    def __init__(self):  # noqa: D102
        self._path = self.__str__()

    def __del__(self):  # noqa: D105
        rmtree(self._path, ignore_errors=True)


def requires_nibabel():
    """Wrap to requires_module with a function call (fewer lines to change)."""
    return partial(requires_module, name='nibabel')


def requires_dipy():
    """Check for dipy."""
    import pytest
    # for some strange reason on CIs we cane get:
    #
    #     can get weird ImportError: dlopen: cannot load any more object
    #     with static TLS
    #
    # so let's import everything in the decorator.
    try:
        from dipy.align import imaffine, imwarp, metrics, transforms  # noqa, analysis:ignore
        from dipy.align.reslice import reslice  # noqa, analysis:ignore
        from dipy.align.imaffine import AffineMap  # noqa, analysis:ignore
        from dipy.align.imwarp import DiffeomorphicMap  # noqa, analysis:ignore
    except Exception:
        have = False
    else:
        have = True
    return pytest.mark.skipif(not have, reason='Requires dipy >= 0.10.1')


def requires_version(library, min_version='0.0'):
    """Check for a library version."""
    import pytest
    return pytest.mark.skipif(not check_version(library, min_version),
                              reason=('Requires %s version >= %s'
                                      % (library, min_version)))


def requires_module(function, name, call=None):
    """Skip a test if package is not available (decorator)."""
    import pytest
    call = ('import %s' % name) if call is None else call
    reason = 'Test %s skipped, requires %s.' % (function.__name__, name)
    try:
        exec(call) in globals(), locals()
    except Exception as exc:
        if len(str(exc)) > 0 and str(exc) != 'No module named %s' % name:
            reason += ' Got exception (%s)' % (exc,)
        skip = True
    else:
        skip = False
    return pytest.mark.skipif(skip, reason=reason)(function)


_pandas_call = """
import pandas
version = LooseVersion(pandas.__version__)
if version < '0.8.0':
    raise ImportError
"""

_mayavi_call = """
with warnings.catch_warnings(record=True):  # traits
    from mayavi import mlab
"""

_mne_call = """
if not has_mne_c():
    raise ImportError
"""

_fs_call = """
if not has_freesurfer():
    raise ImportError
"""

_n2ft_call = """
if 'NEUROMAG2FT_ROOT' not in os.environ:
    raise ImportError
"""

requires_pandas = partial(requires_module, name='pandas', call=_pandas_call)
requires_pylsl = partial(requires_module, name='pylsl')
requires_sklearn = partial(requires_module, name='sklearn')
requires_mayavi = partial(requires_module, name='mayavi', call=_mayavi_call)
requires_mne = partial(requires_module, name='MNE-C', call=_mne_call)


def requires_freesurfer(arg):
    """Require Freesurfer."""
    if isinstance(arg, str):
        # Calling as  @requires_freesurfer('progname'): return decorator
        # after checking for progname existence
        call = """
from . import run_subprocess
run_subprocess([%r, '--version'])
""" % (arg,)
        return partial(
            requires_module, name='Freesurfer (%s)' % (arg,), call=call)
    else:
        # Calling directly as @requires_freesurfer: return decorated function
        # and just check env var existence
        return requires_module(arg, name='Freesurfer', call=_fs_call)


requires_neuromag2ft = partial(requires_module, name='neuromag2ft',
                               call=_n2ft_call)

requires_vtk = partial(requires_module, name='vtk')
requires_pysurfer = partial(requires_module, name='PySurfer',
                            call="""import warnings
with warnings.catch_warnings(record=True):
    from surfer import Brain""")
requires_good_network = partial(
    requires_module, name='good network connection',
    call='if int(os.environ.get("MNE_SKIP_NETWORK_TESTS", 0)):\n'
         '    raise ImportError')
requires_nitime = partial(requires_module, name='nitime')
requires_h5py = partial(requires_module, name='h5py')


def requires_numpydoc(func):
    """Decorate tests that need numpydoc."""
    return requires_version('numpydoc', '1.0')(func)  # validate needs 1.0


def check_version(library, min_version):
    r"""Check minimum library version required.

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\d+ | [a-z]+ | \.)'``. Can also be empty to skip version
        check (just check for library presence).

    Returns
    -------
    ok : bool
        True if the library exists with at least the specified version.
    """
    ok = True
    try:
        library = __import__(library)
    except ImportError:
        ok = False
    else:
        if min_version:
            this_version = LooseVersion(
                getattr(library, '__version__', '0.0').lstrip('v'))
            if this_version < min_version:
                ok = False
    return ok


def _check_mayavi_version(min_version='4.3.0'):
    """Check mayavi version."""
    if not check_version('mayavi', min_version):
        raise RuntimeError("Need mayavi >= %s" % min_version)


def _import_mlab():
    """Quietly import mlab."""
    with warnings.catch_warnings(record=True):
        from mayavi import mlab
    return mlab


@contextmanager
def traits_test_context():
    """Context to raise errors in trait handlers."""
    from traits.api import push_exception_handler

    push_exception_handler(reraise_exceptions=True)
    try:
        yield
    finally:
        push_exception_handler(reraise_exceptions=False)


def traits_test(test_func):
    """Raise errors in trait handlers (decorator)."""
    @wraps(test_func)
    def dec(*args, **kwargs):
        with traits_test_context():
            return test_func(*args, **kwargs)
    return dec


@nottest
def run_tests_if_main():
    """Run tests in a given file if it is run as a script."""
    local_vars = inspect.currentframe().f_back.f_locals
    if local_vars.get('__name__', '') != '__main__':
        return
    import pytest
    code = pytest.main([local_vars['__file__'], '-v'])
    if code:
        raise AssertionError('pytest finished with errors (%d)' % (code,))


def run_command_if_main():
    """Run a given command if it's __main__."""
    local_vars = inspect.currentframe().f_back.f_locals
    if local_vars.get('__name__', '') == '__main__':
        local_vars['run']()


class ArgvSetter(object):
    """Temporarily set sys.argv."""

    def __init__(self, args=(), disable_stdout=True,
                 disable_stderr=True):  # noqa: D102
        self.argv = list(('python',) + args)
        self.stdout = ClosingStringIO() if disable_stdout else sys.stdout
        self.stderr = ClosingStringIO() if disable_stderr else sys.stderr

    def __enter__(self):  # noqa: D105
        self.orig_argv = sys.argv
        sys.argv = self.argv
        self.orig_stdout = sys.stdout
        sys.stdout = self.stdout
        self.orig_stderr = sys.stderr
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):  # noqa: D105
        sys.argv = self.orig_argv
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr


class SilenceStdout(object):
    """Silence stdout."""

    def __init__(self, close=True):
        self.close = close

    def __enter__(self):  # noqa: D105
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        return sys.stdout

    def __exit__(self, *args):  # noqa: D105
        if self.close:
            sys.stdout.close()
        sys.stdout = self.stdout


def has_nibabel():
    """Determine if nibabel is installed.

    Returns
    -------
    has : bool
        True if the user has nibabel.
    """
    try:
        import nibabel  # noqa
    except ImportError:
        return False
    else:
        return True


def has_mne_c():
    """Check for MNE-C."""
    return 'MNE_ROOT' in os.environ


def has_freesurfer():
    """Check for Freesurfer."""
    return 'FREESURFER_HOME' in os.environ


def buggy_mkl_svd(function):
    """Decorate tests that make calls to SVD and intermittently fail."""
    @wraps(function)
    def dec(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except np.linalg.LinAlgError as exp:
            if 'SVD did not converge' in str(exp):
                msg = 'Intel MKL SVD convergence error detected, skipping test'
                warn(msg)
                raise SkipTest(msg)
            raise
    return dec


def assert_and_remove_boundary_annot(annotations, n=1):
    """Assert that there are boundary annotations and remove them."""
    from ..io.base import BaseRaw
    if isinstance(annotations, BaseRaw):  # allow either input
        annotations = annotations.annotations
    for key in ('EDGE', 'BAD'):
        idx = np.where(annotations.description == '%s boundary' % key)[0]
        assert len(idx) == n
        annotations.delete(idx)


def assert_object_equal(a, b):
    """Assert two objects are equal."""
    d = object_diff(a, b)
    assert d == '', d


def _raw_annot(meas_date, orig_time):
    from .. import Annotations, create_info
    from ..annotations import _handle_meas_date
    from ..io import RawArray
    info = create_info(ch_names=10, sfreq=10.)
    raw = RawArray(data=np.empty((10, 10)), info=info, first_samp=10)
    if meas_date is not None:
        meas_date = _handle_meas_date(meas_date)
    raw.info['meas_date'] = meas_date
    raw.info._check_consistency()
    annot = Annotations([.5], [.2], ['dummy'], orig_time)
    raw.set_annotations(annotations=annot)
    return raw


def _get_data(x, ch_idx):
    """Get the (n_ch, n_times) data array."""
    from ..evoked import Evoked
    from ..io import BaseRaw
    if isinstance(x, BaseRaw):
        return x[ch_idx][0]
    elif isinstance(x, Evoked):
        return x.data[ch_idx]


def _check_snr(actual, desired, picks, min_tol, med_tol, msg, kind='MEG'):
    """Check the SNR of a set of channels."""
    actual_data = _get_data(actual, picks)
    desired_data = _get_data(desired, picks)
    bench_rms = np.sqrt(np.mean(desired_data * desired_data, axis=1))
    error = actual_data - desired_data
    error_rms = np.sqrt(np.mean(error * error, axis=1))
    np.clip(error_rms, 1e-60, np.inf, out=error_rms)  # avoid division by zero
    snrs = bench_rms / error_rms
    # min tol
    snr = snrs.min()
    bad_count = (snrs < min_tol).sum()
    msg = ' (%s)' % msg if msg != '' else msg
    assert bad_count == 0, ('SNR (worst %0.2f) < %0.2f for %s/%s '
                            'channels%s' % (snr, min_tol, bad_count,
                                            len(picks), msg))
    # median tol
    snr = np.median(snrs)
    assert snr >= med_tol, ('%s SNR median %0.2f < %0.2f%s'
                            % (kind, snr, med_tol, msg))


def assert_meg_snr(actual, desired, min_tol, med_tol=500., chpi_med_tol=500.,
                   msg=None):
    """Assert channel SNR of a certain level.

    Mostly useful for operations like Maxwell filtering that modify
    MEG channels while leaving EEG and others intact.
    """
    from ..io.pick import pick_types
    picks = pick_types(desired.info, meg=True, exclude=[])
    picks_desired = pick_types(desired.info, meg=True, exclude=[])
    assert_array_equal(picks, picks_desired, err_msg='MEG pick mismatch')
    chpis = pick_types(actual.info, meg=False, chpi=True, exclude=[])
    chpis_desired = pick_types(desired.info, meg=False, chpi=True, exclude=[])
    if chpi_med_tol is not None:
        assert_array_equal(chpis, chpis_desired, err_msg='cHPI pick mismatch')
    others = np.setdiff1d(np.arange(len(actual.ch_names)),
                          np.concatenate([picks, chpis]))
    others_desired = np.setdiff1d(np.arange(len(desired.ch_names)),
                                  np.concatenate([picks_desired,
                                                  chpis_desired]))
    assert_array_equal(others, others_desired, err_msg='Other pick mismatch')
    if len(others) > 0:  # if non-MEG channels present
        assert_allclose(_get_data(actual, others),
                        _get_data(desired, others), atol=1e-11, rtol=1e-5,
                        err_msg='non-MEG channel mismatch')
    _check_snr(actual, desired, picks, min_tol, med_tol, msg, kind='MEG')
    if chpi_med_tol is not None and len(chpis) > 0:
        _check_snr(actual, desired, chpis, 0., chpi_med_tol, msg, kind='cHPI')


def assert_snr(actual, desired, tol):
    """Assert actual and desired arrays are within some SNR tolerance."""
    with np.errstate(divide='ignore'):  # allow infinite
        snr = (linalg.norm(desired, ord='fro') /
               linalg.norm(desired - actual, ord='fro'))
    assert snr >= tol, '%f < %f' % (snr, tol)


def assert_stcs_equal(stc1, stc2):
    """Check that two STC are equal."""
    assert_allclose(stc1.times, stc2.times)
    assert_allclose(stc1.data, stc2.data)
    assert_array_equal(stc1.vertices[0], stc2.vertices[0])
    assert_array_equal(stc1.vertices[1], stc2.vertices[1])
    assert_allclose(stc1.tmin, stc2.tmin)
    assert_allclose(stc1.tstep, stc2.tstep)


def _dig_sort_key(dig):
    """Sort dig keys."""
    return (dig['kind'], dig['ident'])


def assert_dig_allclose(info_py, info_bin, limit=None):
    """Assert dig allclose."""
    from ..bem import fit_sphere_to_headshape
    from ..io.constants import FIFF
    from ..io.meas_info import Info
    from ..channels.montage import DigMontage
    # test dig positions
    dig_py, dig_bin = info_py, info_bin
    if isinstance(dig_py, Info):
        assert isinstance(dig_bin, Info)
        dig_py, dig_bin = dig_py['dig'], dig_bin['dig']
    else:
        assert isinstance(dig_bin, DigMontage)
        assert isinstance(dig_py, DigMontage)
        dig_py, dig_bin = dig_py.dig, dig_bin.dig
        info_py = info_bin = None
    assert isinstance(dig_py, list)
    assert isinstance(dig_bin, list)
    dig_py = sorted(dig_py, key=_dig_sort_key)
    dig_bin = sorted(dig_bin, key=_dig_sort_key)
    assert len(dig_py) == len(dig_bin)
    for ii, (d_py, d_bin) in enumerate(zip(dig_py[:limit], dig_bin[:limit])):
        for key in ('ident', 'kind', 'coord_frame'):
            assert d_py[key] == d_bin[key], key
        assert_allclose(d_py['r'], d_bin['r'], rtol=1e-5, atol=1e-5,
                        err_msg='Failure on %s:\n%s\n%s'
                        % (ii, d_py['r'], d_bin['r']))
    if any(d['kind'] == FIFF.FIFFV_POINT_EXTRA for d in dig_py) and \
            info_py is not None:
        r_bin, o_head_bin, o_dev_bin = fit_sphere_to_headshape(
            info_bin, units='m', verbose='error')
        r_py, o_head_py, o_dev_py = fit_sphere_to_headshape(
            info_py, units='m', verbose='error')
        assert_allclose(r_py, r_bin, atol=1e-6)
        assert_allclose(o_dev_py, o_dev_bin, rtol=1e-5, atol=1e-6)
        assert_allclose(o_head_py, o_head_bin, rtol=1e-5, atol=1e-6)


@contextmanager
def modified_env(**d):
    """Use a modified os.environ with temporarily replaced key/value pairs.

    Parameters
    ----------
    **kwargs : dict
        The key/value pairs of environment variables to replace.
    """
    orig_env = dict()
    for key, val in d.items():
        orig_env[key] = os.getenv(key)
        if val is not None:
            assert isinstance(val, str)
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]
    try:
        yield
    finally:
        for key, val in orig_env.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]


def _click_ch_name(fig, ch_index=0, button=1):
    """Click on a channel name in a raw/epochs/ICA browse-style plot."""
    from ..viz.utils import _fake_click
    fig.canvas.draw()
    x, y = fig.mne.ax_main.get_yticklabels()[ch_index].get_position()
    xrange = np.diff(fig.mne.ax_main.get_xlim())[0]
    _fake_click(fig, fig.mne.ax_main, (x - xrange / 50, y),
                xform='data', button=button)


def _close_event(fig):
    """Force calling of the MPL figure close event."""
    # XXX workaround: plt.close() doesn't spawn close_event on Agg backend
    # (check MPL github issue #18609; scheduled to be fixed by MPL 3.4)
    try:
        fig.canvas.close_event()
    except ValueError:  # old mpl with Qt
        pass  # pragma: no cover
