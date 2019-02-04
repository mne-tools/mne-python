# -*- coding: utf-8 -*-
"""The check functions."""
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
import time
import traceback
from unittest import SkipTest
import warnings

import numpy as np

from ._logging import warn


def _memory_usage(*args, **kwargs):
    if isinstance(args[0], tuple):
        args[0][0](*args[0][1], **args[0][2])
    elif not isinstance(args[0], int):  # can be -1 for current use
        args[0]()
    return [-1]


try:
    from memory_profiler import memory_usage
except ImportError:
    memory_usage = _memory_usage


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


def requires_nibabel(vox2ras_tkr=False):
    """Check for nibabel."""
    import pytest
    extra = ' with vox2ras_tkr support' if vox2ras_tkr else ''
    return pytest.mark.skipif(not has_nibabel(vox2ras_tkr),
                              reason='Requires nibabel%s' % extra)


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

_sklearn_call = """
required_version = '0.14'
import sklearn
version = LooseVersion(sklearn.__version__)
if version < required_version:
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

_fs_or_ni_call = """
if not has_nibabel() and not has_freesurfer():
    raise ImportError
"""

requires_pandas = partial(requires_module, name='pandas', call=_pandas_call)
requires_sklearn = partial(requires_module, name='sklearn', call=_sklearn_call)
requires_mayavi = partial(requires_module, name='mayavi', call=_mayavi_call)
requires_mne = partial(requires_module, name='MNE-C', call=_mne_call)
requires_freesurfer = partial(requires_module, name='Freesurfer',
                              call=_fs_call)
requires_neuromag2ft = partial(requires_module, name='neuromag2ft',
                               call=_n2ft_call)
requires_fs_or_nibabel = partial(requires_module, name='nibabel or Freesurfer',
                                 call=_fs_or_ni_call)

requires_tvtk = partial(requires_module, name='TVTK',
                        call='from tvtk.api import tvtk')
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
requires_numpydoc = partial(requires_module, name='numpydoc')


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
            this_version = LooseVersion(library.__version__.lstrip('v'))
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
def run_tests_if_main(measure_mem=False):
    """Run tests in a given file if it is run as a script."""
    local_vars = inspect.currentframe().f_back.f_locals
    if not local_vars.get('__name__', '') == '__main__':
        return
    # we are in a "__main__"
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:
        pass
    with warnings.catch_warnings(record=True):  # memory_usage internal dep.
        mem = int(round(max(memory_usage(-1)))) if measure_mem else -1
    if mem >= 0:
        print('Memory consumption after import: %s' % mem)
    t0 = time.time()
    peak_mem, peak_name = mem, 'import'
    max_elapsed, elapsed_name = 0, 'N/A'
    count = 0
    for name in sorted(list(local_vars.keys()), key=lambda x: x.lower()):
        val = local_vars[name]
        if name.startswith('_'):
            continue
        elif callable(val) and name.startswith('test'):
            count += 1
            doc = val.__doc__.strip() if val.__doc__ else name
            sys.stdout.write('%s ... ' % doc)
            sys.stdout.flush()
            try:
                t1 = time.time()
                if measure_mem:
                    with warnings.catch_warnings(record=True):  # dep warn
                        mem = int(round(max(memory_usage((val, (), {})))))
                else:
                    val()
                    mem = -1
                if mem >= peak_mem:
                    peak_mem, peak_name = mem, name
                mem = (', mem: %s MB' % mem) if mem >= 0 else ''
                elapsed = int(round(time.time() - t1))
                if elapsed >= max_elapsed:
                    max_elapsed, elapsed_name = elapsed, name
                sys.stdout.write('time: %0.3f sec%s\n' % (elapsed, mem))
                sys.stdout.flush()
            except Exception as err:
                if 'skiptest' in err.__class__.__name__.lower():
                    sys.stdout.write('SKIP (%s)\n' % str(err))
                    sys.stdout.flush()
                else:
                    raise
    elapsed = int(round(time.time() - t0))
    sys.stdout.write('Total: %s tests\n• %0.3f sec (%0.3f sec for %s)\n• '
                     'Peak memory %s MB (%s)\n'
                     % (count, elapsed, max_elapsed, elapsed_name, peak_mem,
                        peak_name))


class ArgvSetter(object):
    """Temporarily set sys.argv."""

    def __init__(self, args=(), disable_stdout=True,
                 disable_stderr=True):  # noqa: D102
        self.argv = list(('python',) + args)
        self.stdout = StringIO() if disable_stdout else sys.stdout
        self.stderr = StringIO() if disable_stderr else sys.stderr

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

    def __enter__(self):  # noqa: D105
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        return self

    def __exit__(self, *args):  # noqa: D105
        sys.stdout = self.stdout


def has_nibabel(vox2ras_tkr=False):
    """Determine if nibabel is installed.

    Parameters
    ----------
    vox2ras_tkr : bool
        If True, require nibabel has vox2ras_tkr support.

    Returns
    -------
    has : bool
        True if the user has nibabel.
    """
    try:
        import nibabel
        out = True
        if vox2ras_tkr:  # we need MGHHeader to have vox2ras_tkr param
            out = (getattr(getattr(getattr(nibabel, 'MGHImage', 0),
                                   'header_class', 0),
                           'get_vox2ras_tkr', None) is not None)
        return out
    except ImportError:
        return False


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
