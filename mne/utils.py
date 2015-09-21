# -*- coding: utf-8 -*-
"""Some utility functions"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import warnings
import logging
import time
from distutils.version import LooseVersion
import os
import os.path as op
from functools import wraps
import inspect
from string import Formatter
import subprocess
import sys
import tempfile
import shutil
from shutil import rmtree
from math import log, ceil
import json
import ftplib
import hashlib
from functools import partial
import atexit

import numpy as np
import scipy
from scipy import linalg, sparse


from .externals.six.moves import urllib
from .externals.six import string_types, StringIO, BytesIO
from .externals.decorator import decorator

from .fixes import isclose

logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


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
    """Decorator to mark a function as not a test"""
    f.__test__ = False
    return f


###############################################################################
# RANDOM UTILITIES

def _sort_keys(x):
    """Sort and return keys of dict"""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


def object_hash(x, h=None):
    """Hash a reasonable python object

    Parameters
    ----------
    x : object
        Object to hash. Can be anything comprised of nested versions of:
        {dict, list, tuple, ndarray, str, bytes, float, int, None}.
    h : hashlib HASH object | None
        Optional, object to add the hash to. None creates an MD5 hash.

    Returns
    -------
    digest : int
        The digest resulting from the hash.
    """
    if h is None:
        h = hashlib.md5()
    if isinstance(x, dict):
        keys = _sort_keys(x)
        for key in keys:
            object_hash(key, h)
            object_hash(x[key], h)
    elif isinstance(x, (list, tuple)):
        h.update(str(type(x)).encode('utf-8'))
        for xx in x:
            object_hash(xx, h)
    elif isinstance(x, bytes):
        # must come before "str" below
        h.update(x)
    elif isinstance(x, (string_types, float, int, type(None))):
        h.update(str(type(x)).encode('utf-8'))
        h.update(str(x).encode('utf-8'))
    elif isinstance(x, np.ndarray):
        x = np.asarray(x)
        h.update(str(x.shape).encode('utf-8'))
        h.update(str(x.dtype).encode('utf-8'))
        h.update(x.tostring())
    else:
        raise RuntimeError('unsupported type: %s (%s)' % (type(x), x))
    return int(h.hexdigest(), 16)


def object_diff(a, b, pre=''):
    """Compute all differences between two python variables

    Parameters
    ----------
    a : object
        Currently supported: dict, list, tuple, ndarray, int, str, bytes,
        float, StringIO, BytesIO.
    b : object
        Must be same type as x1.
    pre : str
        String to prepend to each line.

    Returns
    -------
    diffs : str
        A string representation of the differences.
    """
    out = ''
    if type(a) != type(b):
        out += pre + ' type mismatch (%s, %s)\n' % (type(a), type(b))
    elif isinstance(a, dict):
        k1s = _sort_keys(a)
        k2s = _sort_keys(b)
        m1 = set(k2s) - set(k1s)
        if len(m1):
            out += pre + ' x1 missing keys %s\n' % (m1)
        for key in k1s:
            if key not in k2s:
                out += pre + ' x2 missing key %s\n' % key
            else:
                out += object_diff(a[key], b[key], pre + 'd1[%s]' % repr(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out += pre + ' length mismatch (%s, %s)\n' % (len(a), len(b))
        else:
            for xx1, xx2 in zip(a, b):
                out += object_diff(xx1, xx2, pre='')
    elif isinstance(a, (string_types, int, float, bytes)):
        if a != b:
            out += pre + ' value mismatch (%s, %s)\n' % (a, b)
    elif a is None:
        if b is not None:
            out += pre + ' a is None, b is not (%s)\n' % (b)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            out += pre + ' array mismatch\n'
    elif isinstance(a, (StringIO, BytesIO)):
        if a.getvalue() != b.getvalue():
            out += pre + ' StringIO mismatch\n'
    elif sparse.isspmatrix(a):
        # sparsity and sparse type of b vs a already checked above by type()
        if b.shape != a.shape:
            out += pre + (' sparse matrix a and b shape mismatch'
                          '(%s vs %s)' % (a.shape, b.shape))
        else:
            c = a - b
            c.eliminate_zeros()
            if c.nnz > 0:
                out += pre + (' sparse matrix a and b differ on %s '
                              'elements' % c.nnz)
    else:
        raise RuntimeError(pre + ': unsupported type %s (%s)' % (type(a), a))
    return out


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def split_list(l, n):
    """split list in n (approx) equal pieces"""
    n = int(n)
    sz = len(l) // n
    for i in range(n - 1):
        yield l[i * sz:(i + 1) * sz]
    yield l[(n - 1) * sz:]


def create_chunks(sequence, size):
    """Generate chunks from a sequence

    Parameters
    ----------
    sequence : iterable
        Any iterable object
    size : int
        The chunksize to be returned
    """
    return (sequence[p:p + size] for p in range(0, len(sequence), size))


def sum_squared(X):
    """Compute norm of an array

    Parameters
    ----------
    X : array
        Data whose norm must be found

    Returns
    -------
    value : float
        Sum of squares of the input array X
    """
    X_flat = X.ravel(order='F' if np.isfortran(X) else 'C')
    return np.dot(X_flat, X_flat)


def check_fname(fname, filetype, endings):
    """Enforce MNE filename conventions

    Parameters
    ----------
    fname : str
        Name of the file.
    filetype : str
        Type of file. e.g., ICA, Epochs etc.
    endings : tuple
        Acceptable endings for the filename.
    """
    print_endings = ' or '.join([', '.join(endings[:-1]), endings[-1]])
    if not fname.endswith(endings):
        warnings.warn('This filename (%s) does not conform to MNE naming '
                      'conventions. All %s files should end with '
                      '%s' % (fname, filetype, print_endings))


class WrapStdOut(object):
    """Ridiculous class to work around how doctest captures stdout"""
    def __getattr__(self, name):
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        return getattr(sys.stdout, name)


class _TempDir(str):
    """Class for creating and auto-destroying temp dir

    This is designed to be used with testing modules. Instances should be
    defined inside test functions. Instances defined at module level can not
    guarantee proper destruction of the temporary directory.

    When used at module level, the current use of the __del__() method for
    cleanup can fail because the rmtree function may be cleaned up before this
    object (an alternative could be using the atexit module instead).
    """
    def __new__(self):
        new = str.__new__(self, tempfile.mkdtemp())
        return new

    def __init__(self):
        self._path = self.__str__()

    def __del__(self):
        rmtree(self._path, ignore_errors=True)


def estimate_rank(data, tol=1e-4, return_singular=False,
                  norm=True, copy=True):
    """Helper to estimate the rank of data

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    data : array
        Data to estimate the rank of (should be 2-dimensional).
    tol : float
        Tolerance for singular values to consider non-zero in
        calculating the rank. The singular values are calculated
        in this method such that independent data are expected to
        have singular value around one.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    norm : bool
        If True, data will be scaled by their estimated row-wise norm.
        Else data are assumed to be scaled. Defaults to True.
    copy : bool
        If False, values in data will be modified in-place during
        rank estimation (saves memory).

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    if copy is True:
        data = data.copy()
    if norm is True:
        norms = _compute_row_norms(data)
        data /= norms[:, np.newaxis]
    s = linalg.svd(data, compute_uv=False, overwrite_a=True)
    rank = np.sum(s >= tol)
    if return_singular is True:
        return rank, s
    else:
        return rank


def _compute_row_norms(data):
    """Compute scaling based on estimated norm"""
    norms = np.sqrt(np.sum(data ** 2, axis=1))
    norms[norms == 0] = 1.0
    return norms


def _reject_data_segments(data, reject, flat, decim, info, tstep):
    """Reject data segments using peak-to-peak amplitude
    """
    from .epochs import _is_good
    from .io.pick import channel_indices_by_type

    data_clean = np.empty_like(data)
    idx_by_type = channel_indices_by_type(info)
    step = int(ceil(tstep * info['sfreq']))
    if decim is not None:
        step = int(ceil(step / float(decim)))
    this_start = 0
    this_stop = 0
    drop_inds = []
    for first in range(0, data.shape[1], step):
        last = first + step
        data_buffer = data[:, first:last]
        if data_buffer.shape[1] < (last - first):
            break  # end of the time segment
        if _is_good(data_buffer, info['ch_names'], idx_by_type, reject,
                    flat, ignore_chs=info['bads']):
            this_stop = this_start + data_buffer.shape[1]
            data_clean[:, this_start:this_stop] = data_buffer
            this_start += data_buffer.shape[1]
        else:
            logger.info("Artifact detected in [%d, %d]" % (first, last))
            drop_inds.append((first, last))
    data = data_clean[:, :this_stop]
    if not data.any():
        raise RuntimeError('No clean segment found. Please '
                           'consider updating your rejection '
                           'thresholds.')
    return data, drop_inds


class _FormatDict(dict):
    """Helper for pformat()"""
    def __missing__(self, key):
        return "{" + key + "}"


def pformat(temp, **fmt):
    """Partially format a template string.

    Examples
    --------
    >>> pformat("{a}_{b}", a='x')
    'x_{b}'
    """
    formatter = Formatter()
    mapping = _FormatDict(fmt)
    return formatter.vformat(temp, (), mapping)


def trait_wraith(*args, **kwargs):
    # Stand in for traits to allow importing traits based modules when the
    # traits library is not installed
    return lambda x: x


###############################################################################
# DECORATORS

# Following deprecated class copied from scikit-learn

# force show of DeprecationWarning even on python 2.7
warnings.simplefilter('default')


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from mne.utils import deprecated
        >>> deprecated() # doctest: +ELLIPSIS
        <mne.utils.deprecated object at ...>

        >>> @deprecated()
        ... def some_function(): pass


    Parameters
    ----------
    extra: string
        To be added to the deprecation messages.
    """
    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    # scikit-learn will not import on all platforms b/c it can be
    # sklearn or scikits.learn, so a self-contained example is used above

    def __init__(self, extra=''):
        self.extra = extra

    def __call__(self, obj):
        """Call

        Parameters
        ----------
        obj : object
            Object to call.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


@decorator
def verbose(function, *args, **kwargs):
    """Improved verbose decorator to allow functions to override log-level

    Do not call this directly to set global verbosity level, instead use
    set_log_level().

    Parameters
    ----------
    function : function
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec : function
        The decorated function
    """
    arg_names = inspect.getargspec(function).args
    default_level = verbose_level = None
    if len(arg_names) > 0 and arg_names[0] == 'self':
        default_level = getattr(args[0], 'verbose', None)
    if 'verbose' in arg_names:
        verbose_level = args[arg_names.index('verbose')]
    elif 'verbose' in kwargs:
        verbose_level = kwargs.pop('verbose')

    # This ensures that object.method(verbose=None) will use object.verbose
    verbose_level = default_level if verbose_level is None else verbose_level

    if verbose_level is not None:
        old_level = set_log_level(verbose_level, True)
        # set it back if we get an exception
        try:
            return function(*args, **kwargs)
        finally:
            set_log_level(old_level)
    return function(*args, **kwargs)


@nottest
def slow_test(f):
    """Decorator for slow tests"""
    f.slow_test = True
    return f


@nottest
def ultra_slow_test(f):
    """Decorator for ultra slow tests"""
    f.ultra_slow_test = True
    f.slow_test = True
    return f


def has_nibabel(vox2ras_tkr=False):
    """Determine if nibabel is installed

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
    """Aux function"""
    return 'MNE_ROOT' in os.environ


def has_freesurfer():
    """Aux function"""
    return 'FREESURFER_HOME' in os.environ


def requires_nibabel(vox2ras_tkr=False):
    """Aux function"""
    extra = ' with vox2ras_tkr support' if vox2ras_tkr else ''
    return np.testing.dec.skipif(not has_nibabel(vox2ras_tkr),
                                 'Requires nibabel%s' % extra)


def requires_scipy_version(min_version):
    """Helper for testing"""
    return np.testing.dec.skipif(not check_scipy_version(min_version),
                                 'Requires scipy version >= %s' % min_version)


def requires_module(function, name, call):
    """Decorator to skip test if package is not available"""
    try:
        from nose.plugins.skip import SkipTest
    except ImportError:
        SkipTest = AssertionError

    @wraps(function)
    def dec(*args, **kwargs):
        skip = False
        try:
            exec(call) in globals(), locals()
        except Exception:
            skip = True
        if skip is True:
            raise SkipTest('Test %s skipped, requires %s'
                           % (function.__name__, name))
        return function(*args, **kwargs)
    return dec


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
from mayavi import mlab
mlab.options.backend = 'test'
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
requires_statsmodels = partial(requires_module, name='statsmodels',
                               call='import statsmodels')
requires_patsy = partial(requires_module, name='patsy',
                         call='import patsy')
requires_pysurfer = partial(requires_module, name='PySurfer',
                            call='from surfer import Brain')
requires_PIL = partial(requires_module, name='PIL',
                       call='from PIL import Image')
requires_good_network = partial(
    requires_module, name='good network connection',
    call='if int(os.environ.get("MNE_SKIP_NETWORK_TESTS", 0)):\n'
         '    raise ImportError')
requires_nitime = partial(requires_module, name='nitime',
                          call='import nitime')
requires_traits = partial(requires_module, name='traits',
                          call='import traits')
requires_h5py = partial(requires_module, name='h5py', call='import h5py')


def _check_mayavi_version(min_version='4.3.0'):
    """Raise a RuntimeError if the required version of mayavi is not available

    Parameters
    ----------
    min_version : str
        The version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``
    """
    import mayavi
    require_mayavi = LooseVersion(min_version)
    if LooseVersion(mayavi.__version__) < require_mayavi:
        raise RuntimeError("Need mayavi >= %s" % require_mayavi)


def check_sklearn_version(min_version):
    """Check minimum sklearn version required

    Parameters
    ----------
    min_version : str
        The version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``
    """
    ok = True
    try:
        import sklearn
        this_version = LooseVersion(sklearn.__version__)
        if this_version < min_version:
            ok = False
    except ImportError:
        ok = False
    return ok


def check_scipy_version(min_version):
    """Check minimum sklearn version required

    Parameters
    ----------
    min_version : str
        The version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``
    """
    this_version = LooseVersion(scipy.__version__)
    return False if this_version < min_version else True


@verbose
def run_subprocess(command, verbose=None, *args, **kwargs):
    """Run command using subprocess.Popen

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str
        Command to run as subprocess (see subprocess.Popen documentation).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to self.verbose.
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.

    Returns
    -------
    stdout : str
        Stdout returned by the process.
    stderr : str
        Stderr returned by the process.
    """
    if 'stderr' not in kwargs:
        kwargs['stderr'] = subprocess.PIPE
    if 'stdout' not in kwargs:
        kwargs['stdout'] = subprocess.PIPE

    # Check the PATH environment variable. If run_subprocess() is to be called
    # frequently this should be refactored so as to only check the path once.
    env = kwargs.get('env', os.environ)
    if any(p.startswith('~') for p in env['PATH'].split(os.pathsep)):
        msg = ("Your PATH environment variable contains at least one path "
               "starting with a tilde ('~') character. Such paths are not "
               "interpreted correctly from within Python. It is recommended "
               "that you use '$HOME' instead of '~'.")
        warnings.warn(msg)

    logger.info("Running subprocess: %s" % ' '.join(command))
    try:
        p = subprocess.Popen(command, *args, **kwargs)
    except Exception:
        logger.error('Command not found: %s' % (command[0],))
        raise
    stdout_, stderr = p.communicate()
    stdout_ = '' if stdout_ is None else stdout_.decode('utf-8')
    stderr = '' if stderr is None else stderr.decode('utf-8')

    if stdout_.strip():
        logger.info("stdout:\n%s" % stdout_)
    if stderr.strip():
        logger.info("stderr:\n%s" % stderr)

    output = (stdout_, stderr)
    if p.returncode:
        print(output)
        err_fun = subprocess.CalledProcessError.__init__
        if 'output' in inspect.getargspec(err_fun).args:
            raise subprocess.CalledProcessError(p.returncode, command, output)
        else:
            raise subprocess.CalledProcessError(p.returncode, command)

    return output


###############################################################################
# LOGGING

def set_log_level(verbose=None, return_old_level=False):
    """Convenience function for setting the logging level

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, the environment variable MNE_LOGGING_LEVEL is read, and if
        it doesn't exist, defaults to INFO.
    return_old_level : bool
        If True, return the old verbosity level.
    """
    if verbose is None:
        verbose = get_config('MNE_LOGGING_LEVEL', 'INFO')
    elif isinstance(verbose, bool):
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
    if isinstance(verbose, string_types):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    logger = logging.getLogger('mne')
    old_verbose = logger.level
    logger.setLevel(verbose)
    return (old_verbose if return_old_level else None)


def set_log_file(fname=None, output_format='%(message)s', overwrite=None):
    """Convenience function for setting the log to print to a file

    Parameters
    ----------
    fname : str, or None
        Filename of the log to print to. If None, stdout is used.
        To suppress log outputs, use set_log_level('WARN').
    output_format : str
        Format of the output messages. See the following for examples:

            https://docs.python.org/dev/howto/logging.html

        e.g., "%(asctime)s - %(levelname)s - %(message)s".
    overwrite : bool, or None
        Overwrite the log file (if it exists). Otherwise, statements
        will be appended to the log (default). None is the same as False,
        but additionally raises a warning to notify the user that log
        entries will be appended.
    """
    logger = logging.getLogger('mne')
    handlers = logger.handlers
    for h in handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
        logger.removeHandler(h)
    if fname is not None:
        if op.isfile(fname) and overwrite is None:
            warnings.warn('Log entries will be appended to the file. Use '
                          'overwrite=False to avoid this message in the '
                          'future.')
        mode = 'w' if overwrite is True else 'a'
        lh = logging.FileHandler(fname, mode=mode)
    else:
        """ we should just be able to do:
                lh = logging.StreamHandler(sys.stdout)
            but because doctests uses some magic on stdout, we have to do this:
        """
        lh = logging.StreamHandler(WrapStdOut())

    lh.setFormatter(logging.Formatter(output_format))
    # actually add the stream handler
    logger.addHandler(lh)


###############################################################################
# CONFIG / PREFS

def get_subjects_dir(subjects_dir=None, raise_error=False):
    """Safely use subjects_dir input to return SUBJECTS_DIR

    Parameters
    ----------
    subjects_dir : str | None
        If a value is provided, return subjects_dir. Otherwise, look for
        SUBJECTS_DIR config and return the result.
    raise_error : bool
        If True, raise a KeyError if no value for SUBJECTS_DIR can be found
        (instead of returning None).

    Returns
    -------
    value : str | None
        The SUBJECTS_DIR value.
    """
    if subjects_dir is None:
        subjects_dir = get_config('SUBJECTS_DIR', raise_error=raise_error)
    return subjects_dir


_temp_home_dir = None


def _get_extra_data_path(home_dir=None):
    """Get path to extra data (config, tables, etc.)"""
    global _temp_home_dir
    if home_dir is None:
        # this has been checked on OSX64, Linux64, and Win32
        if 'nt' == os.name.lower():
            home_dir = os.getenv('APPDATA')
        else:
            # This is a more robust way of getting the user's home folder on
            # Linux platforms (not sure about OSX, Unix or BSD) than checking
            # the HOME environment variable. If the user is running some sort
            # of script that isn't launched via the command line (e.g. a script
            # launched via Upstart) then the HOME environment variable will
            # not be set.
            if os.getenv('MNE_DONTWRITE_HOME', '') == 'true':
                if _temp_home_dir is None:
                    _temp_home_dir = tempfile.mkdtemp()
                    atexit.register(partial(shutil.rmtree, _temp_home_dir,
                                            ignore_errors=True))
                home_dir = _temp_home_dir
            else:
                home_dir = os.path.expanduser('~')

        if home_dir is None:
            raise ValueError('mne-python config file path could '
                             'not be determined, please report this '
                             'error to mne-python developers')

    return op.join(home_dir, '.mne')


def get_config_path(home_dir=None):
    """Get path to standard mne-python config file

    Parameters
    ----------
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.

    Returns
    -------
    config_path : str
        The path to the mne-python configuration file. On windows, this
        will be '%APPDATA%\.mne\mne-python.json'. On every other
        system, this will be ~/.mne/mne-python.json.
    """
    val = op.join(_get_extra_data_path(home_dir=home_dir),
                  'mne-python.json')
    return val


def set_cache_dir(cache_dir):
    """Set the directory to be used for temporary file storage.

    This directory is used by joblib to store memmapped arrays,
    which reduces memory requirements and speeds up parallel
    computation.

    Parameters
    ----------
    cache_dir: str or None
        Directory to use for temporary file storage. None disables
        temporary file storage.
    """
    if cache_dir is not None and not op.exists(cache_dir):
        raise IOError('Directory %s does not exist' % cache_dir)

    set_config('MNE_CACHE_DIR', cache_dir)


def set_memmap_min_size(memmap_min_size):
    """Set the minimum size for memmaping of arrays for parallel processing

    Parameters
    ----------
    memmap_min_size: str or None
        Threshold on the minimum size of arrays that triggers automated memory
        mapping for parallel processing, e.g., '1M' for 1 megabyte.
        Use None to disable memmaping of large arrays.
    """
    if memmap_min_size is not None:
        if not isinstance(memmap_min_size, string_types):
            raise ValueError('\'memmap_min_size\' has to be a string.')
        if memmap_min_size[-1] not in ['K', 'M', 'G']:
            raise ValueError('The size has to be given in kilo-, mega-, or '
                             'gigabytes, e.g., 100K, 500M, 1G.')

    set_config('MNE_MEMMAP_MIN_SIZE', memmap_min_size)


# List the known configuration values
known_config_types = [
    'MNE_BROWSE_RAW_SIZE',
    'MNE_CUDA_IGNORE_PRECISION',
    'MNE_DATA',
    'MNE_DATASETS_MEGSIM_PATH',
    'MNE_DATASETS_SAMPLE_PATH',
    'MNE_DATASETS_SOMATO_PATH',
    'MNE_DATASETS_SPM_FACE_PATH',
    'MNE_DATASETS_EEGBCI_PATH',
    'MNE_DATASETS_BRAINSTORM_PATH',
    'MNE_DATASETS_TESTING_PATH',
    'MNE_LOGGING_LEVEL',
    'MNE_USE_CUDA',
    'SUBJECTS_DIR',
    'MNE_CACHE_DIR',
    'MNE_MEMMAP_MIN_SIZE',
    'MNE_SKIP_TESTING_DATASET_TESTS',
    'MNE_DATASETS_SPM_FACE_DATASETS_TESTS'
]

# These allow for partial matches, e.g. 'MNE_STIM_CHANNEL_1' is okay key
known_config_wildcards = [
    'MNE_STIM_CHANNEL',
]


def get_config(key=None, default=None, raise_error=False, home_dir=None):
    """Read mne(-python) preference from env, then mne-python config

    Parameters
    ----------
    key : None | str
        The preference key to look for. The os evironment is searched first,
        then the mne-python config file is parsed.
        If None, all the config parameters present in the path are returned.
    default : str | None
        Value to return if the key is not found.
    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.

    Returns
    -------
    value : dict | str | None
        The preference key value.

    See Also
    --------
    set_config
    """

    if key is not None and not isinstance(key, string_types):
        raise TypeError('key must be a string')

    # first, check to see if key is in env
    if key is not None and key in os.environ:
        return os.environ[key]

    # second, look for it in mne-python config file
    config_path = get_config_path(home_dir=home_dir)
    if not op.isfile(config_path):
        key_found = False
        val = default
    else:
        with open(config_path, 'r') as fid:
            config = json.load(fid)
            if key is None:
                return config
        key_found = key in config
        val = config.get(key, default)

    if not key_found and raise_error is True:
        meth_1 = 'os.environ["%s"] = VALUE' % key
        meth_2 = 'mne.utils.set_config("%s", VALUE)' % key
        raise KeyError('Key "%s" not found in environment or in the '
                       'mne-python config file: %s '
                       'Try either:'
                       ' %s for a temporary solution, or:'
                       ' %s for a permanent one. You can also '
                       'set the environment variable before '
                       'running python.'
                       % (key, config_path, meth_1, meth_2))
    return val


def set_config(key, value, home_dir=None):
    """Set mne-python preference in config

    Parameters
    ----------
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.

    See Also
    --------
    get_config
    """
    if not isinstance(key, string_types):
        raise TypeError('key must be a string')
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    if not isinstance(value, string_types) and value is not None:
        raise TypeError('value must be a string or None')
    if key not in known_config_types and not \
            any(k in key for k in known_config_wildcards):
        warnings.warn('Setting non-standard config type: "%s"' % key)

    # Read all previous values
    config_path = get_config_path(home_dir=home_dir)
    if op.isfile(config_path):
        with open(config_path, 'r') as fid:
            config = json.load(fid)
    else:
        config = dict()
        logger.info('Attempting to create new mne-python configuration '
                    'file:\n%s' % config_path)
    if value is None:
        config.pop(key, None)
    else:
        config[key] = value

    # Write all values. This may fail if the default directory is not
    # writeable.
    directory = op.dirname(config_path)
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, 'w') as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


class ProgressBar(object):
    """Class for generating a command-line progressbar

    Parameters
    ----------
    max_value : int
        Maximum value of process (e.g. number of samples to process, bytes to
        download, etc.).
    initial_value : int
        Initial value of process, useful when resuming process from a specific
        value, defaults to 0.
    mesg : str
        Message to include at end of progress bar.
    max_chars : int
        Number of characters to use for progress bar (be sure to save some room
        for the message and % complete as well).
    progress_character : char
        Character in the progress bar that indicates the portion completed.
    spinner : bool
        Show a spinner.  Useful for long-running processes that may not
        increment the progress bar very often.  This provides the user with
        feedback that the progress has not stalled.

    Example
    -------
    >>> progress = ProgressBar(13000)
    >>> progress.update(3000) # doctest: +SKIP
    [.........                               ] 23.07692 |
    >>> progress.update(6000) # doctest: +SKIP
    [..................                      ] 46.15385 |

    >>> progress = ProgressBar(13000, spinner=True)
    >>> progress.update(3000) # doctest: +SKIP
    [.........                               ] 23.07692 |
    >>> progress.update(6000) # doctest: +SKIP
    [..................                      ] 46.15385 /
    """

    spinner_symbols = ['|', '/', '-', '\\']
    template = '\r[{0}{1}] {2:.05f} {3} {4}   '

    def __init__(self, max_value, initial_value=0, mesg='', max_chars=40,
                 progress_character='.', spinner=False, verbose_bool=True):
        self.cur_value = initial_value
        self.max_value = float(max_value)
        self.mesg = mesg
        self.max_chars = max_chars
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)
        self._do_print = verbose_bool

    def update(self, cur_value, mesg=None):
        """Update progressbar with current value of process

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            (cur_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        # Ensure floating-point division so we can get fractions of a percent
        # for the progressbar.
        self.cur_value = cur_value
        progress = min(float(self.cur_value) / self.max_value, 1.)
        num_chars = int(progress * self.max_chars)
        num_left = self.max_chars - num_chars

        # Update the message
        if mesg is not None:
            self.mesg = mesg

        # The \r tells the cursor to return to the beginning of the line rather
        # than starting a new line.  This allows us to have a progressbar-style
        # display in the console window.
        bar = self.template.format(self.progress_character * num_chars,
                                   ' ' * num_left,
                                   progress * 100,
                                   self.spinner_symbols[self.spinner_index],
                                   self.mesg)
        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        if self._do_print:
            sys.stdout.write(bar)
            sys.stdout.flush()
        # Increament the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

    def update_with_increment_value(self, increment_value, mesg=None):
        """Update progressbar with the value of the increment instead of the
        current value of process as in update()

        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            (self.cur_value + increment_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        self.cur_value += increment_value
        self.update(self.cur_value, mesg)


def _chunk_read(response, local_file, initial_size=0, verbose_bool=True):
    """Download a file chunk by chunk and show advancement

    Can also be used when resuming downloads over http.

    Parameters
    ----------
    response: urllib.response.addinfourl
        Response to the download request in order to get file size.
    local_file: file
        Hard disk file where data should be written.
    initial_size: int, optional
        If resuming, indicate the initial size of the file.

    Notes
    -----
    The chunk size will be automatically adapted based on the connection
    speed.
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    # Returns only amount left to download when resuming, not the size of the
    # entire file
    total_size = int(response.headers.get('Content-Length', '1').strip())
    total_size += initial_size

    progress = ProgressBar(total_size, initial_value=initial_size,
                           max_chars=40, spinner=True, mesg='downloading',
                           verbose_bool=verbose_bool)
    chunk_size = 8192  # 2 ** 13
    while True:
        t0 = time.time()
        chunk = response.read(chunk_size)
        dt = time.time() - t0
        if dt < 0.001:
            chunk_size *= 2
        elif dt > 0.5 and chunk_size > 8192:
            chunk_size = chunk_size // 2
        if not chunk:
            if verbose_bool:
                sys.stdout.write('\n')
                sys.stdout.flush()
            break
        _chunk_write(chunk, local_file, progress)


def _chunk_read_ftp_resume(url, temp_file_name, local_file, verbose_bool=True):
    """Resume downloading of a file from an FTP server"""
    # Adapted from: https://pypi.python.org/pypi/fileDownloader.py
    # but with changes

    parsed_url = urllib.parse.urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    server_path = parsed_url.path.replace(file_name, "")
    unquoted_server_path = urllib.parse.unquote(server_path)
    local_file_size = os.path.getsize(temp_file_name)

    data = ftplib.FTP()
    if parsed_url.port is not None:
        data.connect(parsed_url.hostname, parsed_url.port)
    else:
        data.connect(parsed_url.hostname)
    data.login()
    if len(server_path) > 1:
        data.cwd(unquoted_server_path)
    data.sendcmd("TYPE I")
    data.sendcmd("REST " + str(local_file_size))
    down_cmd = "RETR " + file_name
    file_size = data.size(file_name)
    progress = ProgressBar(file_size, initial_value=local_file_size,
                           max_chars=40, spinner=True, mesg='downloading',
                           verbose_bool=verbose_bool)

    # Callback lambda function that will be passed the downloaded data
    # chunk and will write it to file and update the progress bar
    def chunk_write(chunk):
        return _chunk_write(chunk, local_file, progress)
    data.retrbinary(down_cmd, chunk_write)
    data.close()
    sys.stdout.write('\n')
    sys.stdout.flush()


def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar"""
    local_file.write(chunk)
    progress.update_with_increment_value(len(chunk))


@verbose
def _fetch_file(url, file_name, print_destination=True, resume=True,
                hash_=None, verbose=None):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    url: string
        The url of file to be downloaded.
    file_name: string
        Name, along with the path, of where downloaded file will be saved.
    print_destination: bool, optional
        If true, destination of where file was saved will be printed after
        download finishes.
    resume: bool, optional
        If true, try to resume partially downloaded files.
    hash_ : str | None
        The hash of the file to check. If None, no checking is
        performed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py
    if hash_ is not None and (not isinstance(hash_, string_types) or
                              len(hash_) != 32):
        raise ValueError('Bad hash value given, should be a 32-character '
                         'string:\n%s' % (hash_,))
    temp_file_name = file_name + ".part"
    local_file = None
    initial_size = 0
    verbose_bool = (logger.level <= 20)  # 20 is info
    try:
        # Checking file size and displaying it alongside the download url
        u = urllib.request.urlopen(url, timeout=10.)
        try:
            file_size = int(u.headers.get('Content-Length', '1').strip())
        finally:
            u.close()
            del u
        logger.info('Downloading data from %s (%s)\n'
                    % (url, sizeof_fmt(file_size)))
        # Downloading data
        if resume and os.path.exists(temp_file_name):
            local_file = open(temp_file_name, "ab")
            # Resuming HTTP and FTP downloads requires different procedures
            scheme = urllib.parse.urlparse(url).scheme
            if scheme in ('http', 'https'):
                local_file_size = os.path.getsize(temp_file_name)
                # If the file exists, then only download the remainder
                req = urllib.request.Request(url)
                req.headers["Range"] = "bytes=%s-" % local_file_size
                try:
                    data = urllib.request.urlopen(req)
                except Exception:
                    # There is a problem that may be due to resuming, some
                    # servers may not support the "Range" header. Switch back
                    # to complete download method
                    logger.info('Resuming download failed. Attempting to '
                                'restart downloading the entire file.')
                    local_file.close()
                    _fetch_file(url, file_name, resume=False)
                else:
                    _chunk_read(data, local_file, initial_size=local_file_size,
                                verbose_bool=verbose_bool)
                    data.close()
                    del data  # should auto-close
            else:
                _chunk_read_ftp_resume(url, temp_file_name, local_file,
                                       verbose_bool=verbose_bool)
        else:
            local_file = open(temp_file_name, "wb")
            data = urllib.request.urlopen(url)
            try:
                _chunk_read(data, local_file, initial_size=initial_size,
                            verbose_bool=verbose_bool)
            finally:
                data.close()
                del data  # should auto-close
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        # check md5sum
        if hash_ is not None:
            logger.info('Verifying download hash.')
            md5 = md5sum(temp_file_name)
            if hash_ != md5:
                raise RuntimeError('Hash mismatch for downloaded file %s, '
                                   'expected %s but got %s'
                                   % (temp_file_name, hash_, md5))
        shutil.move(temp_file_name, file_name)
        if print_destination is True:
            logger.info('File saved as %s.\n' % file_name)
    except Exception as e:
        logger.error('Error while fetching file %s.'
                     ' Dataset fetching aborted.' % url)
        logger.error("Error: %s", e)
        raise
    finally:
        if local_file is not None:
            if not local_file.closed:
                local_file.close()


def sizeof_fmt(num):
    """Turn number of bytes into human-readable str"""
    units = ['bytes', 'kB', 'MB', 'GB', 'TB', 'PB']
    decimals = [0, 0, 1, 2, 2, 2]
    """Human friendly file size"""
    if num > 1:
        exponent = min(int(log(num, 1024)), len(units) - 1)
        quotient = float(num) / 1024 ** exponent
        unit = units[exponent]
        num_decimals = decimals[exponent]
        format_string = '{0:.%sf} {1}' % (num_decimals)
        return format_string.format(quotient, unit)
    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'


def _url_to_local_path(url, path):
    """Mirror a url path in a local destination (keeping folder structure)"""
    destination = urllib.parse.urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != '/':
        raise ValueError('Invalid URL')
    destination = os.path.join(path,
                               urllib.request.url2pathname(destination)[1:])
    return destination


def _get_stim_channel(stim_channel, info):
    """Helper to determine the appropriate stim_channel

    First, 'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2', etc.
    are read. If these are not found, it will fall back to 'STI 014' if
    present, then fall back to the first channel of type 'stim', if present.

    Parameters
    ----------
    stim_channel : str | list of str | None
        The stim channel selected by the user.
    info : instance of Info
        An information structure containing information about the channels.

    Returns
    -------
    stim_channel : str | list of str
        The name of the stim channel(s) to use
    """
    if stim_channel is not None:
        if not isinstance(stim_channel, list):
            if not isinstance(stim_channel, string_types):
                raise TypeError('stim_channel must be a str, list, or None')
            stim_channel = [stim_channel]
        if not all(isinstance(s, string_types) for s in stim_channel):
            raise TypeError('stim_channel list must contain all strings')
        return stim_channel

    stim_channel = list()
    ch_count = 0
    ch = get_config('MNE_STIM_CHANNEL')
    while(ch is not None and ch in info['ch_names']):
        stim_channel.append(ch)
        ch_count += 1
        ch = get_config('MNE_STIM_CHANNEL_%d' % ch_count)
    if ch_count > 0:
        return stim_channel

    if 'STI 014' in info['ch_names']:
        return ['STI 014']

    from .io.pick import pick_types
    stim_channel = pick_types(info, meg=False, ref_meg=False, stim=True)
    if len(stim_channel) > 0:
        stim_channel = [info['ch_names'][ch_] for ch_ in stim_channel]
        return stim_channel

    raise ValueError("No stim channels found. Consider specifying them "
                     "manually using the 'stim_channel' parameter.")


def _check_fname(fname, overwrite):
    """Helper to check for file existence"""
    if not isinstance(fname, string_types):
        raise TypeError('file name is not a string')
    if op.isfile(fname):
        if not overwrite:
            raise IOError('Destination file exists. Please use option '
                          '"overwrite=True" to force overwriting.')
        else:
            logger.info('Overwriting existing file.')


def _check_subject(class_subject, input_subject, raise_error=True):
    """Helper to get subject name from class"""
    if input_subject is not None:
        if not isinstance(input_subject, string_types):
            raise ValueError('subject input must be a string')
        else:
            return input_subject
    elif class_subject is not None:
        if not isinstance(class_subject, string_types):
            raise ValueError('Neither subject input nor class subject '
                             'attribute was a string')
        else:
            return class_subject
    else:
        if raise_error is True:
            raise ValueError('Neither subject input nor class subject '
                             'attribute was a string')
        return None


def _check_pandas_installed():
    """Aux function"""
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise RuntimeError('For this method to work the Pandas library is'
                           ' required.')


def _check_pandas_index_arguments(index, defaults):
    """ Helper function to check pandas index arguments """
    if not any(isinstance(index, k) for k in (list, tuple)):
        index = [index]
    invalid_choices = [e for e in index if e not in defaults]
    if invalid_choices:
        options = [', '.join(e) for e in [invalid_choices, defaults]]
        raise ValueError('[%s] is not an valid option. Valid index'
                         'values are \'None\' or %s' % tuple(options))


def _clean_names(names, remove_whitespace=False, before_dash=True):
    """ Remove white-space on topo matching

    This function handles different naming
    conventions for old VS new VectorView systems (`remove_whitespace`).
    Also it allows to remove system specific parts in CTF channel names
    (`before_dash`).

    Usage
    -----
    # for new VectorView (only inside layout)
    ch_names = _clean_names(epochs.ch_names, remove_whitespace=True)

    # for CTF
    ch_names = _clean_names(epochs.ch_names, before_dash=True)

    """
    cleaned = []
    for name in names:
        if ' ' in name and remove_whitespace:
            name = name.replace(' ', '')
        if '-' in name and before_dash:
            name = name.split('-')[0]
        if name.endswith('_virtual'):
            name = name[:-8]
        cleaned.append(name)

    return cleaned


def clean_warning_registry():
    """Safe way to reset warnings """
    warnings.resetwarnings()
    reg = "__warningregistry__"
    bad_names = ['MovedModule']  # this is in six.py, and causes bad things
    for mod in list(sys.modules.values()):
        if mod.__class__.__name__ not in bad_names and hasattr(mod, reg):
            getattr(mod, reg).clear()
    # hack to deal with old scipy/numpy in tests
    if os.getenv('TRAVIS') == 'true' and sys.version.startswith('2.6'):
        warnings.simplefilter('default')
        try:
            np.rank([])
        except Exception:
            pass
        warnings.simplefilter('always')


def _check_type_picks(picks):
    """helper to guarantee type integrity of picks"""
    err_msg = 'picks must be None, a list or an array of integers'
    if picks is None:
        pass
    elif isinstance(picks, list):
        if not all(isinstance(i, int) for i in picks):
            raise ValueError(err_msg)
        picks = np.array(picks)
    elif isinstance(picks, np.ndarray):
        if not picks.dtype.kind == 'i':
            raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)
    return picks


@nottest
def run_tests_if_main(measure_mem=False):
    """Run tests in a given file if it is run as a script"""
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
                sys.stdout.write('time: %s sec%s\n' % (elapsed, mem))
                sys.stdout.flush()
            except Exception as err:
                if 'skiptest' in err.__class__.__name__.lower():
                    sys.stdout.write('SKIP (%s)\n' % str(err))
                    sys.stdout.flush()
                else:
                    raise
    elapsed = int(round(time.time() - t0))
    sys.stdout.write('Total: %s tests\n• %s sec (%s sec for %s)\n• Peak memory'
                     ' %s MB (%s)\n' % (count, elapsed, max_elapsed,
                                        elapsed_name, peak_mem, peak_name))


class ArgvSetter(object):
    """Temporarily set sys.argv"""
    def __init__(self, args=(), disable_stdout=True, disable_stderr=True):
        self.argv = list(('python',) + args)
        self.stdout = StringIO() if disable_stdout else sys.stdout
        self.stderr = StringIO() if disable_stderr else sys.stderr

    def __enter__(self):
        self.orig_argv = sys.argv
        sys.argv = self.argv
        self.orig_stdout = sys.stdout
        sys.stdout = self.stdout
        self.orig_stderr = sys.stderr
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):
        sys.argv = self.orig_argv
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr


def md5sum(fname, block_size=1048576):  # 2 ** 20
    """Calculate the md5sum for a file

    Parameters
    ----------
    fname : str
        Filename.
    block_size : int
        Block size to use when reading.

    Returns
    -------
    hash_ : str
        The hexidecimal digest of the hash.
    """
    md5 = hashlib.md5()
    with open(fname, 'rb') as fid:
        while True:
            data = fid.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def _sphere_to_cartesian(theta, phi, r):
    """Transform spherical coordinates to cartesian"""
    z = r * np.sin(phi)
    rcos_phi = r * np.cos(phi)
    x = rcos_phi * np.cos(theta)
    y = rcos_phi * np.sin(theta)
    return x, y, z


def create_slices(start, stop, step=None, length=1):
    """ Generate slices of time indexes

    Parameters
    ----------
    start : int
        Index where first slice should start.
    stop : int
        Index where last slice should maximally end.
    length : int
        Number of time sample included in a given slice.
    step: int | None
        Number of time samples separating two slices.
        If step = None, step = length.

    Returns
    -------
    slices : list
        List of slice objects.
    """

    # default parameters
    if step is None:
        step = length

    # slicing
    slices = [slice(t, t + length, 1) for t in
              range(start, stop - length + 1, step)]
    return slices


def _time_mask(times, tmin=None, tmax=None, strict=False):
    """Helper to safely find sample boundaries"""
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    mask = (times >= tmin)
    mask &= (times <= tmax)
    if not strict:
        mask |= isclose(times, tmin)
        mask |= isclose(times, tmax)
    return mask


def _get_fast_dot():
    """"Helper to get fast dot"""
    try:
        from sklearn.utils.extmath import fast_dot
    except ImportError:
        fast_dot = np.dot
    return fast_dot


def random_permutation(n_samples, random_state=None):
    """Helper to emulate the randperm matlab function.

    It returns a vector containing a random permutation of the
    integers between 0 and n_samples-1. It returns the same random numbers
    than randperm matlab function whenever the random_state is the same
    as the matlab's random seed.

    This function is useful for comparing against matlab scripts
    which use the randperm function.

    Note: the randperm(n_samples) matlab function generates a random
    sequence between 1 and n_samples, whereas
    random_permutation(n_samples, random_state) function generates
    a random sequence between 0 and n_samples-1, that is:
    randperm(n_samples) = random_permutation(n_samples, random_state) - 1

    Parameters
    ----------
    n_samples : int
        End point of the sequence to be permuted (excluded, i.e., the end point
        is equal to n_samples-1)
    random_state : int | None
        Random seed for initializing the pseudo-random number generator.

    Returns
    -------
    randperm : ndarray, int
        Randomly permuted sequence between 0 and n-1.
    """
    rng = check_random_state(random_state)
    idx = rng.rand(n_samples)

    randperm = np.argsort(idx)

    return randperm
