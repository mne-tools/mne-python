"""Some utility functions"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import warnings
import logging
from distutils.version import LooseVersion
import os
import os.path as op
from functools import wraps
import inspect
from string import Formatter
import subprocess
import sys
from sys import stdout
import tempfile
import shutil
from shutil import rmtree
import atexit
from math import log, ceil
import json
import ftplib
import hashlib

import numpy as np
import scipy
from scipy import linalg


from .externals.six.moves import urllib
from .externals.six import string_types, StringIO, BytesIO
from .externals.decorator import decorator

logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


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
        warnings.warn('This filename does not conform to mne naming convention'
                      's. All %s files should end with '
                      '%s' % (filetype, print_endings))


class WrapStdOut(object):
    """Ridiculous class to work around how doctest captures stdout"""
    def __getattr__(self, name):
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        return getattr(sys.stdout, name)


class _TempDir(str):
    """Class for creating and auto-destroying temp dir

    This is designed to be used with testing modules.

    We cannot simply use __del__() method for cleanup here because the rmtree
    function may be cleaned up before this object, so we use the atexit module
    instead.
    """
    def __new__(self):
        new = str.__new__(self, tempfile.mkdtemp())
        return new

    def __init__(self):
        self._path = self.__str__()
        atexit.register(self.cleanup)

    def cleanup(self):
        rmtree(self._path, ignore_errors=True)


def estimate_rank(data, tol=1e-4, return_singular=False,
                  copy=True):
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
    norms = np.sqrt(np.sum(data ** 2, axis=1))
    norms[norms == 0] = 1.0
    data /= norms[:, np.newaxis]
    s = linalg.svd(data, compute_uv=False, overwrite_a=True)
    rank = np.sum(s >= tol)
    if return_singular is True:
        return rank, s
    else:
        return rank


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


def run_subprocess(command, *args, **kwargs):
    """Run command using subprocess.Popen

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str
        Command to run as subprocess (see subprocess.Popen documentation).
    *args, **kwargs : arguments
        Arguments to pass to subprocess.Popen.

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

    logger.info("Running subprocess: %s" % str(command))
    p = subprocess.Popen(command, *args, **kwargs)
    stdout_, stderr = p.communicate()

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
    in an empty of parentheses:

    >>> from mne.utils import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <mne.utils.deprecated object at ...>

    >>> @deprecated()
    ... def some_function(): pass
    """
    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    # scikit-learn will not import on all platforms b/c it can be
    # sklearn or scikits.learn, so a self-contained example is used above

    def __init__(self, extra=''):
        """
        Parameters
        ----------
        extra: string
          to be added to the deprecation messages

        """
        self.extra = extra

    def __call__(self, obj):
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

    Do not call this directly to set global verbosrity level, instead use
    set_log_level().

    Parameters
    ----------
    function - function
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec - function
        The decorated function
    """
    arg_names = inspect.getargspec(function).args

    if len(arg_names) > 0 and arg_names[0] == 'self':
        default_level = getattr(args[0], 'verbose', None)
    else:
        default_level = None

    if('verbose' in arg_names):
        verbose_level = args[arg_names.index('verbose')]
    else:
        verbose_level = default_level

    if verbose_level is not None:
        old_level = set_log_level(verbose_level, True)
        # set it back if we get an exception
        try:
            ret = function(*args, **kwargs)
        except:
            set_log_level(old_level)
            raise
        set_log_level(old_level)
        return ret
    else:
        ret = function(*args, **kwargs)
        return ret


def has_command_line_tools():
    if 'MNE_ROOT' not in os.environ:
        return False
    else:
        return True


requires_mne = np.testing.dec.skipif(not has_command_line_tools(),
                                     'Requires MNE command line tools')


def has_nibabel(vox2ras_tkr=False):
    try:
        import nibabel
        if vox2ras_tkr:  # we need MGHHeader to have vox2ras_tkr param
            mgh_ihdr = getattr(nibabel, 'MGHImage', None)
            mgh_ihdr = getattr(mgh_ihdr, 'header_class', None)
            get_vox2ras_tkr = getattr(mgh_ihdr, 'get_vox2ras_tkr', None)
            if get_vox2ras_tkr is not None:
                return True
            else:
                return False
        else:
            return True
    except ImportError:
        return False


def has_freesurfer():
    """Aux function"""
    if not 'FREESURFER_HOME' in os.environ:
        return False
    else:
        return True


requires_fs_or_nibabel = np.testing.dec.skipif(not has_nibabel() and
                                               not has_freesurfer(),
                                               'Requires nibabel or '
                                               'Freesurfer')


def has_neuromag2ft():
    """Aux function"""
    if not 'NEUROMAG2FT_ROOT' in os.environ:
        return False
    else:
        return True


requires_neuromag2ft = np.testing.dec.skipif(not has_neuromag2ft(),
                                             'Requires neuromag2ft')


def requires_nibabel(vox2ras_tkr=False):
    """Aux function"""
    if vox2ras_tkr:
        extra = ' with vox2ras_tkr support'
    else:
        extra = ''
    return np.testing.dec.skipif(not has_nibabel(vox2ras_tkr),
                                 'Requires nibabel%s' % extra)

requires_freesurfer = np.testing.dec.skipif(not has_freesurfer(),
                                            'Requires Freesurfer')


def requires_mem_gb(requirement):
    """Decorator to skip test if insufficient memory is available"""
    def real_decorator(function):
        # convert to gb
        req = int(1e9 * requirement)
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False

        @wraps(function)
        def dec(*args, **kwargs):
            if has_psutil and psutil.virtual_memory().available >= req:
                skip = False
            else:
                skip = True

            if skip is True:
                from nose.plugins.skip import SkipTest
                raise SkipTest('Test %s skipped, requires >= %0.1f GB free '
                               'memory' % (function.__name__, requirement))
            ret = function(*args, **kwargs)
            return ret
        return dec
    return real_decorator


def requires_pandas(function):
    """Decorator to skip test if pandas is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        skip = False
        try:
            import pandas
            version = LooseVersion(pandas.__version__)
            if version < '0.8.0':
                skip = True
        except ImportError:
            skip = True

        if skip is True:
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires pandas'
                           % function.__name__)
        ret = function(*args, **kwargs)

        return ret

    return dec


def requires_tvtk(function):
    """Decorator to skip test if TVTK is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        skip = False
        try:
            from tvtk.api import tvtk  # analysis:ignore
        except ImportError:
            skip = True

        if skip is True:
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires TVTK'
                           % function.__name__)
        ret = function(*args, **kwargs)

        return ret

    return dec


def requires_statsmodels(function):
    """Decorator to skip test if statsmodels is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        skip = False
        try:
            import statsmodels  # noqa, analysis:ignore
        except ImportError:
            skip = True

        if skip is True:
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires statsmodels'
                           % function.__name__)
        ret = function(*args, **kwargs)

        return ret

    return dec


def requires_patsy(function):
    """
    Decorator to skip test if patsy is not available. Patsy should be a
    statsmodels dependency but apparently it's possible to install statsmodels
    without it.
    """
    @wraps(function)
    def dec(*args, **kwargs):
        skip = False
        try:
            import patsy  # noqa, analysis:ignore
        except ImportError:
            skip = True

        if skip is True:
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires patsy'
                           % function.__name__)
        ret = function(*args, **kwargs)

        return ret

    return dec


def requires_sklearn(function):
    """Decorator to skip test if sklearn is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        required_version = '0.14'
        skip = False
        try:
            import sklearn
            version = LooseVersion(sklearn.__version__)
            if version < required_version:
                skip = True
        except ImportError:
            skip = True

        if skip is True:
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires sklearn (version >= %s)'
                           % (function.__name__, required_version))
        ret = function(*args, **kwargs)

        return ret

    return dec


def requires_good_network(function):
    """Helper for testing"""

    @wraps(function)
    def dec(*args, **kwargs):
        if int(os.environ.get('MNE_SKIP_NETWORK_TESTS', 0)):
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires a good network '
                           'connection' % function.__name__)
        ret = function(*args, **kwargs)

        return ret

    return dec


def make_skipper_dec(module, skip_str):
    """Helper to make skipping decorators"""
    skip = False
    try:
        __import__(module)
    except ImportError:
        skip = True
    return np.testing.dec.skipif(skip, skip_str)


requires_nitime = make_skipper_dec('nitime', 'nitime not installed')
requires_traits = make_skipper_dec('traits', 'traits not installed')


def _mne_fs_not_in_env():
    """Aux function"""
    return (('FREESURFER_HOME' not in os.environ) or
            ('MNE_ROOT' not in os.environ))

requires_mne_fs_in_env = np.testing.dec.skipif(_mne_fs_not_in_env)


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
    has_mayavi = LooseVersion(mayavi.__version__)
    if has_mayavi < require_mayavi:
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


def requires_scipy_version(min_version):
    """Helper for testing"""
    ok = check_scipy_version(min_version)
    return np.testing.dec.skipif(not ok, 'Requires scipy version >= %s'
                                 % min_version)


def _check_pytables():
    """Helper to error if Pytables is not found"""
    try:
        import tables as tb
    except ImportError:
        raise ImportError('pytables could not be imported')
    return tb


def requires_pytables():
    """Helper for testing"""
    have = True
    try:
        _check_pytables()
    except ImportError:
        have = False
    return np.testing.dec.skipif(not have, 'Requires pytables')


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
        If None, the environment variable MNE_LOG_LEVEL is read, and if
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
        if not verbose in logging_types:
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
            http://docs.python.org/dev/howto/logging.html
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


def _get_extra_data_path(home_dir=None):
    """Get path to extra data (config, tables, etc.)"""
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
    'MNE_DATASETS_SPM_FACE_PATH',
    'MNE_DATASETS_EEGBCI_PATH',
    'MNE_LOGGING_LEVEL',
    'MNE_USE_CUDA',
    'SUBJECTS_DIR',
    'MNE_CACHE_DIR',
    'MNE_MEMMAP_MIN_SIZE',
    'MNE_SKIP_SAMPLE_DATASET_TESTS',
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
    """

    if key is not None and not isinstance(key, string_types):
        raise ValueError('key must be a string')

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
    """
    if not isinstance(key, string_types):
        raise ValueError('key must be a string')
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    if not isinstance(value, string_types) and value is not None:
        raise ValueError('value must be a string or None')
    if not key in known_config_types and not \
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
                 progress_character='.', spinner=False):
        self.cur_value = initial_value
        self.max_value = float(max_value)
        self.mesg = mesg
        self.max_chars = max_chars
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)

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
        progress = float(self.cur_value) / self.max_value
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
        sys.stdout.write(bar)
        # Increament the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        sys.stdout.flush()

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


class _HTTPResumeURLOpener(urllib.request.FancyURLopener):
    """Create sub-class in order to overide error 206.

    This error means a partial file is being sent, which is ok in this case.
    Do nothing with this error.
    """
    # Adapted from:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py
    # http://code.activestate.com/recipes/83208-resuming-download-of-a-file/

    def http_error_206(self, url, fp, errcode, errmsg, headers, data=None):
        pass


def _chunk_read(response, local_file, chunk_size=65536, initial_size=0):
    """Download a file chunk by chunk and show advancement

    Can also be used when resuming downloads over http.

    Parameters
    ----------
    response: urllib.response.addinfourl
        Response to the download request in order to get file size.
    local_file: file
        Hard disk file where data should be written.
    chunk_size: integer, optional
        Size of downloaded chunks. Default: 8192
    initial_size: int, optional
        If resuming, indicate the initial size of the file.
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    bytes_so_far = initial_size
    # Returns only amount left to download when resuming, not the size of the
    # entire file
    total_size = int(response.headers['Content-Length'].strip())
    total_size += initial_size

    progress = ProgressBar(total_size, initial_value=bytes_so_far,
                           max_chars=40, spinner=True, mesg='downloading')
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        if not chunk:
            sys.stderr.write('\n')
            break
        _chunk_write(chunk, local_file, progress)


def _chunk_read_ftp_resume(url, temp_file_name, local_file):
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
                           max_chars=40, spinner=True, mesg='downloading')
    # Callback lambda function that will be passed the downloaded data
    # chunk and will write it to file and update the progress bar
    chunk_write = lambda chunk: _chunk_write(chunk, local_file, progress)
    data.retrbinary(down_cmd, chunk_write)
    data.close()


def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar"""
    local_file.write(chunk)
    progress.update_with_increment_value(len(chunk))


def _fetch_file(url, file_name, print_destination=True, resume=True):
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
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    temp_file_name = file_name + ".part"
    local_file = None
    initial_size = 0
    try:
        # Checking file size and displaying it alongside the download url
        u = urllib.request.urlopen(url)
        try:
            file_size = int(u.headers['Content-Length'].strip())
        finally:
            del u
        print('Downloading data from %s (%s)' % (url, sizeof_fmt(file_size)))
        # Downloading data
        if resume and os.path.exists(temp_file_name):
            local_file = open(temp_file_name, "ab")
            # Resuming HTTP and FTP downloads requires different procedures
            scheme = urllib.parse.urlparse(url).scheme
            if scheme == 'http':
                url_opener = _HTTPResumeURLOpener()
                local_file_size = os.path.getsize(temp_file_name)
                # If the file exists, then only download the remainder
                url_opener.addheader("Range", "bytes=%s-" % (local_file_size))
                try:
                    data = url_opener.open(url)
                except urllib.request.HTTPError:
                    # There is a problem that may be due to resuming, some
                    # servers may not support the "Range" header. Switch back
                    # to complete download method
                    print('Resuming download failed. Attempting to restart '
                          'downloading the entire file.')
                    _fetch_file(url, resume=False)
                else:
                    _chunk_read(data, local_file, initial_size=local_file_size)
                    del data  # should auto-close
            else:
                _chunk_read_ftp_resume(url, temp_file_name, local_file)
        else:
            local_file = open(temp_file_name, "wb")
            data = urllib.request.urlopen(url)
            try:
                _chunk_read(data, local_file, initial_size=initial_size)
            finally:
                del data  # should auto-close
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_file_name, file_name)
        if print_destination is True:
            stdout.write('File saved as %s.\n' % file_name)
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


def _get_stim_channel(stim_channel):
    """Helper to determine the appropriate stim_channel"""
    if stim_channel is not None:
        if not isinstance(stim_channel, list):
            if not isinstance(stim_channel, string_types):
                raise ValueError('stim_channel must be a str, list, or None')
            stim_channel = [stim_channel]
        if not all([isinstance(s, string_types) for s in stim_channel]):
            raise ValueError('stim_channel list must contain all strings')
        return stim_channel

    stim_channel = list()
    ch_count = 0
    ch = get_config('MNE_STIM_CHANNEL')
    while(ch is not None):
        stim_channel.append(ch)
        ch_count += 1
        ch = get_config('MNE_STIM_CHANNEL_%d' % ch_count)
    if ch_count == 0:
        stim_channel = ['STI 014']
    return stim_channel


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
    invalid_choices = [e for e in index if not e in defaults]
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


def _check_type_picks(picks):
    """helper to guarantee type integrity of picks"""
    err_msg = 'picks must be None, a list or an array of integers'
    if picks is None:
        pass
    elif isinstance(picks, list):
        if not all([isinstance(i, int) for i in picks]):
            raise ValueError(err_msg)
        picks = np.array(picks)
    elif isinstance(picks, np.ndarray):
        if not picks.dtype.kind == 'i':
            raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)
    return picks
