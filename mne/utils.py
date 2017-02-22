# -*- coding: utf-8 -*-
"""Some utility functions."""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import atexit
from distutils.version import LooseVersion
from functools import wraps
import ftplib
from functools import partial
import hashlib
import inspect
import json
import logging
from math import log, ceil
import os
import os.path as op
import platform
import shutil
from shutil import rmtree
from string import Formatter
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import webbrowser

import numpy as np
from scipy import linalg, sparse

from .externals.six.moves import urllib
from .externals.six import string_types, StringIO, BytesIO, integer_types
from .externals.decorator import decorator

from .fixes import _get_args

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
    """Decorator to mark a function as not a test."""
    f.__test__ = False
    return f


# # # WARNING # # #
# This list must also be updated in doc/_templates/class.rst if it is
# changed here!
_doc_special_members = ('__contains__', '__getitem__', '__iter__', '__len__',
                        '__call__', '__add__', '__sub__', '__mul__', '__div__',
                        '__neg__', '__hash__')

###############################################################################
# RANDOM UTILITIES


def _pl(x):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, (integer_types, np.generic)) else len(x)
    return '' if len_x == 1 else 's'


def _explain_exception(start=-1, stop=None, prefix='> '):
    """Explain an exception."""
    # start=-1 means "only the most recent caller"
    etype, value, tb = sys.exc_info()
    string = traceback.format_list(traceback.extract_tb(tb)[start:stop])
    string = (''.join(string).split('\n') +
              traceback.format_exception_only(etype, value))
    string = ':\n' + prefix + ('\n' + prefix).join(string)
    return string


def _get_call_line(in_verbose=False):
    """Get the call line from within a function."""
    # XXX Eventually we could auto-triage whether in a `verbose` decorated
    # function or not.
    # NB This probably only works for functions that are undecorated,
    # or decorated by `verbose`.
    back = 2 if not in_verbose else 4
    call_frame = inspect.getouterframes(inspect.currentframe())[back][0]
    context = inspect.getframeinfo(call_frame).code_context
    context = 'unknown' if context is None else context[0].strip()
    return context


def _sort_keys(x):
    """Sort and return keys of dict."""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


def object_hash(x, h=None):
    """Hash a reasonable python object.

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
    if hasattr(x, 'keys'):
        # dict-like types
        keys = _sort_keys(x)
        for key in keys:
            object_hash(key, h)
            object_hash(x[key], h)
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
    elif hasattr(x, '__len__'):
        # all other list-like types
        h.update(str(type(x)).encode('utf-8'))
        for xx in x:
            object_hash(xx, h)
    else:
        raise RuntimeError('unsupported type: %s (%s)' % (type(x), x))
    return int(h.hexdigest(), 16)


def object_size(x):
    """Estimate the size of a reasonable python object.

    Parameters
    ----------
    x : object
        Object to approximate the size of.
        Can be anything comprised of nested versions of:
        {dict, list, tuple, ndarray, str, bytes, float, int, None}.

    Returns
    -------
    size : int
        The estimated size in bytes of the object.
    """
    # Note: this will not process object arrays properly (since those only)
    # hold references
    if isinstance(x, (bytes, string_types, int, float, type(None))):
        size = sys.getsizeof(x)
    elif isinstance(x, np.ndarray):
        # On newer versions of NumPy, just doing sys.getsizeof(x) works,
        # but on older ones you always get something small :(
        size = sys.getsizeof(np.array([])) + x.nbytes
    elif isinstance(x, np.generic):
        size = x.nbytes
    elif isinstance(x, dict):
        size = sys.getsizeof(x)
        for key, value in x.items():
            size += object_size(key)
            size += object_size(value)
    elif isinstance(x, (list, tuple)):
        size = sys.getsizeof(x) + sum(object_size(xx) for xx in x)
    elif sparse.isspmatrix_csc(x) or sparse.isspmatrix_csr(x):
        size = sum(sys.getsizeof(xx)
                   for xx in [x, x.data, x.indices, x.indptr])
    else:
        raise RuntimeError('unsupported type: %s (%s)' % (type(x), x))
    return size


def object_diff(a, b, pre=''):
    """Compute all differences between two python variables.

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
            out += pre + ' left missing keys %s\n' % (m1)
        for key in k1s:
            if key not in k2s:
                out += pre + ' right missing key %s\n' % key
            else:
                out += object_diff(a[key], b[key], pre + '[%s]' % repr(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out += pre + ' length mismatch (%s, %s)\n' % (len(a), len(b))
        else:
            for ii, (xx1, xx2) in enumerate(zip(a, b)):
                out += object_diff(xx1, xx2, pre + '[%s]' % ii)
    elif isinstance(a, (string_types, int, float, bytes)):
        if a != b:
            out += pre + ' value mismatch (%s, %s)\n' % (a, b)
    elif a is None:
        if b is not None:
            out += pre + ' left is None, right is not (%s)\n' % (b)
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
    """Turn seed into a np.random.RandomState instance.

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
    """Split list in n (approx) equal pieces."""
    n = int(n)
    sz = len(l) // n
    for i in range(n - 1):
        yield l[i * sz:(i + 1) * sz]
    yield l[(n - 1) * sz:]


def create_chunks(sequence, size):
    """Generate chunks from a sequence.

    Parameters
    ----------
    sequence : iterable
        Any iterable object
    size : int
        The chunksize to be returned
    """
    return (sequence[p:p + size] for p in range(0, len(sequence), size))


def sum_squared(X):
    """Compute norm of an array.

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


def warn(message, category=RuntimeWarning):
    """Emit a warning with trace outside the mne namespace.

    This function takes arguments like warnings.warn, and sends messages
    using both ``warnings.warn`` and ``logger.warn``. Warnings can be
    generated deep within nested function calls. In order to provide a
    more helpful warning, this function traverses the stack until it
    reaches a frame outside the ``mne`` namespace that caused the error.

    Parameters
    ----------
    message : str
        Warning message.
    category : instance of Warning
        The warning class. Defaults to ``RuntimeWarning``.
    """
    import mne
    root_dir = op.dirname(mne.__file__)
    frame = None
    stack = inspect.stack()
    last_fname = ''
    for fi, frame in enumerate(stack):
        fname, lineno = frame[1:3]
        if fname == '<string>' and last_fname == 'utils.py':  # in verbose dec
            last_fname = fname
            continue
        # treat tests as scripts
        # and don't capture unittest/case.py (assert_raises)
        if not (fname.startswith(root_dir) or
                ('unittest' in fname and 'case' in fname)) or \
                op.basename(op.dirname(fname)) == 'tests':
            break
        last_fname = op.basename(fname)
    if logger.level <= logging.WARN:
        # We need to use this instead of warn(message, category, stacklevel)
        # because we move out of the MNE stack, so warnings won't properly
        # recognize the module name (and our warnings.simplefilter will fail)
        warnings.warn_explicit(message, category, fname, lineno,
                               'mne', globals().get('__warningregistry__', {}))
    logger.warning(message)


def check_fname(fname, filetype, endings, endings_err=()):
    """Enforce MNE filename conventions.

    Parameters
    ----------
    fname : str
        Name of the file.
    filetype : str
        Type of file. e.g., ICA, Epochs etc.
    endings : tuple
        Acceptable endings for the filename.
    endings_err : tuple
        Obligatory possible endings for the filename.
    """
    if len(endings_err) > 0 and not fname.endswith(endings_err):
        print_endings = ' or '.join([', '.join(endings_err[:-1]),
                                     endings_err[-1]])
        raise IOError('The filename (%s) for file type %s must end with %s'
                      % (fname, filetype, print_endings))
    print_endings = ' or '.join([', '.join(endings[:-1]), endings[-1]])
    if not fname.endswith(endings):
        warn('This filename (%s) does not conform to MNE naming conventions. '
             'All %s files should end with %s'
             % (fname, filetype, print_endings))


class WrapStdOut(object):
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError("'file' object has not attribute '%s'" % name)


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


def estimate_rank(data, tol='auto', return_singular=False, norm=True):
    """Estimate the rank of data.

    This function will normalize the rows of the data (typically
    channels or vertices) such that non-zero singular values
    should be close to one.

    Parameters
    ----------
    data : array
        Data to estimate the rank of (should be 2-dimensional).
    tol : float | str
        Tolerance for singular values to consider non-zero in
        calculating the rank. The singular values are calculated
        in this method such that independent data are expected to
        have singular value around one. Can be 'auto' to use the
        same thresholding as ``scipy.linalg.orth``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    norm : bool
        If True, data will be scaled by their estimated row-wise norm.
        Else data are assumed to be scaled. Defaults to True.

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    data = data.copy()  # operate on a copy
    if norm is True:
        norms = _compute_row_norms(data)
        data /= norms[:, np.newaxis]
    s = linalg.svd(data, compute_uv=False, overwrite_a=True)
    if isinstance(tol, string_types):
        if tol != 'auto':
            raise ValueError('tol must be "auto" or float')
        eps = np.finfo(float).eps
        tol = np.max(data.shape) * np.amax(s) * eps
    tol = float(tol)
    rank = np.sum(s > tol)
    if return_singular is True:
        return rank, s
    else:
        return rank


def _compute_row_norms(data):
    """Compute scaling based on estimated norm."""
    norms = np.sqrt(np.sum(data ** 2, axis=1))
    norms[norms == 0] = 1.0
    return norms


def _reject_data_segments(data, reject, flat, decim, info, tstep):
    """Reject data segments using peak-to-peak amplitude."""
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


def _get_inst_data(inst):
    """Get data view from MNE object instance like Raw, Epochs or Evoked."""
    from .io.base import BaseRaw
    from .epochs import BaseEpochs
    from . import Evoked
    from .time_frequency.tfr import _BaseTFR

    if isinstance(inst, (BaseRaw, BaseEpochs, Evoked, _BaseTFR)):
        if not inst.preload:
            inst.load_data()
        return inst._data
    else:
        raise TypeError('The argument must be an instance of Raw, Epochs, '
                        'Evoked, EpochsTFR or AverageTFR, got {0}.'.format(
                            type(inst)))


class _FormatDict(dict):
    """Helper for pformat()."""

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


###############################################################################
# DECORATORS

# Following deprecated class copied from scikit-learn

# force show of DeprecationWarning even on python 2.7
warnings.filterwarnings('always', category=DeprecationWarning, module='mne')


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

    def __init__(self, extra=''):  # noqa: D102
        self.extra = extra

    def __call__(self, obj):  # noqa: D105
        """Call.

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

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = deprecation_wrapped

        deprecation_wrapped.__name__ = '__init__'
        deprecation_wrapped.__doc__ = self._update_doc(init.__doc__)
        deprecation_wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        deprecation_wrapped.__name__ = fun.__name__
        deprecation_wrapped.__dict__ = fun.__dict__
        deprecation_wrapped.__doc__ = self._update_doc(fun.__doc__)

        return deprecation_wrapped

    def _update_doc(self, olddoc):
        newdoc = ".. warning:: DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


@decorator
def verbose(function, *args, **kwargs):
    """Verbose decorator to allow functions to override log-level.

    This decorator is used to set the verbose level during a function or method
    call, such as :func:`mne.compute_covariance`. The `verbose` keyword
    argument can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', True (an
    alias for 'INFO'), or False (an alias for 'WARNING'). To set the global
    verbosity level for all functions, use :func:`mne.set_log_level`.

    Parameters
    ----------
    function : function
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec : function
        The decorated function

    Examples
    --------
    You can use the ``verbose`` argument to set the verbose level on the fly::
        >>> import mne
        >>> cov = mne.compute_raw_covariance(raw, verbose='WARNING')  # doctest: +SKIP
        >>> cov = mne.compute_raw_covariance(raw, verbose='INFO')  # doctest: +SKIP
        Using up to 49 segments
        Number of samples used : 5880
        [done]

    See Also
    --------
    set_log_level
    set_config
    """  # noqa: E501
    arg_names = _get_args(function)
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
        # set it back if we get an exception
        with use_log_level(verbose_level):
            return function(*args, **kwargs)
    return function(*args, **kwargs)


class use_log_level(object):
    """Context handler for logging level.

    Parameters
    ----------
    level : int
        The level to use.
    """

    def __init__(self, level):  # noqa: D102
        self.level = level

    def __enter__(self):  # noqa: D105
        self.old_level = set_log_level(self.level, True)

    def __exit__(self, *args):  # noqa: D105
        set_log_level(self.old_level)


@nottest
def slow_test(f):
    """Decorator for slow tests."""
    f.slow_test = True
    return f


@nottest
def ultra_slow_test(f):
    """Decorator for ultra slow tests."""
    f.ultra_slow_test = True
    f.slow_test = True
    return f


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
    """Aux function."""
    return 'MNE_ROOT' in os.environ


def has_freesurfer():
    """Aux function."""
    return 'FREESURFER_HOME' in os.environ


def requires_nibabel(vox2ras_tkr=False):
    """Aux function."""
    extra = ' with vox2ras_tkr support' if vox2ras_tkr else ''
    return np.testing.dec.skipif(not has_nibabel(vox2ras_tkr),
                                 'Requires nibabel%s' % extra)


def buggy_mkl_svd(function):
    """Decorator for tests that make calls to SVD and intermittently fail."""
    @wraps(function)
    def dec(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except np.linalg.LinAlgError as exp:
            if 'SVD did not converge' in str(exp):
                from nose.plugins.skip import SkipTest
                msg = 'Intel MKL SVD convergence error detected, skipping test'
                warn(msg)
                raise SkipTest(msg)
            raise
    return dec


def requires_version(library, min_version):
    """Helper for testing."""
    return np.testing.dec.skipif(not check_version(library, min_version),
                                 'Requires %s version >= %s'
                                 % (library, min_version))


def requires_module(function, name, call=None):
    """Decorator to skip test if package is not available."""
    call = ('import %s' % name) if call is None else call
    try:
        from nose.plugins.skip import SkipTest
    except ImportError:
        SkipTest = AssertionError

    @wraps(function)
    def dec(*args, **kwargs):  # noqa: D102
        try:
            exec(call) in globals(), locals()
        except Exception as exc:
            raise SkipTest('Test %s skipped, requires %s. Got exception (%s)'
                           % (function.__name__, name, exc))
        return function(*args, **kwargs)
    return dec


def copy_doc(source):
    """Decorator to copy the docstring from another function.

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated function

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B (A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """
    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise ValueError('Cannot copy docstring: docstring was empty.')
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


def copy_function_doc_to_method_doc(source):
    """Use the docstring from a function as docstring for a method.

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator. Additionally, the first parameter
    specified in the docstring of the source function is removed in the new
    docstring.

    This decorator is useful when implementing a method that just calls a
    function.  This pattern is prevalent in for example the plotting functions
    of MNE.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated method

    Examples
    --------
    >>> def plot_function(object, a, b):
    ...     '''Docstring for plotting function.
    ...
    ...     Parameters
    ...     ----------
    ...     object : instance of object
    ...         The object to plot
    ...     a : int
    ...         Some parameter
    ...     b : int
    ...         Some parameter
    ...     '''
    ...     pass
    ...
    >>> class A:
    ...     @copy_function_doc_to_method_doc(plot_function)
    ...     def plot(self, a, b):
    ...         '''
    ...         Notes
    ...         -----
    ...         .. versionadded:: 0.13.0
    ...         '''
    ...         plot_function(self, a, b)
    >>> print(A.plot.__doc__)
    Docstring for plotting function.
    <BLANKLINE>
        Parameters
        ----------
        a : int
            Some parameter
        b : int
            Some parameter
    <BLANKLINE>
            Notes
            -----
            .. versionadded:: 0.13.0
    <BLANKLINE>

    Notes
    -----
    The parsing performed is very basic and will break easily on docstrings
    that are not formatted exactly according to the ``numpydoc`` standard.
    Always inspect the resulting docstring when using this decorator.
    """
    def wrapper(func):
        doc = source.__doc__.split('\n')

        # Find parameter block
        for line, text in enumerate(doc[:-2]):
            if (text.strip() == 'Parameters' and
                    doc[line + 1].strip() == '----------'):
                parameter_block = line
                break
        else:
            # No parameter block found
            raise ValueError('Cannot copy function docstring: no parameter '
                             'block found. To simply copy the docstring, use '
                             'the @copy_doc decorator instead.')

        # Find first parameter
        for line, text in enumerate(doc[parameter_block:], parameter_block):
            if ':' in text:
                first_parameter = line
                parameter_indentation = len(text) - len(text.lstrip(' '))
                break
        else:
            raise ValueError('Cannot copy function docstring: no parameters '
                             'found. To simply copy the docstring, use the '
                             '@copy_doc decorator instead.')

        # Find end of first parameter
        for line, text in enumerate(doc[first_parameter + 1:],
                                    first_parameter + 1):
            # Ignore empty lines
            if len(text.strip()) == 0:
                continue

            line_indentation = len(text) - len(text.lstrip(' '))
            if line_indentation <= parameter_indentation:
                # Reach end of first parameter
                first_parameter_end = line

                # Of only one parameter is defined, remove the Parameters
                # heading as well
                if ':' not in text:
                    first_parameter = parameter_block

                break
        else:
            # End of docstring reached
            first_parameter_end = line
            first_parameter = parameter_block

        # Copy the docstring, but remove the first parameter
        doc = ('\n'.join(doc[:first_parameter]) + '\n' +
               '\n'.join(doc[first_parameter_end:]))
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


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

_sklearn_0_15_call = """
required_version = '0.15'
import sklearn
version = LooseVersion(sklearn.__version__)
if version < required_version:
    raise ImportError
"""

_mayavi_call = """
with warnings.catch_warnings(record=True):  # traits
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
requires_sklearn_0_15 = partial(requires_module, name='sklearn',
                                call=_sklearn_0_15_call)
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
requires_statsmodels = partial(requires_module, name='statsmodels')
requires_pysurfer = partial(requires_module, name='PySurfer',
                            call="""import warnings
with warnings.catch_warnings(record=True):
    from surfer import Brain""")
requires_PIL = partial(requires_module, name='PIL',
                       call='from PIL import Image')
requires_good_network = partial(
    requires_module, name='good network connection',
    call='if int(os.environ.get("MNE_SKIP_NETWORK_TESTS", 0)):\n'
         '    raise ImportError')
requires_ftp = partial(
    requires_module, name='ftp downloading capability',
    call='if int(os.environ.get("MNE_SKIP_FTP_TESTS", 0)):\n'
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
        ``'(\d+ | [a-z]+ | \.)'``

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
        this_version = LooseVersion(library.__version__)
        if this_version < min_version:
            ok = False
    return ok


def _check_mayavi_version(min_version='4.3.0'):
    """Helper for mayavi."""
    if not check_version('mayavi', min_version):
        raise RuntimeError("Need mayavi >= %s" % min_version)


def _check_pyface_backend():
    """Check the currently selected Pyface backend.

    Returns
    -------
    backend : str
        Name of the backend.
    result : 0 | 1 | 2
        0: the backend has been tested and works.
        1: the backend has not been tested.
        2: the backend not been tested.

    Notes
    -----
    See also: http://docs.enthought.com/pyface/
    """
    try:
        from traits.trait_base import ETSConfig
    except ImportError:
        return None, 2

    backend = ETSConfig.toolkit
    if backend == 'qt4':
        status = 0
    else:
        status = 1
    return backend, status


def _import_mlab():
    """Quietly import mlab."""
    with warnings.catch_warnings(record=True):
        from mayavi import mlab
    return mlab


@verbose
def run_subprocess(command, verbose=None, *args, **kwargs):
    """Run command using subprocess.Popen.

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str | str
        Command to run as subprocess (see subprocess.Popen documentation).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        self.verbose.
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.

    Returns
    -------
    stdout : str
        Stdout returned by the process.
    stderr : str
        Stderr returned by the process.
    """
    for stdxxx, sys_stdxxx in (['stderr', sys.stderr],
                               ['stdout', sys.stdout]):
        if stdxxx not in kwargs:
            kwargs[stdxxx] = subprocess.PIPE
        elif kwargs[stdxxx] is sys_stdxxx:
            if isinstance(sys_stdxxx, StringIO):
                # nose monkey patches sys.stderr and sys.stdout to StringIO
                kwargs[stdxxx] = subprocess.PIPE
            else:
                kwargs[stdxxx] = sys_stdxxx

    # Check the PATH environment variable. If run_subprocess() is to be called
    # frequently this should be refactored so as to only check the path once.
    env = kwargs.get('env', os.environ)
    if any(p.startswith('~') for p in env['PATH'].split(os.pathsep)):
        warn('Your PATH environment variable contains at least one path '
             'starting with a tilde ("~") character. Such paths are not '
             'interpreted correctly from within Python. It is recommended '
             'that you use "$HOME" instead of "~".')
    if isinstance(command, string_types):
        command_str = command
    else:
        command_str = ' '.join(command)
    logger.info("Running subprocess: %s" % command_str)
    try:
        p = subprocess.Popen(command, *args, **kwargs)
    except Exception:
        if isinstance(command, string_types):
            command_name = command.split()[0]
        else:
            command_name = command[0]
        logger.error('Command not found: %s' % command_name)
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
        if 'output' in _get_args(err_fun):
            raise subprocess.CalledProcessError(p.returncode, command, output)
        else:
            raise subprocess.CalledProcessError(p.returncode, command)

    return output


###############################################################################
# LOGGING

def set_log_level(verbose=None, return_old_level=False):
    """Set the logging level.

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
    """Set the log to print to a file.

    Parameters
    ----------
    fname : str, or None
        Filename of the log to print to. If None, stdout is used.
        To suppress log outputs, use set_log_level('WARN').
    output_format : str
        Format of the output messages. See the following for examples:

            https://docs.python.org/dev/howto/logging.html

        e.g., "%(asctime)s - %(levelname)s - %(message)s".
    overwrite : bool | None
        Overwrite the log file (if it exists). Otherwise, statements
        will be appended to the log (default). None is the same as False,
        but additionally raises a warning to notify the user that log
        entries will be appended.
    """
    logger = logging.getLogger('mne')
    handlers = logger.handlers
    for h in handlers:
        # only remove our handlers (get along nicely with nose)
        if isinstance(h, (logging.FileHandler, logging.StreamHandler)):
            if isinstance(h, logging.FileHandler):
                h.close()
            logger.removeHandler(h)
    if fname is not None:
        if op.isfile(fname) and overwrite is None:
            # Don't use warn() here because we just want to
            # emit a warnings.warn here (not logger.warn)
            warnings.warn('Log entries will be appended to the file. Use '
                          'overwrite=False to avoid this message in the '
                          'future.', RuntimeWarning, stacklevel=2)
            overwrite = False
        mode = 'w' if overwrite else 'a'
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


class catch_logging(object):
    """Helper to store logging.

    This will remove all other logging handlers, and return the handler to
    stdout when complete.
    """

    def __enter__(self):  # noqa: D105
        self._data = StringIO()
        self._lh = logging.StreamHandler(self._data)
        self._lh.setFormatter(logging.Formatter('%(message)s'))
        for lh in logger.handlers:
            logger.removeHandler(lh)
        logger.addHandler(self._lh)
        return self._data

    def __exit__(self, *args):  # noqa: D105
        logger.removeHandler(self._lh)
        set_log_file(None)


###############################################################################
# CONFIG / PREFS

def get_subjects_dir(subjects_dir=None, raise_error=False):
    """Safely use subjects_dir input to return SUBJECTS_DIR.

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
    """Get path to extra data (config, tables, etc.)."""
    global _temp_home_dir
    if home_dir is None:
        home_dir = os.environ.get('_MNE_FAKE_HOME_DIR')
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
    r"""Get path to standard mne-python config file.

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

    set_config('MNE_CACHE_DIR', cache_dir, set_env=False)


def set_memmap_min_size(memmap_min_size):
    """Set the minimum size for memmaping of arrays for parallel processing.

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

    set_config('MNE_MEMMAP_MIN_SIZE', memmap_min_size, set_env=False)


# List the known configuration values
known_config_types = (
    'MNE_BROWSE_RAW_SIZE',
    'MNE_CACHE_DIR',
    'MNE_COREG_COPY_ANNOT',
    'MNE_COREG_GUESS_MRI_SUBJECT',
    'MNE_COREG_HEAD_HIGH_RES',
    'MNE_COREG_HEAD_OPACITY',
    'MNE_COREG_PREPARE_BEM',
    'MNE_COREG_SCALE_LABELS',
    'MNE_COREG_SCENE_HEIGHT',
    'MNE_COREG_SCENE_WIDTH',
    'MNE_COREG_SUBJECTS_DIR',
    'MNE_CUDA_IGNORE_PRECISION',
    'MNE_DATA',
    'MNE_DATASETS_BRAINSTORM_PATH',
    'MNE_DATASETS_EEGBCI_PATH',
    'MNE_DATASETS_MEGSIM_PATH',
    'MNE_DATASETS_MISC_PATH',
    'MNE_DATASETS_SAMPLE_PATH',
    'MNE_DATASETS_SOMATO_PATH',
    'MNE_DATASETS_MULTIMODAL_PATH',
    'MNE_DATASETS_SPM_FACE_DATASETS_TESTS',
    'MNE_DATASETS_SPM_FACE_PATH',
    'MNE_DATASETS_TESTING_PATH',
    'MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
    'MNE_FORCE_SERIAL',
    'MNE_KIT2FIFF_STIM_CHANNELS',
    'MNE_KIT2FIFF_STIM_CHANNEL_CODING',
    'MNE_KIT2FIFF_STIM_CHANNEL_SLOPE',
    'MNE_KIT2FIFF_STIM_CHANNEL_THRESHOLD',
    'MNE_LOGGING_LEVEL',
    'MNE_MEMMAP_MIN_SIZE',
    'MNE_SKIP_FTP_TESTS',
    'MNE_SKIP_NETWORK_TESTS',
    'MNE_SKIP_TESTING_DATASET_TESTS',
    'MNE_STIM_CHANNEL',
    'MNE_USE_CUDA',
    'MNE_SKIP_FS_FLASH_CALL',
    'SUBJECTS_DIR',
)

# These allow for partial matches, e.g. 'MNE_STIM_CHANNEL_1' is okay key
known_config_wildcards = (
    'MNE_STIM_CHANNEL',
)


def _load_config(config_path, raise_error=False):
    """Safely load a config file."""
    with open(config_path, 'r') as fid:
        try:
            config = json.load(fid)
        except ValueError:
            # No JSON object could be decoded --> corrupt file?
            msg = ('The MNE-Python config file (%s) is not a valid JSON '
                   'file and might be corrupted' % config_path)
            if raise_error:
                raise RuntimeError(msg)
            warn(msg)
            config = dict()
    return config


def get_config(key=None, default=None, raise_error=False, home_dir=None):
    """Read MNE-Python preferences from environment or config file.

    Parameters
    ----------
    key : None | str
        The preference key to look for. The os evironment is searched first,
        then the mne-python config file is parsed.
        If None, all the config parameters present in environment variables or
        the path are returned.
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
        config = {}
    else:
        config = _load_config(config_path)

    if key is None:
        # update config with environment variables
        env_keys = (set(config).union(known_config_types).
                    intersection(os.environ))
        config.update({key: os.environ[key] for key in env_keys})
        return config
    elif raise_error is True and key not in config:
        meth_1 = 'os.environ["%s"] = VALUE' % key
        meth_2 = 'mne.utils.set_config("%s", VALUE, set_env=True)' % key
        raise KeyError('Key "%s" not found in environment or in the '
                       'mne-python config file: %s '
                       'Try either:'
                       ' %s for a temporary solution, or:'
                       ' %s for a permanent one. You can also '
                       'set the environment variable before '
                       'running python.'
                       % (key, config_path, meth_1, meth_2))
    else:
        return config.get(key, default)


def set_config(key, value, home_dir=None, set_env=True):
    """Set a MNE-Python preference key in the config file and environment.

    Parameters
    ----------
    key : str | None
        The preference key to set. If None, a tuple of the valid
        keys is returned, and ``value`` and ``home_dir`` are ignored.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    set_env : bool
        If True (default), update :data:`os.environ` in addition to
        updating the MNE-Python config file.

    See Also
    --------
    get_config
    """
    if key is None:
        return known_config_types
    if not isinstance(key, string_types):
        raise TypeError('key must be a string')
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    if not isinstance(value, string_types) and value is not None:
        raise TypeError('value must be a string or None')
    if key not in known_config_types and not \
            any(k in key for k in known_config_wildcards):
        warn('Setting non-standard config type: "%s"' % key)

    # Read all previous values
    config_path = get_config_path(home_dir=home_dir)
    if op.isfile(config_path):
        config = _load_config(config_path, raise_error=True)
    else:
        config = dict()
        logger.info('Attempting to create new mne-python configuration '
                    'file:\n%s' % config_path)
    if value is None:
        config.pop(key, None)
        if set_env and key in os.environ:
            del os.environ[key]
    else:
        config[key] = value
        if set_env:
            os.environ[key] = value

    # Write all values. This may fail if the default directory is not
    # writeable.
    directory = op.dirname(config_path)
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, 'w') as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


class ProgressBar(object):
    """Generate a command-line progressbar.

    Parameters
    ----------
    max_value : int | iterable
        Maximum value of process (e.g. number of samples to process, bytes to
        download, etc.). If an iterable is given, then `max_value` will be set
        to the length of this iterable.
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
                 progress_character='.', spinner=False,
                 verbose_bool=True):  # noqa: D102
        self.cur_value = initial_value
        if isinstance(max_value, (float, int)):
            self.max_value = max_value
            self.iterable = None
        else:
            # input is an iterable
            self.max_value = len(max_value)
            self.iterable = max_value
        self.mesg = mesg
        self.max_chars = max_chars
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)
        self._do_print = verbose_bool

    def update(self, cur_value, mesg=None):
        """Update progressbar with current value of process.

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
            if mesg == 'file_sizes':
                mesg = '(%s / %s)' % (sizeof_fmt(self.cur_value),
                                      sizeof_fmt(self.max_value))
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
        """Update progressbar with an increment.

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

    def __iter__(self):
        """Iterate to auto-increment the pbar with 1."""
        if self.iterable is None:
            raise ValueError("Must give an iterable to be used in a loop.")
        for obj in self.iterable:
            yield obj
            self.update_with_increment_value(1)


def _get_ftp(url, temp_file_name, initial_size, file_size, verbose_bool):
    """Safely (resume a) download to a file from FTP."""
    # Adapted from: https://pypi.python.org/pypi/fileDownloader.py
    # but with changes

    parsed_url = urllib.parse.urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    server_path = parsed_url.path.replace(file_name, "")
    unquoted_server_path = urllib.parse.unquote(server_path)

    data = ftplib.FTP()
    if parsed_url.port is not None:
        data.connect(parsed_url.hostname, parsed_url.port)
    else:
        data.connect(parsed_url.hostname)
    data.login()
    if len(server_path) > 1:
        data.cwd(unquoted_server_path)
    data.sendcmd("TYPE I")
    data.sendcmd("REST " + str(initial_size))
    down_cmd = "RETR " + file_name
    assert file_size == data.size(file_name)
    progress = ProgressBar(file_size, initial_value=initial_size,
                           max_chars=40, spinner=True, mesg='file_sizes',
                           verbose_bool=verbose_bool)

    # Callback lambda function that will be passed the downloaded data
    # chunk and will write it to file and update the progress bar
    mode = 'ab' if initial_size > 0 else 'wb'
    with open(temp_file_name, mode) as local_file:
        def chunk_write(chunk):
            return _chunk_write(chunk, local_file, progress)
        data.retrbinary(down_cmd, chunk_write)
        data.close()
    sys.stdout.write('\n')
    sys.stdout.flush()


def _get_http(url, temp_file_name, initial_size, file_size, verbose_bool):
    """Safely (resume a) download to a file from http(s)."""
    # Actually do the reading
    req = urllib.request.Request(url)
    if initial_size > 0:
        req.headers['Range'] = 'bytes=%s-' % (initial_size,)
    try:
        response = urllib.request.urlopen(req)
    except Exception:
        # There is a problem that may be due to resuming, some
        # servers may not support the "Range" header. Switch
        # back to complete download method
        logger.info('Resuming download failed (server '
                    'rejected the request). Attempting to '
                    'restart downloading the entire file.')
        del req.headers['Range']
        response = urllib.request.urlopen(req)
    total_size = int(response.headers.get('Content-Length', '1').strip())
    if initial_size > 0 and file_size == total_size:
        logger.info('Resuming download failed (resume file size '
                    'mismatch). Attempting to restart downloading the '
                    'entire file.')
        initial_size = 0
    total_size += initial_size
    if total_size != file_size:
        raise RuntimeError('URL could not be parsed properly')
    mode = 'ab' if initial_size > 0 else 'wb'
    progress = ProgressBar(total_size, initial_value=initial_size,
                           max_chars=40, spinner=True, mesg='file_sizes',
                           verbose_bool=verbose_bool)
    chunk_size = 8192  # 2 ** 13
    with open(temp_file_name, mode) as local_file:
        while True:
            t0 = time.time()
            chunk = response.read(chunk_size)
            dt = time.time() - t0
            if dt < 0.005:
                chunk_size *= 2
            elif dt > 0.1 and chunk_size > 8192:
                chunk_size = chunk_size // 2
            if not chunk:
                if verbose_bool:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                break
            local_file.write(chunk)
            progress.update_with_increment_value(len(chunk),
                                                 mesg='file_sizes')


def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar."""
    local_file.write(chunk)
    progress.update_with_increment_value(len(chunk))


@verbose
def _fetch_file(url, file_name, print_destination=True, resume=True,
                hash_=None, timeout=10., verbose=None):
    """Load requested file, downloading it if needed or requested.

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
    timeout : float
        The URL open timeout.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py
    if hash_ is not None and (not isinstance(hash_, string_types) or
                              len(hash_) != 32):
        raise ValueError('Bad hash value given, should be a 32-character '
                         'string:\n%s' % (hash_,))
    temp_file_name = file_name + ".part"
    verbose_bool = (logger.level <= 20)  # 20 is info
    try:
        # Check file size and displaying it alongside the download url
        u = urllib.request.urlopen(url, timeout=timeout)
        u.close()
        # this is necessary to follow any redirects
        url = u.geturl()
        u = urllib.request.urlopen(url, timeout=timeout)
        try:
            file_size = int(u.headers.get('Content-Length', '1').strip())
        finally:
            u.close()
            del u
        logger.info('Downloading data from %s (%s)\n'
                    % (url, sizeof_fmt(file_size)))

        # Triage resume
        if not os.path.exists(temp_file_name):
            resume = False
        if resume:
            with open(temp_file_name, 'rb', buffering=0) as local_file:
                local_file.seek(0, 2)
                initial_size = local_file.tell()
            del local_file
        else:
            initial_size = 0
        # This should never happen if our functions work properly
        if initial_size > file_size:
            raise RuntimeError('Local file (%s) is larger than remote '
                               'file (%s), cannot resume download'
                               % (sizeof_fmt(initial_size),
                                  sizeof_fmt(file_size)))

        scheme = urllib.parse.urlparse(url).scheme
        fun = _get_http if scheme in ('http', 'https') else _get_ftp
        fun(url, temp_file_name, initial_size, file_size, verbose_bool)

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
    except Exception:
        logger.error('Error while fetching file %s.'
                     ' Dataset fetching aborted.' % url)
        raise


def sizeof_fmt(num):
    """Turn number of bytes into human-readable str."""
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


class SizeMixin(object):
    """Estimate MNE object sizes."""

    @property
    def _size(self):
        """Estimate the object size."""
        try:
            size = object_size(self.info)
        except Exception:
            warn('Could not get size for self.info')
            return -1
        if hasattr(self, 'data'):
            size += object_size(self.data)
        elif hasattr(self, '_data'):
            size += object_size(self._data)
        return size

    def __hash__(self):
        """Hash the object.

        Returns
        -------
        hash : int
            The hash
        """
        from .evoked import Evoked
        from .epochs import BaseEpochs
        from .io.base import BaseRaw
        if isinstance(self, Evoked):
            return object_hash(dict(info=self.info, data=self.data))
        elif isinstance(self, (BaseEpochs, BaseRaw)):
            if not self.preload:
                raise RuntimeError('Cannot hash %s unless data are loaded'
                                   % self.__class__.__name__)
            return object_hash(dict(info=self.info, data=self._data))
        else:
            raise RuntimeError('Hashing unknown object type: %s' % type(self))


def _url_to_local_path(url, path):
    """Mirror a url path in a local destination (keeping folder structure)."""
    destination = urllib.parse.urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != '/':
        raise ValueError('Invalid URL')
    destination = os.path.join(path,
                               urllib.request.url2pathname(destination)[1:])
    return destination


def _get_stim_channel(stim_channel, info, raise_error=True):
    """Determine the appropriate stim_channel.

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

    if 'STI101' in info['ch_names']:  # combination channel for newer systems
        return ['STI101']
    if 'STI 014' in info['ch_names']:  # for older systems
        return ['STI 014']

    from .io.pick import pick_types
    stim_channel = pick_types(info, meg=False, ref_meg=False, stim=True)
    if len(stim_channel) > 0:
        stim_channel = [info['ch_names'][ch_] for ch_ in stim_channel]
    elif raise_error:
        raise ValueError("No stim channels found. Consider specifying them "
                         "manually using the 'stim_channel' parameter.")
    return stim_channel


def _check_fname(fname, overwrite=False, must_exist=False):
    """Check for file existence."""
    if not isinstance(fname, string_types):
        raise TypeError('file name is not a string')
    if must_exist and not op.isfile(fname):
        raise IOError('File "%s" does not exist' % fname)
    if op.isfile(fname):
        if not overwrite:
            raise IOError('Destination file exists. Please use option '
                          '"overwrite=True" to force overwriting.')
        else:
            logger.info('Overwriting existing file.')


def _check_subject(class_subject, input_subject, raise_error=True):
    """Get subject name from class."""
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
    """Aux function."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise RuntimeError('For this method to work the Pandas library is'
                           ' required.')


def _check_pandas_index_arguments(index, defaults):
    """Check pandas index arguments."""
    if not any(isinstance(index, k) for k in (list, tuple)):
        index = [index]
    invalid_choices = [e for e in index if e not in defaults]
    if invalid_choices:
        options = [', '.join(e) for e in [invalid_choices, defaults]]
        raise ValueError('[%s] is not an valid option. Valid index'
                         'values are \'None\' or %s' % tuple(options))


def _clean_names(names, remove_whitespace=False, before_dash=True):
    """Remove white-space on topo matching.

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


def _check_type_picks(picks):
    """Guarantee type integrity of picks."""
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


def md5sum(fname, block_size=1048576):  # 2 ** 20
    """Calculate the md5sum for a file.

    Parameters
    ----------
    fname : str
        Filename.
    block_size : int
        Block size to use when reading.

    Returns
    -------
    hash_ : str
        The hexadecimal digest of the hash.
    """
    md5 = hashlib.md5()
    with open(fname, 'rb') as fid:
        while True:
            data = fid.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def create_slices(start, stop, step=None, length=1):
    """Generate slices of time indexes.

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


def _time_mask(times, tmin=None, tmax=None, sfreq=None, raise_error=True):
    """Safely find sample boundaries."""
    orig_tmin = tmin
    orig_tmax = tmax
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    if not np.isfinite(tmin):
        tmin = times[0]
    if not np.isfinite(tmax):
        tmax = times[-1]
    if sfreq is not None:
        # Push to a bit past the nearest sample boundary first
        sfreq = float(sfreq)
        tmin = int(round(tmin * sfreq)) / sfreq - 0.5 / sfreq
        tmax = int(round(tmax * sfreq)) / sfreq + 0.5 / sfreq
    if raise_error and tmin > tmax:
        raise ValueError('tmin (%s) must be less than or equal to tmax (%s)'
                         % (orig_tmin, orig_tmax))
    mask = (times >= tmin)
    mask &= (times <= tmax)
    if raise_error and not mask.any():
        raise ValueError('No samples remain when using tmin=%s and tmax=%s '
                         '(original time bounds are [%s, %s])'
                         % (orig_tmin, orig_tmax, times[0], times[-1]))
    return mask


def _get_fast_dot():
    """Get fast dot."""
    try:
        from sklearn.utils.extmath import fast_dot
    except ImportError:
        fast_dot = np.dot
    return fast_dot


def random_permutation(n_samples, random_state=None):
    """Emulate the randperm matlab function.

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


def compute_corr(x, y):
    """Compute pearson correlations between a vector and a matrix."""
    if len(x) == 0 or len(y) == 0:
        raise ValueError('x or y has zero length')
    fast_dot = _get_fast_dot()
    X = np.array(x, float)
    Y = np.array(y, float)
    X -= X.mean(0)
    Y -= Y.mean(0)
    x_sd = X.std(0, ddof=1)
    # if covariance matrix is fully expanded, Y needs a
    # transpose / broadcasting else Y is correct
    y_sd = Y.std(0, ddof=1)[:, None if X.shape == Y.shape else Ellipsis]
    return (fast_dot(X.T, Y) / float(len(X) - 1)) / (x_sd * y_sd)


def grand_average(all_inst, interpolate_bads=True, drop_bads=True):
    """Make grand average of a list evoked or AverageTFR data.

    For evoked data, the function interpolates bad channels based on
    `interpolate_bads` parameter. If `interpolate_bads` is True, the grand
    average file will contain good channels and the bad channels interpolated
    from the good MEG/EEG channels.
    For AverageTFR data, the function takes the subset of channels not marked
    as bad in any of the instances.

    The grand_average.nave attribute will be equal to the number
    of evoked datasets used to calculate the grand average.

    Note: Grand average evoked should not be used for source localization.

    Parameters
    ----------
    all_inst : list of Evoked or AverageTFR data
        The evoked datasets.
    interpolate_bads : bool
        If True, bad MEG and EEG channels are interpolated. Ignored for
        AverageTFR.
    drop_bads : bool
        If True, drop all bad channels marked as bad in any data set.
        If neither interpolate_bads nor drop_bads is True, in the output file,
        every channel marked as bad in at least one of the input files will be
        marked as bad, but no interpolation or dropping will be performed.

    Returns
    -------
    grand_average : Evoked | AverageTFR
        The grand average data. Same type as input.

    Notes
    -----
    .. versionadded:: 0.11.0
    """
    # check if all elements in the given list are evoked data
    from .evoked import Evoked
    from .time_frequency import AverageTFR
    from .channels.channels import equalize_channels
    if not all(isinstance(inst, (Evoked, AverageTFR)) for inst in all_inst):
        raise ValueError("Not all input elements are Evoked or AverageTFR")

    # Copy channels to leave the original evoked datasets intact.
    all_inst = [inst.copy() for inst in all_inst]

    # Interpolates if necessary
    if isinstance(all_inst[0], Evoked):
        if interpolate_bads:
            all_inst = [inst.interpolate_bads() if len(inst.info['bads']) > 0
                        else inst for inst in all_inst]
        equalize_channels(all_inst)  # apply equalize_channels
        from .evoked import combine_evoked as combine
    else:  # isinstance(all_inst[0], AverageTFR):
        from .time_frequency.tfr import combine_tfr as combine

    if drop_bads:
        bads = list(set((b for inst in all_inst for b in inst.info['bads'])))
        if bads:
            for inst in all_inst:
                inst.drop_channels(bads)

    # make grand_average object using combine_[evoked/tfr]
    grand_average = combine(all_inst, weights='equal')
    # change the grand_average.nave to the number of Evokeds
    grand_average.nave = len(all_inst)
    # change comment field
    grand_average.comment = "Grand average (n = %d)" % grand_average.nave
    return grand_average


def _get_root_dir():
    """Get as close to the repo root as possible."""
    root_dir = op.abspath(op.dirname(__file__))
    up_dir = op.join(root_dir, '..')
    if op.isfile(op.join(up_dir, 'setup.py')) and all(
            op.isdir(op.join(up_dir, x)) for x in ('mne', 'examples', 'doc')):
        root_dir = op.abspath(up_dir)
    return root_dir


def sys_info(fid=None, show_paths=False):
    """Print the system information for debugging.

    This function is useful for printing system information
    to help triage bugs.

    Parameters
    ----------
    fid : file-like | None
        The file to write to. Will be passed to :func:`print()`.
        Can be None to use :data:`sys.stdout`.
    show_paths : bool
        If True, print paths for each module.

    Examples
    --------
    Running this function with no arguments prints an output that is
    useful when submitting bug reports::

        >>> import mne
        >>> mne.sys_info() # doctest: +SKIP
        Platform:      Linux-4.2.0-27-generic-x86_64-with-Ubuntu-15.10-wily
        Python:        2.7.10 (default, Oct 14 2015, 16:09:02)  [GCC 5.2.1 20151010]
        Executable:    /usr/bin/python

        mne:           0.12.dev0
        numpy:         1.12.0.dev0+ec5bd81 {lapack=mkl_rt, blas=mkl_rt}
        scipy:         0.18.0.dev0+3deede3
        matplotlib:    1.5.1+1107.g1fa2697

        sklearn:       0.18.dev0
        nibabel:       2.1.0dev
        mayavi:        4.3.1
        pycuda:        2015.1.3
        skcuda:        0.5.2
        pandas:        0.17.1+25.g547750a

    """  # noqa: E501
    ljust = 15
    out = 'Platform:'.ljust(ljust) + platform.platform() + '\n'
    out += 'Python:'.ljust(ljust) + str(sys.version).replace('\n', ' ') + '\n'
    out += 'Executable:'.ljust(ljust) + sys.executable + '\n\n'
    old_stdout = sys.stdout
    capture = StringIO()
    try:
        sys.stdout = capture
        np.show_config()
    finally:
        sys.stdout = old_stdout
    lines = capture.getvalue().split('\n')
    libs = []
    for li, line in enumerate(lines):
        for key in ('lapack', 'blas'):
            if line.startswith('%s_opt_info' % key):
                libs += ['%s=' % key +
                         lines[li + 1].split('[')[1].split("'")[1]]
    libs = ', '.join(libs)
    version_texts = dict(pycuda='VERSION_TEXT')
    for mod_name in ('mne', 'numpy', 'scipy', 'matplotlib', '',
                     'sklearn', 'nibabel', 'mayavi', 'pycuda', 'skcuda',
                     'pandas'):
        if mod_name == '':
            out += '\n'
            continue
        out += ('%s:' % mod_name).ljust(ljust)
        try:
            mod = __import__(mod_name)
        except Exception:
            out += 'Not found\n'
        else:
            version = getattr(mod, version_texts.get(mod_name, '__version__'))
            extra = (' (%s)' % op.dirname(mod.__file__)) if show_paths else ''
            if mod_name == 'numpy':
                extra = ' {%s}%s' % (libs, extra)
            out += '%s%s\n' % (version, extra)
    print(out, end='', file=fid)


class ETSContext(object):
    """Add more meaningful message to errors generated by ETS Toolkit."""

    def __enter__(self):  # noqa: D105
        pass

    def __exit__(self, type, value, traceback):  # noqa: D105
        if isinstance(value, SystemExit) and value.code.\
                startswith("This program needs access to the screen"):
            value.code += ("\nThis can probably be solved by setting "
                           "ETS_TOOLKIT=qt4. On bash, type\n\n    $ export "
                           "ETS_TOOLKIT=qt4\n\nand run the command again.")


def open_docs(kind=None, version=None):
    """Launch a new web browser tab with the MNE documentation.

    Parameters
    ----------
    kind : str | None
        Can be "api" (default), "tutorials", or "examples".
        The default can be changed by setting the configuration value
        MNE_DOCS_KIND.
    version : str | None
        Can be "stable" (default) or "dev".
        The default can be changed by setting the configuration value
        MNE_DOCS_VERSION.
    """
    if kind is None:
        kind = get_config('MNE_DOCS_KIND', 'api')
    help_dict = dict(api='python_reference.html', tutorials='tutorials.html',
                     examples='auto_examples/index.html')
    if kind not in help_dict:
        raise ValueError('kind must be one of %s, got %s'
                         % (sorted(help_dict.keys()), kind))
    kind = help_dict[kind]
    if version is None:
        version = get_config('MNE_DOCS_VERSION', 'stable')
    versions = ('stable', 'dev')
    if version not in versions:
        raise ValueError('version must be one of %s, got %s'
                         % (version, versions))
    webbrowser.open_new_tab('https://martinos.org/mne/%s/%s' % (version, kind))
