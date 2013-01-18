"""Some utility functions"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import warnings
import numpy as np
import logging
import os
import os.path as op
from functools import wraps
import inspect
import sys
from sys import stdout
import tempfile
from shutil import rmtree
import atexit
from math import log
import json
import urllib2
import urlparse

logger = logging.getLogger('mne')


###############################################################################
# RANDOM UTILITIES

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
    sz = len(l) / n
    for i in range(n - 1):
        yield l[i * sz:(i + 1) * sz]
    yield l[(n - 1) * sz:]


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
    instead. Passing del_after and print_del kwargs to the constructor are
    helpful primarily for debugging purposes.
    """
    def __new__(self, del_after=True, print_del=False):
        new = str.__new__(self, tempfile.mkdtemp())
        self._del_after = del_after
        self._print_del = print_del
        return new

    def __init__(self):
        self._path = self.__str__()
        atexit.register(self.cleanup)

    def cleanup(self):
        if self._del_after is True:
            if self._print_del is True:
                print 'Deleting %s ...' % self._path
            rmtree(self._path, ignore_errors=True)


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

    >>> from mne.utils import deprecated_func
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


@deprecated
def deprecated_func():
    pass


def verbose(function):
    """Decorator to allow functions to override default log level

    Do not call this function directly to set the global verbosity level,
    instead use set_log_level().

    Parameters (to decorated function)
    ----------------------------------
    verbose : bool, str, int, or None
        The level of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        None defaults to using the current log level [e.g., set using
        mne.set_log_level()].
    """
    arg_names = inspect.getargspec(function).args
    # this wrap allows decorated functions to be pickled (e.g., for parallel)

    @wraps(function)
    def dec(*args, **kwargs):
        # Check if the first arg is "self", if it has verbose, make it default
        if len(arg_names) > 0 and arg_names[0] == 'self':
            default_level = getattr(args[0], 'verbose', None)
        else:
            default_level = None
        verbose_level = kwargs.get('verbose', default_level)
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
            return function(*args, **kwargs)

    # set __wrapped__ attribute so ?? in IPython gets the right source
    dec.__wrapped__ = function

    return dec


def requires_mne(function):
    """Decorator to skip test if MNE command line tools are not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        if 'MNE_ROOT' not in os.environ:
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires MNE command line tools'
                           % function.__name__)
        ret = function(*args, **kwargs)

        return ret

    return dec


def requires_pandas(function):
    """Decorator to skip test if pandas is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        skip = False
        try:
            import pandas
            if int(pandas.__version__.replace('.', '')) < 73:
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


def make_skipper_dec(module, skip_str):
    """Helper to make skipping decorators"""
    skip = False
    try:
        __import__(module)
    except:
        skip = True
    return np.testing.dec.skipif(skip, skip_str)


requires_sklearn = make_skipper_dec('sklearn', 'scikit-learn not installed')
requires_nitime = make_skipper_dec('nitime', 'nitime not installed')


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
    if isinstance(verbose, basestring):
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

def get_subjects_dir(subjects_dir=None):
    """Safely use subjects_dir input to return SUBJECTS_DIR"""
    if subjects_dir is None:
        subjects_dir = get_config('SUBJECTS_DIR')
    return subjects_dir


def get_config_path():
    """Get path to standard mne-python config file

    Returns
    -------
    config_path : str
        The path to the mne-python configuration file. On windows, this
        will be '%APPDATA%\.mne\mne-python.json'. On every other
        system, this will be $HOME/.mne/mne-python.json.
    """

    # this has been checked on OSX64, Linux64, and Win32
    val = os.getenv('APPDATA' if 'nt' == os.name.lower() else 'HOME', None)
    if val is None:
        raise ValueError('mne-python config file path could '
                         'not be determined, please report this '
                         'error to mne-python developers')

    val = op.join(val, '.mne', 'mne-python.json')
    return val


# List the known configuration values
known_config_types = [
    'MNE_DATASETS_MEGSIM_PATH',
    'MNE_DATASETS_SAMPLE_PATH',
    'MNE_LOGGING_LEVEL',
    'MNE_USE_CUDA',
    'SUBJECTS_DIR',
    ]


def get_config(key, default=None, raise_error=False):
    """Read mne(-python) preference from env, then mne-python config

    Parameters
    ----------
    key : str
        The preference key to look for. The os evironment is searched first,
        then the mne-python config file is parsed.

    default : str | None
        Value to return if the key is not found.

    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).

    Returns
    -------
    value : str | None
        The preference key value.
    """

    if not isinstance(key, basestring):
        raise ValueError('key must be a string')

    # first, check to see if key is in env
    if key in os.environ:
        return os.environ[key]

    # second, look for it in mne-python config file
    config_path = get_config_path()
    if not op.isfile(config_path):
        key_found = False
        val = default
    else:
        with open(config_path, 'r') as fid:
            config = json.load(fid)
        key_found = True if key in config else False
        val = config.get(key, default)

    if not key_found and raise_error is True:
        meth_1 = 'os.environ["%s"] = VALUE' % key
        meth_2 = 'mne.utils.set_config("%s", VALUE)' % key
        raise KeyError('Key "%s" not found in environment or in the '
                       'mne-python config file:\n%s\nTry either:\n'
                       '    %s\nfor a temporary solution, or:\n'
                       '    %s\nfor a permanent one. You can also '
                       'set the environment variable before '
                       'running python.'
                       % (key, config_path, meth_1, meth_2))
    return val


def set_config(key, value):
    """Set mne-python preference in config

    Parameters
    ----------
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    """

    if not isinstance(key, basestring):
        raise ValueError('key must be a string')
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    if not isinstance(value, basestring) and value is not None:
        raise ValueError('value must be a string or None')
    if not key in known_config_types:
        warnings.warn('Setting non-standard config type: "%s"' % key)

    # Read all previous values
    config_path = get_config_path()
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

    # Write all values
    directory = op.split(config_path)[0]
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, 'w') as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


def _download_status(url, file_name, print_destination=True):
    """Download a URL to a file destination, with status updates"""

    # Old, simpler code:
    #opener = urllib.urlopen(url)
    #open(archive_name, 'wb').write(opener.read())

    u = urllib2.urlopen(url)
    with open(file_name, 'wb') as f:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        stdout.write('Downloading: %s (%s)\n' % (url, sizeof_fmt(file_size)))
        stdout.write('0%' + 64 * '.' + '100%\n' + ' |')
        char_span = 64.0

        file_size_dl = 0
        block_sz = 65536
        n_written = 0
        while True:
            buf = u.read(block_sz)
            if not buf:
                break

            file_size_dl += len(buf)
            f.write(buf)
            n_char = (int(float(file_size_dl) / file_size * char_span)
                      - n_written)
            if n_char > 0:
                stdout.write('>' * n_char)
                stdout.flush()
                n_written += n_char
        stdout.write('|\n')
        if print_destination is True:
            stdout.write('File saved as %s.\n' % file_name)


def sizeof_fmt(num):
    """Turn number of bytes into human-readable str"""
    unit_list = zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'],
                    [0, 0, 1, 2, 2, 2])
    """Human friendly file size"""
    if num > 1:
        exponent = min(int(log(num, 1024)), len(unit_list) - 1)
        quotient = float(num) / 1024 ** exponent
        unit, num_decimals = unit_list[exponent]
        format_string = '{:.%sf} {}' % (num_decimals)
        return format_string.format(quotient, unit)
    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'


def _url_to_local_path(url, path):
    """Mirror a url path in a local destination (keeping folder structure)"""
    destination = urlparse.urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != '/':
        raise ValueError('Invalid URL')
    destination = os.path.join(path, urllib2.url2pathname(destination)[1:])
    return destination
