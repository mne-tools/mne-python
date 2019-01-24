# -*- coding: utf-8 -*-
"""Some miscellaneous utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import atexit
from functools import partial
import fnmatch
import inspect
from io import StringIO
import json
import logging
import multiprocessing
import os
import os.path as op
import platform
from string import Formatter
import subprocess
import sys
import shutil
import traceback
import tempfile
import warnings
import webbrowser

import numpy as np

from ..externals.decorator import decorator
from ..fixes import _get_args
from ._logging import warn, WrapStdOut
from .check import _validate_type

logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


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
    if isinstance(verbose, str):
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
    """Store logging.

    This will remove all other logging handlers, and return the handler to
    stdout when complete.
    """

    def __enter__(self):  # noqa: D105
        self._data = StringIO()
        self._lh = logging.StreamHandler(self._data)
        self._lh.setFormatter(logging.Formatter('%(message)s'))
        self._lh._mne_file_like = True  # monkey patch for warn() use
        for lh in logger.handlers:
            logger.removeHandler(lh)
        logger.addHandler(self._lh)
        return self._data

    def __exit__(self, *args):  # noqa: D105
        logger.removeHandler(self._lh)
        set_log_file(None)


###############################################################################
# RANDOM UTILITIES

def _get_argvalues():
    """Return all arguments (except self) and values of read_raw_xxx."""
    # call stack
    # read_raw_xxx -> EOF -> verbose() -> BaseRaw.__init__ -> get_argvalues
    frame = inspect.stack()[4][0]
    fname = frame.f_code.co_filename
    if not fnmatch.fnmatch(fname, '*/mne/io/*'):
        return None
    args, _, _, values = inspect.getargvalues(frame)
    params = dict()
    for arg in args:
        params[arg] = values[arg]
    params.pop('self', None)
    return params


def _pl(x, non_pl=''):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, (int, np.generic)) else len(x)
    return non_pl if len_x == 1 else 's'


def _explain_exception(start=-1, stop=None, prefix='> '):
    """Explain an exception."""
    # start=-1 means "only the most recent caller"
    etype, value, tb = sys.exc_info()
    string = traceback.format_list(traceback.extract_tb(tb)[start:stop])
    string = (''.join(string).split('\n') +
              traceback.format_exception_only(etype, value))
    string = ':\n' + prefix + ('\n' + prefix).join(string)
    return string


def _sort_keys(x):
    """Sort and return keys of dict."""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


class _Counter():
    count = 1

    def __call__(self, *args, **kargs):
        c = self.count
        self.count += 1
        return c


class _FormatDict(dict):
    """Help pformat() work properly."""

    def __missing__(self, key):
        return "{" + key + "}"


def pformat(temp, **fmt):
    """Format a template string partially.

    Examples
    --------
    >>> pformat("{a}_{b}", a='x')
    'x_{b}'
    """
    formatter = Formatter()
    mapping = _FormatDict(fmt)
    return formatter.vformat(temp, (), mapping)


def copy_doc(source):
    """Copy the docstring from another function (decorator).

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
    for stdxxx, sys_stdxxx, thresh in (
            ['stderr', sys.stderr, logging.ERROR],
            ['stdout', sys.stdout, logging.WARNING]):
        if stdxxx not in kwargs and logger.level >= thresh:
            kwargs[stdxxx] = subprocess.PIPE
        elif kwargs.get(stdxxx, sys_stdxxx) is sys_stdxxx:
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
    if isinstance(command, str):
        command_str = command
    else:
        command_str = ' '.join(command)
    logger.info("Running subprocess: %s" % command_str)
    try:
        p = subprocess.Popen(command, *args, **kwargs)
    except Exception:
        if isinstance(command, str):
            command_name = command.split()[0]
        else:
            command_name = command[0]
        logger.error('Command not found: %s' % command_name)
        raise
    stdout_, stderr = p.communicate()
    stdout_ = u'' if stdout_ is None else stdout_.decode('utf-8')
    stderr = u'' if stderr is None else stderr.decode('utf-8')
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
            _validate_type(stim_channel, 'str', "Stim channel")
            stim_channel = [stim_channel]
        for channel in stim_channel:
            _validate_type(channel, 'str', "Each provided stim channel")
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
        if name.endswith('_v'):
            name = name[:-2]
        cleaned.append(name)

    return cleaned


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
        cupy:          4.1.0
        pandas:        0.17.1+25.g547750a
        dipy:          0.14.0

    """  # noqa: E501
    ljust = 15
    out = 'Platform:'.ljust(ljust) + platform.platform() + '\n'
    out += 'Python:'.ljust(ljust) + str(sys.version).replace('\n', ' ') + '\n'
    out += 'Executable:'.ljust(ljust) + sys.executable + '\n'
    out += 'CPU:'.ljust(ljust) + ('%s: %s cores\n' %
                                  (platform.processor(),
                                   multiprocessing.cpu_count()))
    out += 'Memory:'.ljust(ljust)
    try:
        import psutil
    except ImportError:
        out += 'Unavailable (requires "psutil" package)'
    else:
        out += '%0.1f GB\n' % (psutil.virtual_memory().total / float(2 ** 30),)
    out += '\n'
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
                lib = lines[li + 1]
                if 'NOT AVAILABLE' in lib:
                    lib = 'unknown'
                else:
                    try:
                        lib = lib.split('[')[1].split("'")[1]
                    except IndexError:
                        pass  # keep whatever it was
                libs += ['%s=%s' % (key, lib)]
    libs = ', '.join(libs)
    for mod_name in ('mne', 'numpy', 'scipy', 'matplotlib', '', 'sklearn',
                     'nibabel', 'mayavi', 'cupy', 'pandas', 'dipy'):
        if mod_name == '':
            out += '\n'
            continue
        out += ('%s:' % mod_name).ljust(ljust)
        try:
            mod = __import__(mod_name)
            if mod_name == 'mayavi':
                # the real test
                from mayavi import mlab  # noqa, analysis:ignore
        except Exception:
            out += 'Not found\n'
        else:
            extra = (' (%s)' % op.dirname(mod.__file__)) if show_paths else ''
            if mod_name == 'numpy':
                extra = ' {%s}%s' % (libs, extra)
            elif mod_name == 'matplotlib':
                extra = ' {backend=%s}%s' % (mod.get_backend(), extra)
            elif mod_name == 'mayavi':
                try:
                    from pyface.qt import qt_api
                except Exception:
                    qt_api = 'unknown'
                if qt_api == 'pyqt5':
                    try:
                        from PyQt5.Qt import PYQT_VERSION_STR
                        qt_api += ', PyQt5=%s' % (PYQT_VERSION_STR,)
                    except Exception:
                        pass
                extra = ' {qt_api=%s}%s' % (qt_api, extra)
            out += '%s%s\n' % (mod.__version__, extra)
    # print(out, end='', file=fid)
    # XXX


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


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    This has been adapted to deal with our "verbose" decorator.

    Adapted from SciPy (doc/source/conf.py).
    """
    import mne
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None
    if fn == '<string>':  # verbose decorator
        fn = inspect.getmodule(obj).__file__
    fn = op.relpath(fn, start=op.dirname(mne.__file__))
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in mne.__version__:
        kind = 'master'
    else:
        kind = 'maint/%s' % ('.'.join(mne.__version__.split('.')[:2]))
    return "http://github.com/mne-tools/mne-python/blob/%s/mne/%s%s" % (  # noqa
       kind, fn, linespec)


###############################################################################
# CONFIG UTILITIES

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
        if not isinstance(memmap_min_size, str):
            raise ValueError('\'memmap_min_size\' has to be a string.')
        if memmap_min_size[-1] not in ['K', 'M', 'G']:
            raise ValueError('The size has to be given in kilo-, mega-, or '
                             'gigabytes, e.g., 100K, 500M, 1G.')

    set_config('MNE_MEMMAP_MIN_SIZE', memmap_min_size, set_env=False)


# List the known configuration values
known_config_types = (
    'MNE_BROWSE_RAW_SIZE',
    'MNE_CACHE_DIR',
    'MNE_COREG_ADVANCED_RENDERING',
    'MNE_COREG_COPY_ANNOT',
    'MNE_COREG_GUESS_MRI_SUBJECT',
    'MNE_COREG_HEAD_HIGH_RES',
    'MNE_COREG_HEAD_OPACITY',
    'MNE_COREG_INTERACTION',
    'MNE_COREG_MARK_INSIDE',
    'MNE_COREG_PREPARE_BEM',
    'MNE_COREG_PROJECT_EEG',
    'MNE_COREG_ORIENT_TO_SURFACE',
    'MNE_COREG_SCALE_LABELS',
    'MNE_COREG_SCALE_BY_DISTANCE',
    'MNE_COREG_SCENE_SCALE',
    'MNE_COREG_WINDOW_HEIGHT',
    'MNE_COREG_WINDOW_WIDTH',
    'MNE_COREG_SUBJECTS_DIR',
    'MNE_CUDA_IGNORE_PRECISION',
    'MNE_DATA',
    'MNE_DATASETS_BRAINSTORM_PATH',
    'MNE_DATASETS_EEGBCI_PATH',
    'MNE_DATASETS_HF_SEF_PATH',
    'MNE_DATASETS_MEGSIM_PATH',
    'MNE_DATASETS_MISC_PATH',
    'MNE_DATASETS_MTRF_PATH',
    'MNE_DATASETS_SAMPLE_PATH',
    'MNE_DATASETS_SOMATO_PATH',
    'MNE_DATASETS_MULTIMODAL_PATH',
    'MNE_DATASETS_OPM_PATH',
    'MNE_DATASETS_SPM_FACE_DATASETS_TESTS',
    'MNE_DATASETS_SPM_FACE_PATH',
    'MNE_DATASETS_TESTING_PATH',
    'MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
    'MNE_DATASETS_KILOWORD_PATH',
    'MNE_DATASETS_FIELDTRIP_CMC_PATH',
    'MNE_DATASETS_PHANTOM_4DBTI_PATH',
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
        will be '%USERPROFILE%\.mne\mne-python.json'. On every other
        system, this will be ~/.mne/mne-python.json.
    """
    val = op.join(_get_extra_data_path(home_dir=home_dir),
                  'mne-python.json')
    return val


def get_config(key=None, default=None, raise_error=False, home_dir=None):
    """Read MNE-Python preferences from environment or config file.

    Parameters
    ----------
    key : None | str
        The preference key to look for. The os environment is searched first,
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
    _validate_type(key, (str, type(None)), "key", 'string or None')

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
    _validate_type(key, 'str', "key")
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    _validate_type(value, (str, type(None)), "value",
                   "None or string")

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


def _get_extra_data_path(home_dir=None):
    """Get path to extra data (config, tables, etc.)."""
    global _temp_home_dir
    if home_dir is None:
        home_dir = os.environ.get('_MNE_FAKE_HOME_DIR')
    if home_dir is None:
        # this has been checked on OSX64, Linux64, and Win32
        if 'nt' == os.name.lower():
            if op.isdir(op.join(os.getenv('APPDATA'), '.mne')):
                home_dir = os.getenv('APPDATA')
            else:
                home_dir = os.getenv('USERPROFILE')
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
