# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import contextlib
import inspect
from io import StringIO
import re
import sys
import logging
import os.path as op
import warnings

from ..externals.decorator import FunctionMaker


logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


def verbose(function):
    """Verbose decorator to allow functions to override log-level.

    Parameters
    ----------
    function : callable
        Function to be decorated by setting the verbosity level.

    Returns
    -------
    dec : callable
        The decorated function.

    See Also
    --------
    set_log_level
    set_config

    Notes
    -----
    This decorator is used to set the verbose level during a function or method
    call, such as :func:`mne.compute_covariance`. The `verbose` keyword
    argument can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', True (an
    alias for 'INFO'), or False (an alias for 'WARNING'). To set the global
    verbosity level for all functions, use :func:`mne.set_log_level`.

    This function also serves as a docstring filler.

    Examples
    --------
    You can use the ``verbose`` argument to set the verbose level on the fly::
        >>> import mne
        >>> cov = mne.compute_raw_covariance(raw, verbose='WARNING')  # doctest: +SKIP
        >>> cov = mne.compute_raw_covariance(raw, verbose='INFO')  # doctest: +SKIP
        Using up to 49 segments
        Number of samples used : 5880
        [done]
    """  # noqa: E501
    # See https://decorator.readthedocs.io/en/latest/tests.documentation.html
    # #dealing-with-third-party-decorators
    from .docs import fill_doc
    try:
        fill_doc(function)
    except TypeError:  # nothing to add
        pass

    # Anything using verbose should either have `verbose=None` in the signature
    # or have a `self.verbose` attribute (if in a method). This code path
    # will raise an error if neither is the case.
    wrap_src = """\
try:
    verbose
except UnboundLocalError:
    try:
        verbose = self.verbose
    except NameError:
        raise RuntimeError('Function %%s does not accept verbose parameter'
                           %% (_function_,))
    except AttributeError:
        raise RuntimeError('Method %%s class does not have self.verbose'
                           %% (_function_,))
if verbose is None:
    try:
        verbose = self.verbose
    except (NameError, AttributeError):
        pass
if verbose is not None:
    with _use_log_level_(verbose):
        return _function_(%(signature)s)
return _function_(%(signature)s)"""
    evaldict = dict(
        _use_log_level_=use_log_level, _function_=function)
    return FunctionMaker.create(
        function, wrap_src, evaldict,
        __wrapped__=function, __qualname__=function.__qualname__,
        module=function.__module__)


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

    Returns
    -------
    old_level : int
        The old level. Only returned if ``return_old_level`` is True.
    """
    from .config import get_config
    from .check import _check_option
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
        _check_option('verbose', verbose, logging_types, '(when a string)')
        verbose = logging_types[verbose]
    logger = logging.getLogger('mne')
    old_verbose = logger.level
    if verbose != old_verbose:
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


_verbose_dec_re = re.compile('^<decorator-gen-[0-9]+>$')


def warn(message, category=RuntimeWarning, module='mne'):
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
    module : str
        The name of the module emitting the warning.
    """
    import mne
    root_dir = op.dirname(mne.__file__)
    frame = None
    if logger.level <= logging.WARN:
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_code.co_filename
            lineno = frame.f_lineno
            # in verbose dec
            if not _verbose_dec_re.search(fname):
                # treat tests as scripts
                # and don't capture unittest/case.py (assert_raises)
                if not (fname.startswith(root_dir) or
                        ('unittest' in fname and 'case' in fname)) or \
                        op.basename(op.dirname(fname)) == 'tests':
                    break
            frame = frame.f_back
        del frame
        # We need to use this instead of warn(message, category, stacklevel)
        # because we move out of the MNE stack, so warnings won't properly
        # recognize the module name (and our warnings.simplefilter will fail)
        warnings.warn_explicit(
            message, category, fname, lineno, module,
            globals().get('__warningregistry__', {}))
    # To avoid a duplicate warning print, we only emit the logger.warning if
    # one of the handlers is a FileHandler. See gh-5592
    if any(isinstance(h, logging.FileHandler) or getattr(h, '_mne_file_like',
                                                         False)
           for h in logger.handlers):
        logger.warning(message)


def _get_call_line():
    """Get the call line from within a function."""
    frame = inspect.currentframe().f_back.f_back
    if _verbose_dec_re.search(frame.f_code.co_filename):
        frame = frame.f_back
    context = inspect.getframeinfo(frame).code_context
    context = 'unknown' if context is None else context[0].strip()
    return context


def filter_out_warnings(warn_record, category=None, match=None):
    r"""Remove particular records from ``warn_record``.

    This helper takes a list of :class:`warnings.WarningMessage` objects,
    and remove those matching category and/or text.

    Parameters
    ----------
    category: WarningMessage type | None
       class of the message to filter out

    match : str | None
        text or regex that matches the error message to filter out

    Examples
    --------
    This can be used as::

        >>> import pytest
        >>> import warnings
        >>> from mne.utils import filter_out_warnings
        >>> with pytest.warns(None) as recwarn:
        ...     warnings.warn("value must be 0 or None", UserWarning)
        >>> filter_out_warnings(recwarn, match=".* 0 or None")
        >>> assert len(recwarn.list) == 0

        >>> with pytest.warns(None) as recwarn:
        ...     warnings.warn("value must be 42", UserWarning)
        >>> filter_out_warnings(recwarn, match=r'.* must be \d+$')
        >>> assert len(recwarn.list) == 0

        >>> with pytest.warns(None) as recwarn:
        ...     warnings.warn("this is not here", UserWarning)
        >>> filter_out_warnings(recwarn, match=r'.* must be \d+$')
        >>> assert len(recwarn.list) == 1
    """
    regexp = re.compile('.*' if match is None else match)
    is_category = [w.category == category if category is not None else True
                   for w in warn_record._list]
    is_match = [regexp.match(w.message.args[0]) is not None
                for w in warn_record._list]
    ind = [ind for ind, (c, m) in enumerate(zip(is_category, is_match))
           if c and m]

    for i in reversed(ind):
        warn_record._list.pop(i)


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


@contextlib.contextmanager
def wrapped_stdout(indent=''):
    """Wrap stdout writes to logger.info, with an optional indent prefix."""
    orig_stdout = sys.stdout
    my_out = StringIO()
    sys.stdout = my_out
    try:
        yield
    finally:
        sys.stdout = orig_stdout
        for line in my_out.getvalue().split('\n'):
            logger.info(indent + line)
