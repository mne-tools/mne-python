"""Some utility functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import contextlib
import importlib
import inspect
import logging
import os.path as op
import re
import sys
import warnings
from collections.abc import Callable
from io import StringIO
from typing import Any, TypeVar

from decorator import FunctionMaker

from .docs import fill_doc

logger = logging.getLogger("mne")  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


# class to provide frame information (should be low overhead, just on logger
# calls)


class _FrameFilter(logging.Filter):
    def __init__(self):
        self.add_frames = 0

    def filter(self, record):
        record.frame_info = "Unknown"
        if self.add_frames:
            # 5 is the offset necessary to get out of here and the logging
            # module, reversal is to put the oldest at the top
            frame_info = _frame_info(5 + self.add_frames)[5:][::-1]
            if len(frame_info):
                frame_info[-1] = (frame_info[-1] + " :").ljust(30)
                if len(frame_info) > 1:
                    frame_info[0] = "┌" + frame_info[0]
                    frame_info[-1] = "└" + frame_info[-1]
                for ii, info in enumerate(frame_info[1:-1], 1):
                    frame_info[ii] = "├" + info
                record.frame_info = "\n".join(frame_info)
        return True


_filter = _FrameFilter()
logger.addFilter(_filter)


# Provide help for static type checkers:
# https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])


def verbose(function: _FuncT) -> _FuncT:
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
    try:
        fill_doc(function)
    except TypeError:  # nothing to add
        pass

    # Anything using verbose should have `verbose=None` in the signature.
    # This code path will raise an error if this is not the case.
    body = """\
def %(name)s(%(signature)s):\n
    try:
        do_level_change = verbose is not None
    except (NameError, UnboundLocalError):
        raise RuntimeError('Function/method %%s does not accept verbose '
                           'parameter' %% (_function_,)) from None
    if do_level_change:
        with _use_log_level_(verbose):
            return _function_(%(shortsignature)s)
    else:
        return _function_(%(shortsignature)s)"""
    evaldict = dict(_use_log_level_=use_log_level, _function_=function)
    fm = FunctionMaker(function)
    attrs = dict(
        __wrapped__=function,
        __qualname__=function.__qualname__,
        __globals__=function.__globals__,
    )
    return fm.make(body, evaldict, addsource=True, **attrs)


@fill_doc
class use_log_level:
    """Context manager for logging level.

    Parameters
    ----------
    %(verbose)s
    %(add_frames)s

    See Also
    --------
    mne.verbose

    Notes
    -----
    See the :ref:`logging documentation <tut-logging>` for details.

    Examples
    --------
    >>> from mne import use_log_level
    >>> from mne.utils import logger
    >>> with use_log_level(False):
    ...     # Most MNE logger messages are "info" level, False makes them not
    ...     # print:
    ...     logger.info('This message will not be printed')
    >>> with use_log_level(True):
    ...     # Using verbose=True in functions, methods, or this context manager
    ...     # will ensure they are printed
    ...     logger.info('This message will be printed!')
    This message will be printed!
    """

    def __init__(self, verbose=None, *, add_frames=None):
        self._level = verbose
        self._add_frames = add_frames
        self._old_frames = _filter.add_frames

    def __enter__(self):  # noqa: D105
        self._old_level = set_log_level(
            self._level, return_old_level=True, add_frames=self._add_frames
        )

    def __exit__(self, *args):  # noqa: D105
        add_frames = self._old_frames if self._add_frames is not None else None
        set_log_level(self._old_level, add_frames=add_frames)


_LOGGING_TYPES = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


@fill_doc
def set_log_level(verbose=None, return_old_level=False, add_frames=None):
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
    %(add_frames)s

    Returns
    -------
    old_level : int
        The old level. Only returned if ``return_old_level`` is True.
    """
    old_verbose = logger.level
    verbose = _parse_verbose(verbose)

    if verbose != old_verbose:
        logger.setLevel(verbose)
    if add_frames is not None:
        _filter.add_frames = int(add_frames)
        fmt = "%(frame_info)s " if add_frames else ""
        fmt += "%(message)s"
        fmt = logging.Formatter(fmt)
        for handler in logger.handlers:
            handler.setFormatter(fmt)
    return old_verbose if return_old_level else None


def _parse_verbose(verbose):
    from .check import _check_option, _validate_type
    from .config import get_config

    _validate_type(verbose, (bool, str, int, None), "verbose")
    if verbose is None:
        verbose = get_config("MNE_LOGGING_LEVEL", "INFO")
    elif isinstance(verbose, bool):
        if verbose is True:
            verbose = "INFO"
        else:
            verbose = "WARNING"
    if isinstance(verbose, str):
        verbose = verbose.upper()
        _check_option("verbose", verbose, _LOGGING_TYPES, "(when a string)")
        verbose = _LOGGING_TYPES[verbose]

    return verbose


def set_log_file(fname=None, output_format="%(message)s", overwrite=None):
    """Set the log to print to a file.

    Parameters
    ----------
    fname : path-like | None
        Filename of the log to print to. If None, stdout is used.
        To suppress log outputs, use set_log_level('WARNING').
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
    _remove_close_handlers(logger)
    if fname is not None:
        if op.isfile(fname) and overwrite is None:
            # Don't use warn() here because we just want to
            # emit a warnings.warn here (not logger.warn)
            warnings.warn(
                "Log entries will be appended to the file. Use "
                "overwrite=False to avoid this message in the "
                "future.",
                RuntimeWarning,
                stacklevel=2,
            )
            overwrite = False
        mode = "w" if overwrite else "a"
        lh = logging.FileHandler(fname, mode=mode)
    else:
        """we should just be able to do:
            lh = logging.StreamHandler(sys.stdout)
        but because doctests uses some magic on stdout, we have to do this:
        """
        lh = logging.StreamHandler(WrapStdOut())

    lh.setFormatter(logging.Formatter(output_format))
    # actually add the stream handler
    logger.addHandler(lh)


def _remove_close_handlers(logger):
    for h in list(logger.handlers):
        # only remove our handlers (get along nicely with nose)
        if isinstance(h, logging.FileHandler | logging.StreamHandler):
            if isinstance(h, logging.FileHandler):
                h.close()
            logger.removeHandler(h)


class ClosingStringIO(StringIO):
    """StringIO that closes after getvalue()."""

    def getvalue(self, close=True):
        """Get the value."""
        out = super().getvalue()
        if close:
            self.close()
        return out


class catch_logging:
    """Store logging.

    This will remove all other logging handlers, and return the handler to
    stdout when complete.
    """

    def __init__(self, verbose=None):
        self.verbose = verbose

    def __enter__(self):  # noqa: D105
        if self.verbose is not None:
            self._ctx = use_log_level(self.verbose)
        else:
            self._ctx = contextlib.nullcontext()
        self._data = ClosingStringIO()
        self._lh = logging.StreamHandler(self._data)
        self._lh.setFormatter(logging.Formatter("%(message)s"))
        self._lh._mne_file_like = True  # monkey patch for warn() use
        _remove_close_handlers(logger)
        logger.addHandler(self._lh)
        self._ctx.__enter__()
        return self._data

    def __exit__(self, *args):  # noqa: D105
        self._ctx.__exit__(*args)
        logger.removeHandler(self._lh)
        set_log_file(None)


@contextlib.contextmanager
def _record_warnings():
    # this is a helper that mostly acts like pytest.warns(None) did before
    # pytest 7
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


class WrapStdOut:
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
            raise AttributeError(f"'file' object has not attribute '{name}'")


_verbose_dec_re = re.compile("^<decorator-gen-[0-9]+>$")


def warn(message, category=RuntimeWarning, module="mne", ignore_namespaces=("mne",)):
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
    ignore_namespaces : list of str
        Namespaces to ignore when traversing the stack.

        .. versionadded:: 0.24
    """
    root_dirs = [importlib.import_module(ns) for ns in ignore_namespaces]
    root_dirs = [op.dirname(ns.__file__) for ns in root_dirs]
    frame = None
    if logger.level <= logging.WARNING:
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_code.co_filename
            lineno = frame.f_lineno
            # in verbose dec
            if not _verbose_dec_re.search(fname):
                # treat tests as scripts
                # and don't capture unittest/case.py (assert_raises)
                if (
                    not (
                        any(fname.startswith(rd) for rd in root_dirs)
                        or ("unittest" in fname and "case" in fname)
                    )
                    or op.basename(op.dirname(fname)) == "tests"
                ):
                    break
            frame = frame.f_back
        del frame
        # We need to use this instead of warn(message, category, stacklevel)
        # because we move out of the MNE stack, so warnings won't properly
        # recognize the module name (and our warnings.simplefilter will fail)
        warnings.warn_explicit(
            message,
            category,
            fname,
            lineno,
            module,
            globals().get("__warningregistry__", {}),
        )
    # To avoid a duplicate warning print, we only emit the logger.warning if
    # one of the handlers is a FileHandler. See gh-5592
    # But it's also nice to be able to do:
    # with mne.utils.use_log_level('warning', add_frames=3):
    # so also check our add_frames attribute.
    if (
        any(
            isinstance(h, logging.FileHandler) or getattr(h, "_mne_file_like", False)
            for h in logger.handlers
        )
        or _filter.add_frames
    ):
        logger.warning(message)


def _get_call_line():
    """Get the call line from within a function."""
    frame = inspect.currentframe().f_back.f_back
    if _verbose_dec_re.search(frame.f_code.co_filename):
        frame = frame.f_back
    context = inspect.getframeinfo(frame).code_context
    context = "unknown" if context is None else context[0].strip()
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
    """
    regexp = re.compile(".*" if match is None else match)
    is_category = [
        w.category == category if category is not None else True
        for w in warn_record._list
    ]
    is_match = [regexp.match(w.message.args[0]) is not None for w in warn_record._list]
    ind = [ind for ind, (c, m) in enumerate(zip(is_category, is_match)) if c and m]

    for i in reversed(ind):
        warn_record._list.pop(i)


@contextlib.contextmanager
def wrapped_stdout(indent="", cull_newlines=False):
    """Wrap stdout writes to logger.info, with an optional indent prefix.

    Parameters
    ----------
    indent : str
        The indentation to add.
    cull_newlines : bool
        If True, cull any new/blank lines at the end.
    """
    orig_stdout = sys.stdout
    my_out = ClosingStringIO()
    sys.stdout = my_out
    try:
        yield
    finally:
        sys.stdout = orig_stdout
        pending_newlines = 0
        for line in my_out.getvalue().split("\n"):
            if not line.strip() and cull_newlines:
                pending_newlines += 1
                continue
            for _ in range(pending_newlines):
                logger.info("\n")
            logger.info(indent + line)


def _frame_info(n):
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        infos = list()
        for _ in range(n):
            try:
                name = frame.f_globals["__name__"]
            except KeyError:  # in our verbose dec
                pass
            else:
                infos.append(f"{name.lstrip('mne.')}:{frame.f_lineno}")
            frame = frame.f_back
            if frame is None:
                break
        return infos
    except Exception:
        return ["unknown"]
    finally:
        del frame


def _verbose_safe_false(*, level="warning"):
    lev = _LOGGING_TYPES[level.upper()]
    return lev if logger.level <= lev else None
