"""Some miscellaneous utility functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import fnmatch
import gc
import hashlib
import inspect
import os
import subprocess
import sys
import traceback
import weakref
from contextlib import ExitStack, contextmanager
from importlib.resources import files
from math import log
from queue import Empty, Queue
from string import Formatter
from textwrap import dedent
from threading import Thread

import numpy as np
from decorator import FunctionMaker

from ._logging import logger, verbose, warn
from .check import _check_option, _validate_type


def _identity_function(x):
    return x


# TODO: no longer needed when py3.9 is minimum supported version
def _empty_hash(kind="md5"):
    func = getattr(hashlib, kind)
    if "usedforsecurity" in inspect.signature(func).parameters:
        return func(usedforsecurity=False)
    else:
        return func()


def _pl(x, non_pl="", pl="s"):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, int | np.generic) else len(x)
    return non_pl if len_x == 1 else pl


def _explain_exception(start=-1, stop=None, prefix="> "):
    """Explain an exception."""
    # start=-1 means "only the most recent caller"
    etype, value, tb = sys.exc_info()
    string = traceback.format_list(traceback.extract_tb(tb)[start:stop])
    string = "".join(string).split("\n") + traceback.format_exception_only(etype, value)
    string = ":\n" + prefix + ("\n" + prefix).join(string)
    return string


def _sort_keys(x):
    """Sort and return keys of dict."""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


class _DefaultEventParser:
    """Parse none standard events."""

    def __init__(self):
        self.event_ids = dict()

    def __call__(self, description, offset=1):
        if description not in self.event_ids:
            self.event_ids[description] = offset + len(self.event_ids)

        return self.event_ids[description]


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


def _enqueue_output(out, queue):
    for line in iter(out.readline, b""):
        queue.put(line)


@verbose
def run_subprocess(command, return_code=False, verbose=None, *args, **kwargs):
    """Run command using subprocess.Popen.

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str | str
        Command to run as subprocess (see subprocess.Popen documentation).
    return_code : bool
        If True, return the return code instead of raising an error if it's
        non-zero.

        .. versionadded:: 0.20
    %(verbose)s
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.

    Returns
    -------
    stdout : str
        Stdout returned by the process.
    stderr : str
        Stderr returned by the process.
    code : int
        The return code, only returned if ``return_code == True``.
    """
    all_out = ""
    all_err = ""
    # non-blocking adapted from https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python#4896288  # noqa: E501
    out_q = Queue()
    err_q = Queue()
    control_stdout = "stdout" not in kwargs
    control_stderr = "stderr" not in kwargs
    with running_subprocess(command, *args, **kwargs) as p:
        if control_stdout:
            out_t = Thread(target=_enqueue_output, args=(p.stdout, out_q))
            out_t.daemon = True
            out_t.start()
        if control_stderr:
            err_t = Thread(target=_enqueue_output, args=(p.stderr, err_q))
            err_t.daemon = True
            err_t.start()
        while True:
            do_break = p.poll() is not None
            # read all current lines without blocking
            while True:  # process stdout
                try:
                    out = out_q.get(timeout=0.01)
                except Empty:
                    break
                else:
                    out = out.decode("utf-8")
                    log_out = out.removesuffix("\n")
                    logger.info(log_out)
                    all_out += out

            while True:  # process stderr
                try:
                    err = err_q.get(timeout=0.01)
                except Empty:
                    break
                else:
                    err = err.decode("utf-8")
                    err_out = err.removesuffix("\n")

                    # Leave this as logger.warning rather than warn(...) to
                    # mirror the logger.info above for stdout. This function
                    # is basically just a version of subprocess.call, and
                    # shouldn't emit Python warnings due to stderr outputs
                    # (the calling function can check for stderr output and
                    # emit a warning if it wants).
                    logger.warning(err_out)
                    all_err += err

            if do_break:
                break
    output = (all_out, all_err)

    if return_code:
        output = output + (p.returncode,)
    elif p.returncode:
        stdout = all_out if control_stdout else None
        stderr = all_err if control_stderr else None
        raise subprocess.CalledProcessError(
            p.returncode, command, output=stdout, stderr=stderr
        )

    return output


@contextmanager
def running_subprocess(command, after="wait", verbose=None, *args, **kwargs):
    """Context manager to do something with a command running via Popen.

    Parameters
    ----------
    command : list of str | str
        Command to run as subprocess (see :class:`python:subprocess.Popen`).
    after : str
        Can be:

        - "wait" to use :meth:`~python:subprocess.Popen.wait`
        - "communicate" to use :meth:`~python.subprocess.Popen.communicate`
        - "terminate" to use :meth:`~python:subprocess.Popen.terminate`
        - "kill" to use :meth:`~python:subprocess.Popen.kill`

    %(verbose)s
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.

    Returns
    -------
    p : instance of Popen
        The process.
    """
    _validate_type(after, str, "after")
    _check_option("after", after, ["wait", "terminate", "kill", "communicate"])
    contexts = list()
    for stdxxx in ("stderr", "stdout"):
        if stdxxx not in kwargs:
            kwargs[stdxxx] = subprocess.PIPE
            contexts.append(stdxxx)

    # Check the PATH environment variable. If run_subprocess() is to be called
    # frequently this should be refactored so as to only check the path once.
    env = kwargs.get("env", os.environ)
    if any(p.startswith("~") for p in env["PATH"].split(os.pathsep)):
        warn(
            "Your PATH environment variable contains at least one path "
            'starting with a tilde ("~") character. Such paths are not '
            "interpreted correctly from within Python. It is recommended "
            'that you use "$HOME" instead of "~".'
        )
    if isinstance(command, str):
        command_str = command
    else:
        command = [str(s) for s in command]
        command_str = " ".join(s for s in command)
    logger.info(f"Running subprocess: {command_str}")
    try:
        p = subprocess.Popen(command, *args, **kwargs)
    except Exception:
        if isinstance(command, str):
            command_name = command.split()[0]
        else:
            command_name = command[0]
        logger.error(f"Command not found: {command_name}")
        raise
    try:
        with ExitStack() as stack:
            for context in contexts:
                stack.enter_context(getattr(p, context))
            yield p
    finally:
        getattr(p, after)()
        p.wait()


def _clean_names(names, remove_whitespace=False, before_dash=True):
    """Remove white-space on topo matching.

    This function handles different naming conventions for old VS new VectorView systems
    (`remove_whitespace`) and removes system specific parts in CTF channel names
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
        if " " in name and remove_whitespace:
            name = name.replace(" ", "")
        if "-" in name and before_dash:
            name = name.split("-")[0]
        if name.endswith("_v"):
            name = name[:-2]
        cleaned.append(name)
    if len(set(cleaned)) != len(names):
        # this was probably not a VectorView or CTF dataset, and we now broke the
        # dataset by creating duplicates, so let's use the original channel names.
        return names
    return cleaned


def _get_argvalues():
    """Return all arguments (except self) and values of read_raw_xxx."""
    # call stack
    # read_raw_xxx -> <decorator-gen-000> -> BaseRaw.__init__ -> _get_argvalues

    # This is equivalent to `frame = inspect.stack(0)[4][0]` but faster
    frame = inspect.currentframe()
    try:
        for _ in range(3):
            frame = frame.f_back
        fname = frame.f_code.co_filename
        if not fnmatch.fnmatch(fname, "*/mne/io/*"):
            return None
        args, _, _, values = inspect.getargvalues(frame)
    finally:
        del frame
    params = dict()
    for arg in args:
        params[arg] = values[arg]
    params.pop("self", None)
    return params


def sizeof_fmt(num):
    """Turn number of bytes into human-readable str.

    Parameters
    ----------
    num : int
        The number of bytes.

    Returns
    -------
    size : str
        The size in human-readable format.
    """
    units = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB"]
    decimals = [0, 0, 1, 2, 2, 2]
    if num > 1:
        exponent = min(int(log(num, 1024)), len(units) - 1)
        quotient = float(num) / 1024**exponent
        unit = units[exponent]
        num_decimals = decimals[exponent]
        format_string = f"{{0:.{num_decimals}f}} {{1}}"
        return format_string.format(quotient, unit)
    if num == 0:
        return "0 bytes"
    if num == 1:
        return "1 byte"


def _file_like(obj):
    # An alternative would be::
    #
    #   isinstance(obj, (TextIOBase, BufferedIOBase, RawIOBase, IOBase))
    #
    # but this might be more robust to file-like objects not properly
    # inheriting from these classes:
    return all(callable(getattr(obj, name, None)) for name in ("read", "seek"))


def _fullname(obj):
    klass = obj.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


def _assert_no_instances(cls, when=""):
    __tracebackhide__ = True
    n = 0
    ref = list()
    gc.collect()
    objs = gc.get_objects()
    for obj in objs:
        try:
            check = isinstance(obj, cls)
        except Exception:  # such as a weakref
            check = False
        if check:
            if cls.__name__ == "Brain":
                ref.append(f'Brain._cleaned = {getattr(obj, "_cleaned", None)}')
            rr = gc.get_referrers(obj)
            count = 0
            for r in rr:
                if (
                    r is not objs
                    and r is not globals()
                    and r is not locals()
                    and not inspect.isframe(r)
                ):
                    if isinstance(r, list | dict | tuple):
                        rep = f"len={len(r)}"
                        r_ = gc.get_referrers(r)
                        types = (_fullname(x) for x in r_)
                        types = "/".join(sorted(set(x for x in types if x is not None)))
                        rep += f", {len(r_)} referrers: {types}"
                        del r_
                    else:
                        rep = repr(r)[:100].replace("\n", " ")
                        # If it's a __closure__, get more information
                        if rep.startswith("<cell at "):
                            try:
                                rep += f" ({repr(r.cell_contents)[:100]})"
                            except Exception:
                                pass
                    name = _fullname(r)
                    ref.append(f"{name}: {rep}")
                    count += 1
                del r
            del rr
            n += count > 0
        del obj
    del objs
    gc.collect()
    assert n == 0, f"\n{n} {cls.__name__} @ {when}:\n" + "\n".join(ref)


def _resource_path(submodule, filename):
    """Return a full system path to a package resource (AKA a file).

    Parameters
    ----------
    submodule : str
        An import-style module or submodule name
        (e.g., "mne.datasets.testing").
    filename : str
        The file whose full path you want.

    Returns
    -------
    path : str
        The full system path to the requested file.
    """
    return files(submodule).joinpath(filename)


def repr_html(f):
    """Decorate _repr_html_ methods.

    If a _repr_html_ method is decorated with this decorator, the repr in a
    notebook will show HTML or plain text depending on the config value
    MNE_REPR_HTML (by default "true", which will render HTML).

    Parameters
    ----------
    f : function
        The function to decorate.

    Returns
    -------
    wrapper : function
        The decorated function.
    """
    from ..utils import get_config

    def wrapper(*args, **kwargs):
        if get_config("MNE_REPR_HTML", "true").lower() == "false":
            import html

            r = "<pre>" + html.escape(repr(args[0])) + "</pre>"
            return r.replace("\n", "<br/>")
        else:
            return f(*args, **kwargs)

    return wrapper


def _auto_weakref(function):
    """Create weakrefs to self (or other free vars in __closure__) then evaluate.

    When a nested function is defined within an instance method, and the function makes
    use of ``self``, it creates a reference cycle that the Python garbage collector is
    not smart enough to resolve, so the parent object is never GC'd. (The reference to
    ``self`` becomes part of the ``__closure__`` of the nested function).

    This decorator allows the nested function to access ``self`` without increasing the
    reference counter on ``self``, which will prevent the memory leak. If the referent
    is not found (usually because already GC'd) it will short-circuit the decorated
    function and return ``None``.
    """
    names = function.__code__.co_freevars
    assert len(names) == len(function.__closure__)
    __weakref_values__ = dict()
    evaldict = dict(__weakref_values__=__weakref_values__)
    for name, value in zip(names, function.__closure__):
        __weakref_values__[name] = weakref.ref(value.cell_contents)
    body = dedent(inspect.getsource(function))
    body = body.splitlines()
    for li, line in enumerate(body):
        if line.startswith(" "):
            body = body[li:]
            break
    old_body = "\n".join(body)
    body = """\
def %(name)s(%(signature)s):
"""
    for name in names:
        body += f"""
    {name} = __weakref_values__[{repr(name)}]()
    if {name} is None:
        return
"""
    body = body + old_body
    fm = FunctionMaker(function)
    fun = fm.make(body, evaldict, addsource=True)
    fun.__globals__.update(function.__globals__)
    assert fun.__closure__ is None, fun.__closure__
    return fun
