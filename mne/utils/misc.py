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


def _empty_hash(kind="md5"):
    return getattr(hashlib, kind)(usedforsecurity=False)


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


def _fullname(obj, *, referent=None):
    if inspect.ismodule(obj):
        # Otherwise every module shows up identically as just "module",
        # which is exactly the info we need to tell which one is which.
        name = getattr(obj, "__name__", "<unknown module>")
    else:
        klass = obj.__class__
        module = klass.__module__
        name = klass.__qualname__
        if module != "builtins":
            name = f"{module}.{name}"
    if referent is not None:
        if isinstance(obj, list | tuple):
            for ii, item in enumerate(obj):
                if item is referent:
                    name += f"[{ii}]"
                    break
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key is referent:
                    name += "-key"
                    break
                if value is referent:
                    name += f"[{key!r}]"
                    break
    return name


def _gc_collect_once(request=None):
    """Call gc.collect(), deduplicated once per test item if given a request.

    ``gc.collect()`` cost scales with the number of tracked objects in the
    whole process, so when several independent test fixtures each want a
    fresh collect during the same test's teardown, doing it more than once
    is a significant and unnecessary fraction of total test time. When
    ``request`` (a pytest fixture request) is given, only the first call
    for a given test item actually collects; later calls are no-ops.
    """
    if request is None:
        gc.collect()
        return
    node = request.node
    if getattr(node, "_mne_gc_collected", False):
        return
    node._mne_gc_collected = True
    gc.collect()


def _safe_repr(obj, *, maxlen=100):
    """Get a repr that cannot raise (e.g., on a deleted VTK/Qt C++ object)."""
    try:
        rep = repr(obj)
    except Exception as exc:
        return f"<repr failed: {type(exc).__name__}: {exc}>"
    return rep[:maxlen].replace("\n", " ")


def _dict_owner(d):
    """Find the object whose __dict__ (or similar) *is* d, if any."""
    for o in gc.get_referrers(d):
        if getattr(o, "__dict__", None) is d:
            return o
    return None


def _module_global_name(obj):
    """Find the "module.attr" name of obj if it is itself a module-level global.

    This is what lets a failure message name e.g. a long-lived module-level
    registry (cache, weak-value dict, etc.) directly, which is often the
    actual reason an object outlives a single test/example: a plain
    ``gc.get_referrers`` walk only shows an anonymous ``dict``/``list``.
    """
    for modname, mod in list(sys.modules.items()):
        d = getattr(mod, "__dict__", None)
        if not d:
            continue
        try:
            items = list(d.items())
        except Exception:
            continue
        for key, val in items:
            if val is obj:
                return f"{modname}.{key}"
    return None


def _describe_referrer(r, referent):
    """Build a short, safe description of r, something that refers to referent.

    Returns
    -------
    desc : str
        Human-readable, safe description of r.
    next_obj : object | None
        What to keep tracing referrers of (``None`` to stop here). This is
        usually ``r`` itself, but for e.g. an instance's ``__dict__`` it's
        the owning instance (tracing the dict's own referrers is normally
        just uninformative interpreter-internal noise), and for a
        module-level global it's ``None`` (a named global is already a
        fully-explained anchor; nothing more useful to say).
    """
    global_name = _module_global_name(r)
    if inspect.ismethod(r):
        desc = f"bound method {r.__func__.__qualname__} of {_safe_repr(r.__self__)}"
        return desc, r
    name = _fullname(r, referent=referent)
    if isinstance(r, dict):
        owner = _dict_owner(r)
        if owner is not None:
            return f"{_fullname(owner)}.__dict__ ({name})", owner
        if global_name is not None:
            return f"{global_name} (module-level dict, len={len(r)})", None
        return f"{name} (len={len(r)})", r
    if isinstance(r, list | tuple):
        if global_name is not None:
            return f"{global_name} (module-level {name}, len={len(r)})", None
        return f"{name} (len={len(r)})", r
    if global_name is not None:
        return f"{global_name} ({_safe_repr(r)})", None
    rep = _safe_repr(r)
    if rep.startswith("<cell at "):  # a closure variable
        try:
            rep += f" ({_safe_repr(r.cell_contents)})"
        except Exception:
            pass
    return f"{name} | {rep}", r


def _append_referrers(o, depth, *, max_depth, max_lines, excluded, recursed, lines):
    """Recursively describe referrers of o into lines, indented by depth.

    ``excluded`` objects (e.g. the huge ``gc.get_objects()`` snapshot) are
    never shown or recursed into. ``recursed`` tracks objects already used
    as a recursion root, so a cycle in the referrer graph can't recurse
    forever; unlike ``excluded`` it doesn't prevent an object from being
    *listed* (only from being expanded again).
    """
    any_shown = False
    refs = gc.get_referrers(o)
    # While this list is alive (i.e. for the duration of this call, including
    # any recursive calls below), it itself shows up as a "referrer" of any
    # of its own elements if we ask gc.get_referrers() about them -- that's
    # an artifact of doing this traversal at all, not a real anchor.
    excluded.add(id(refs))
    for r in refs:
        if len(lines) >= max_lines:
            lines.append("  " * depth + "- ... (truncated)")
            return any_shown
        if inspect.isframe(r) or id(r) in excluded:
            continue
        any_shown = True
        desc, next_obj = _describe_referrer(r, o)
        lines.append("  " * depth + "- " + desc)
        if (
            next_obj is not None
            and id(next_obj) not in recursed
            and id(next_obj) not in excluded
            and depth + 1 < max_depth
        ):
            recursed.add(id(next_obj))
            _append_referrers(
                next_obj,
                depth + 1,
                max_depth=max_depth,
                max_lines=max_lines,
                excluded=excluded,
                recursed=recursed,
                lines=lines,
            )
        del r
    del refs
    return any_shown


def _referrer_chain(obj, *, max_depth=5, max_lines=40, exclude_ids=()):
    """Describe, recursively, what holds references to obj.

    Referrers are walked (and indented) up to ``max_depth`` hops so that a
    leaked object's actual anchor (e.g. a module-level registry several
    containers away) is visible directly in the failure message, instead of
    just the immediate (often uninformative, e.g. a bare ``list``) referrer.
    """
    lines = list()
    excluded = set(exclude_ids)
    recursed = {id(obj)}
    has_referrers = _append_referrers(
        obj,
        0,
        max_depth=max_depth,
        max_lines=max_lines,
        excluded=excluded,
        recursed=recursed,
        lines=lines,
    )
    return lines, has_referrers


def _assert_no_instances(cls, when="", *, request=None, objs=None):
    __tracebackhide__ = True
    n = 0
    ref = list()
    _gc_collect_once(request)
    if objs is None:
        objs = gc.get_objects()
    for obj in objs:  # e.g., vtkPolyData, Brain, Plotter, etc.
        try:
            check = isinstance(obj, cls)
        except Exception:  # such as a weakref
            check = False
        if check:
            extra = list()
            if cls.__name__ == "Brain":
                extra.append(f"Brain._cleaned = {getattr(obj, '_cleaned', None)}")
            lines, has_referrers = _referrer_chain(
                obj, exclude_ids={id(objs), id(ref), id(globals())}
            )
            if has_referrers:
                ref.extend(extra)
                ref.append(f"{_fullname(obj)}:")
                ref.extend(lines)
                n += 1
        del obj
    del objs
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
        try:
            __weakref_values__[name] = weakref.ref(value.cell_contents)
        except TypeError:  # pragma: no cover
            raise TypeError(
                f"Cannot create weak reference to {name} "
                f"(type {type(value.cell_contents)})"
            )
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
