# -*- coding: utf-8 -*-
"""Some miscellaneous utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from contextlib import contextmanager
import fnmatch
import inspect
from math import log
import os
from queue import Queue, Empty
from string import Formatter
import subprocess
import sys
from threading import Thread
import traceback

import numpy as np

from ..utils import _check_option, _validate_type
from ..fixes import _get_args
from ._logging import logger, verbose, warn


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
    for line in iter(out.readline, b''):
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
    all_out = ''
    all_err = ''
    # non-blocking adapted from https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python#4896288  # noqa: E501
    out_q = Queue()
    err_q = Queue()
    with running_subprocess(command, *args, **kwargs) as p:
        out_t = Thread(target=_enqueue_output, args=(p.stdout, out_q))
        err_t = Thread(target=_enqueue_output, args=(p.stderr, err_q))
        out_t.daemon = True
        err_t.daemon = True
        out_t.start()
        err_t.start()
        while True:
            do_break = p.poll() is not None
            # read all current lines without blocking
            while True:
                try:
                    out = out_q.get(timeout=0.01)
                except Empty:
                    break
                else:
                    out = out.decode('utf-8')
                    logger.info(out)
                    all_out += out
            while True:
                try:
                    err = err_q.get(timeout=0.01)
                except Empty:
                    break
                else:
                    err = err.decode('utf-8')
                    logger.warning(err)
                    all_err += err
            if do_break:
                break
    p.stdout.close()
    p.stderr.close()
    output = (all_out, all_err)

    if return_code:
        output = output + (p.returncode,)
    elif p.returncode:
        print(output)
        err_fun = subprocess.CalledProcessError.__init__
        if 'output' in _get_args(err_fun):
            raise subprocess.CalledProcessError(p.returncode, command, output)
        else:
            raise subprocess.CalledProcessError(p.returncode, command)

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
    _validate_type(after, str, 'after')
    _check_option('after', after, ['wait', 'terminate', 'kill', 'communicate'])
    for stdxxx, sys_stdxxx in (['stderr', sys.stderr], ['stdout', sys.stdout]):
        if stdxxx not in kwargs:
            kwargs[stdxxx] = subprocess.PIPE

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
        command = [str(s) for s in command]
        command_str = ' '.join(s for s in command)
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
    try:
        yield p
    finally:
        getattr(p, after)()
        p.wait()


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
        if not fnmatch.fnmatch(fname, '*/mne/io/*'):
            return None
        args, _, _, values = inspect.getargvalues(frame)
    finally:
        del frame
    params = dict()
    for arg in args:
        params[arg] = values[arg]
    params.pop('self', None)
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
    units = ['bytes', 'kB', 'MB', 'GB', 'TB', 'PB']
    decimals = [0, 0, 1, 2, 2, 2]
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


def _file_like(obj):
    # An alternative would be::
    #
    #   isinstance(obj, (TextIOBase, BufferedIOBase, RawIOBase, IOBase))
    #
    # but this might be more robust to file-like objects not properly
    # inheriting from these classes:
    return all(callable(getattr(obj, name, None)) for name in ('read', 'seek'))
