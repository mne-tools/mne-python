from contextlib import nullcontext
import os
import subprocess
import sys

import pytest

import mne
from mne.utils import sizeof_fmt, run_subprocess, catch_logging


def test_sizeof_fmt():
    """Test sizeof_fmt."""
    assert sizeof_fmt(0) == '0 bytes'
    assert sizeof_fmt(1) == '1 byte'
    assert sizeof_fmt(1000) == '1000 bytes'


def test_html_repr():
    """Test switching _repr_html_ between HTML and plain text."""
    key = "MNE_REPR_HTML"
    existing_value = os.getenv(key, None)

    os.environ[key] = "True"  # HTML repr on
    info = mne.create_info(10, 256)
    r = info._repr_html_()
    assert r.startswith("<table")
    assert r.endswith("</table>")
    os.environ[key] = "False"  # HTML repr off
    r = info._repr_html_()
    assert r.startswith("<pre>")
    assert r.endswith("</pre>")

    del os.environ[key]
    if existing_value is not None:
        os.environ[key, existing_value]


@pytest.mark.parametrize('kind', ('stdout', 'stderr'))
@pytest.mark.parametrize('do_raise', (True, False))
def test_run_subprocess(tmp_path, capsys, kind, do_raise):
    """Test run_subprocess."""
    fname = tmp_path / 'subp.py'
    extra = ''
    if do_raise:
        extra = """
raise RuntimeError('This is a test')
"""
        raise_context = pytest.raises(subprocess.CalledProcessError)
    else:
        extra = ''
        raise_context = nullcontext()
    with open(fname, 'w') as fid:
        fid.write(f"""\
import sys
import time
print('foo', file=sys.{kind})
print('bar', file=sys.{kind})
""" + extra)
    with catch_logging() as log, raise_context:
        stdout, stderr = run_subprocess(
            [sys.executable, str(fname)], verbose=True)
    if do_raise:
        exc = raise_context.excinfo.value
        stdout = exc.stdout
        stderr = exc.stderr
    log = log.getvalue()
    log = '\n'.join(log.split('\n')[1:])  # get rid of header
    log = log.replace('\r\n', '\n')  # Windows
    orig_log = log
    stdout = stdout.replace('\r\n', '\n')
    stderr = stderr.replace('\r\n', '\n')
    if do_raise:  # remove traceback

        def remove_traceback(log):
            return '\n'.join(
                line for line in log.split('\n')
                if not line.strip().startswith(
                    ('File ', 'raise ', 'RuntimeError: ', 'Traceback ')))

        log = remove_traceback(log)
        stderr = remove_traceback(stderr)
    want = 'foo\nbar\n'
    assert log == want, orig_log
    if kind == 'stdout':
        std = stdout
        other = stderr
    else:
        std = stderr
        other = stdout
    assert std == want
    assert other == ''
    stdout, stderr = capsys.readouterr()
    assert stdout == ''
    assert stderr == ''

    # Now make sure we can pass other stuff as stdout/stderr
    capsys.readouterr()  # clear
    stdout_fname = tmp_path / 'stdout.txt'
    stderr_fname = tmp_path / 'stderr.txt'
    stdout_file = open(stdout_fname, 'w')
    stderr_file = open(stderr_fname, 'w')
    if do_raise:
        raise_context = pytest.raises(subprocess.CalledProcessError)
    else:
        raise_context = nullcontext()
    with catch_logging() as log, stdout_file, stderr_file, raise_context:
        stdout, stderr = run_subprocess(
            [sys.executable, str(fname)], verbose=False,
            stdout=stdout_file, stderr=stderr_file)
    if do_raise:
        exc = raise_context.excinfo.value
        assert exc.stdout is None
        assert exc.stderr is None
    else:
        assert stdout == ''
        assert stderr == ''
    log = log.getvalue()
    assert log == ''
    stdout, stderr = capsys.readouterr()
    assert stdout == ''
    assert stderr == ''
    stdout = stdout_fname.read_text()
    stderr = stderr_fname.read_text()
    if do_raise:
        stderr = '\n'.join(stderr.split('\n')[:-5])
        if stderr:
            stderr += '\n'
    if kind == 'stdout':
        std = stdout
        other = stderr
    else:
        std = stderr
        other = stdout
    assert std == want
    assert other == ''
