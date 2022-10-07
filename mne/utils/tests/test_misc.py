import os
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
def test_run_subprocess(tmp_path, kind):
    """Test run_subprocess."""
    fname = tmp_path / 'subp.py'
    with open(fname, 'w') as fid:
        fid.write(f"""\
import sys
print('foo', file=sys.{kind})
print('bar', file=sys.{kind})
""")
    with catch_logging() as log:
        stdout, stderr = run_subprocess(
            [sys.executable, str(fname)], verbose=True)
    log = log.getvalue()
    log = '\n'.join(log.split('\n')[1:])  # get rid of header
    log = log.replace('\r\n', '\n')  # Windows
    stdout = stdout.replace('\r\n', '\n')
    stderr = stderr.replace('\r\n', '\n')
    want = 'foo\nbar\n'
    assert log == want
    if kind == 'stdout':
        std = stdout
        other = stderr
    else:
        std = stderr
        other = stdout
    assert std == want
    assert other == ''
