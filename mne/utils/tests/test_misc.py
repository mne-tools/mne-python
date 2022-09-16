import os

import mne
from mne.utils import sizeof_fmt


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
