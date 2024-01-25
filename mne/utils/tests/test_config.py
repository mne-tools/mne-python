# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import os
import platform
import re
from functools import partial
from pathlib import Path
from urllib.error import URLError

import pytest

import mne
import mne.utils.config
from mne.utils import (
    ClosingStringIO,
    _get_stim_channel,
    get_config,
    get_config_path,
    get_subjects_dir,
    requires_good_network,
    set_config,
    set_memmap_min_size,
    sys_info,
)


def test_config(tmp_path):
    """Test mne-python config file support."""
    tempdir = str(tmp_path)
    key = "_MNE_PYTHON_CONFIG_TESTING"
    value = "123456"
    value2 = "123"
    value3 = Path("/foo/bar")

    old_val = os.getenv(key, None)
    os.environ[key] = value
    assert get_config(key) == value
    del os.environ[key]
    # catch the warning about it being a non-standard config key
    known_config_keys = get_config("")
    assert len(known_config_keys) > 10  # dict of valid keys
    for k, val in known_config_keys.items():
        assert isinstance(k, str)
        assert isinstance(val, str), k
        assert len(val) > 0, k
    with pytest.warns(RuntimeWarning, match="non-standard"):
        set_config(key, None, home_dir=tempdir, set_env=False)
    assert get_config(key, home_dir=tempdir) is None
    pytest.raises(KeyError, get_config, key, raise_error=True)
    assert key not in os.environ
    with pytest.warns(RuntimeWarning, match="non-standard"):
        set_config(key, value, home_dir=tempdir, set_env=True)
    assert key in os.environ
    assert get_config(key, home_dir=tempdir) == value
    with pytest.warns(RuntimeWarning, match="non-standard"):
        set_config(key, None, home_dir=tempdir, set_env=True)
    assert key not in os.environ
    with pytest.warns(RuntimeWarning, match="non-standard"):
        set_config(key, None, home_dir=tempdir, set_env=True)
    assert key not in os.environ
    if old_val is not None:
        os.environ[key] = old_val

    # Check serialization from Path to string
    with pytest.warns(RuntimeWarning, match="non-standard"):
        set_config(key, value3, home_dir=tempdir)

    # Check if get_config with key=None returns all config
    key = "MNE_PYTHON_TESTING_KEY"
    assert key not in get_config(home_dir=tempdir)
    with pytest.warns(RuntimeWarning, match="non-standard"):
        set_config(key, value, home_dir=tempdir)
    assert get_config(home_dir=tempdir)[key] == value
    old_val = os.environ.get(key)
    try:  # os.environ should take precedence over config file
        os.environ[key] = value2
        assert get_config(home_dir=tempdir)[key] == value2
    finally:  # reset os.environ
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val
    # Check what happens when we use a corrupted file
    json_fname = get_config_path(home_dir=tempdir)
    with open(json_fname, "w") as fid:
        fid.write("foo{}")
    with pytest.warns(RuntimeWarning, match="not a valid JSON"):
        assert key not in get_config(home_dir=tempdir)
    with pytest.warns(RuntimeWarning, match="non-standard"):
        pytest.raises(RuntimeError, set_config, key, "true", home_dir=tempdir)

    # degenerate conditions
    with pytest.raises(TypeError, match="must be an instance"):
        set_memmap_min_size(1)
    pytest.raises(ValueError, set_memmap_min_size, "foo")
    pytest.raises(TypeError, get_config, 1)
    pytest.raises(TypeError, set_config, 1)
    pytest.raises(TypeError, set_config, "foo", 1)
    pytest.raises(TypeError, _get_stim_channel, 1, None)
    pytest.raises(TypeError, _get_stim_channel, [1], None)


def test_sys_info():
    """Test info-showing utility."""
    out = ClosingStringIO()
    sys_info(fid=out, check_version=False)
    out = out.getvalue()
    assert "numpy" in out

    if platform.system() == "Darwin":
        assert "Platform             macOS-" in out
    elif platform.system() == "Linux":
        assert "Platform             Linux" in out


def test_sys_info_qt_browser():
    """Test if mne_qt_browser is correctly detected."""
    pytest.importorskip("mne_qt_browser")
    out = ClosingStringIO()
    sys_info(fid=out, check_version=False)
    out = out.getvalue()
    assert "mne-qt-browser" in out


def test_get_subjects_dir(tmp_path, monkeypatch):
    """Test get_subjects_dir()."""
    subjects_dir = tmp_path / "foo"
    subjects_dir.mkdir()

    # String
    assert get_subjects_dir(str(subjects_dir)) == subjects_dir

    # Path
    assert get_subjects_dir(subjects_dir) == subjects_dir

    # `None`
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(tmp_path))
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    assert get_subjects_dir() is None

    # Expand `~`
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    assert str(get_subjects_dir("~/foo")) == str(subjects_dir)


@pytest.mark.slowtest
@requires_good_network
def test_sys_info_check_outdated(monkeypatch):
    """Test sys info checking."""
    # Old (actually ping GitHub)
    monkeypatch.setattr(mne, "__version__", "0.1")
    out = ClosingStringIO()
    sys_info(fid=out, check_version=10)
    out = out.getvalue()
    assert "(outdated, release " in out
    assert "updating.html" in out

    # Timeout (will call urllib.open)
    out = ClosingStringIO()
    sys_info(fid=out, check_version=1e-12)
    out = out.getvalue()
    assert re.match(".*unable to check.*timeout.*", out, re.DOTALL) is not None
    assert "updating.html" not in out


def test_sys_info_check_other(monkeypatch):
    """Test other failure modes of the sys info check."""

    def bad_open(url, timeout, msg):
        raise URLError(msg)

    # SSL error
    out = ClosingStringIO()
    with monkeypatch.context() as m:
        m.setattr(mne.utils.config, "urlopen", partial(bad_open, msg="SSL: CERT"))
        sys_info(fid=out)
    out = out.getvalue()
    assert re.match(".*unable to check.*SSL.*", out, re.DOTALL) is not None

    # Other error
    out = ClosingStringIO()
    with monkeypatch.context() as m:
        m.setattr(mne.utils.config, "urlopen", partial(bad_open, msg="foo bar"))
        sys_info(fid=out)
    out = out.getvalue()
    match = re.match(".*unable to .*unknown error: .*foo bar.*", out, re.DOTALL)
    assert match is not None

    # Match
    monkeypatch.setattr(
        mne.utils.config,
        "_get_latest_version",
        lambda timeout: "1.5.1",
    )
    monkeypatch.setattr(mne, "__version__", "1.5.1")
    out = ClosingStringIO()
    sys_info(fid=out)
    out = out.getvalue()
    assert " 1.5.1 (latest release)" in out

    # Devel
    monkeypatch.setattr(mne, "__version__", "1.6.dev0")
    out = ClosingStringIO()
    sys_info(fid=out)
    out = out.getvalue()
    assert "devel, " in out
    assert "updating.html" not in out
