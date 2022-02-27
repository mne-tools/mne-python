import os
import platform
import pytest
from pathlib import Path

from mne.utils import (set_config, get_config, get_config_path,
                       set_memmap_min_size, _get_stim_channel, sys_info,
                       ClosingStringIO, get_subjects_dir)


def test_config(tmp_path):
    """Test mne-python config file support."""
    tempdir = str(tmp_path)
    key = '_MNE_PYTHON_CONFIG_TESTING'
    value = '123456'
    value2 = '123'
    value3 = Path('/foo/bar')

    old_val = os.getenv(key, None)
    os.environ[key] = value
    assert (get_config(key) == value)
    del os.environ[key]
    # catch the warning about it being a non-standard config key
    assert (len(get_config('')) > 10)  # tuple of valid keys
    with pytest.warns(RuntimeWarning, match='non-standard'):
        set_config(key, None, home_dir=tempdir, set_env=False)
    assert (get_config(key, home_dir=tempdir) is None)
    pytest.raises(KeyError, get_config, key, raise_error=True)
    assert (key not in os.environ)
    with pytest.warns(RuntimeWarning, match='non-standard'):
        set_config(key, value, home_dir=tempdir, set_env=True)
    assert (key in os.environ)
    assert (get_config(key, home_dir=tempdir) == value)
    with pytest.warns(RuntimeWarning, match='non-standard'):
        set_config(key, None, home_dir=tempdir, set_env=True)
    assert (key not in os.environ)
    with pytest.warns(RuntimeWarning, match='non-standard'):
        set_config(key, None, home_dir=tempdir, set_env=True)
    assert (key not in os.environ)
    if old_val is not None:
        os.environ[key] = old_val

    # Check serialization from Path to string
    with pytest.warns(RuntimeWarning, match='non-standard'):
        set_config(key, value3, home_dir=tempdir)

    # Check if get_config with key=None returns all config
    key = 'MNE_PYTHON_TESTING_KEY'
    assert key not in get_config(home_dir=tempdir)
    with pytest.warns(RuntimeWarning, match='non-standard'):
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
    with open(json_fname, 'w') as fid:
        fid.write('foo{}')
    with pytest.warns(RuntimeWarning, match='not a valid JSON'):
        assert key not in get_config(home_dir=tempdir)
    with pytest.warns(RuntimeWarning, match='non-standard'):
        pytest.raises(RuntimeError, set_config, key, 'true', home_dir=tempdir)

    # degenerate conditions
    pytest.raises(ValueError, set_memmap_min_size, 1)
    pytest.raises(ValueError, set_memmap_min_size, 'foo')
    pytest.raises(TypeError, get_config, 1)
    pytest.raises(TypeError, set_config, 1)
    pytest.raises(TypeError, set_config, 'foo', 1)
    pytest.raises(TypeError, _get_stim_channel, 1, None)
    pytest.raises(TypeError, _get_stim_channel, [1], None)


def test_sys_info():
    """Test info-showing utility."""
    out = ClosingStringIO()
    sys_info(fid=out)
    out = out.getvalue()
    assert ('numpy:' in out)

    if platform.system() == 'Darwin':
        assert 'Platform:         macOS-' in out
    elif platform.system() == 'Linux':
        assert 'Platform:         Linux' in out


def test_get_subjects_dir(tmp_path, monkeypatch):
    """Test get_subjects_dir()."""
    subjects_dir = tmp_path / 'foo'
    subjects_dir.mkdir()

    # String
    assert get_subjects_dir(str(subjects_dir)) == str(subjects_dir)

    # Path
    assert get_subjects_dir(subjects_dir) == str(subjects_dir)

    # `None`
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', str(tmp_path))
    monkeypatch.delenv('SUBJECTS_DIR', raising=False)
    assert get_subjects_dir() is None

    # Expand `~`
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))  # Windows
    assert get_subjects_dir('~/foo') == str(subjects_dir)
