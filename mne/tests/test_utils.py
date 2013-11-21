from numpy.testing import assert_equal
from nose.tools import assert_true, assert_raises
import os.path as op
import os
import warnings
import urllib2

from ..utils import set_log_level, set_log_file, _TempDir, \
                    get_config, set_config, deprecated, _fetch_file
from ..fiff import Evoked, show_fiff

base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')
fname_raw = op.join(base_dir, 'test_raw.fif')
fname_log = op.join(base_dir, 'test-ave.log')
fname_log_2 = op.join(base_dir, 'test-ave-2.log')
tempdir = _TempDir()
test_name = op.join(tempdir, 'test.log')


def clean_lines(lines):
    # Function to scrub filenames for checking logging output (in test_logging)
    return [l if 'Reading ' not in l else 'Reading test file' for l in lines]


def test_logging():
    """Test logging (to file)
    """
    old_log_file = open(fname_log, 'r')
    old_lines = clean_lines(old_log_file.readlines())
    old_log_file.close()
    old_log_file_2 = open(fname_log_2, 'r')
    old_lines_2 = clean_lines(old_log_file_2.readlines())
    old_log_file_2.close()

    if op.isfile(test_name):
        os.remove(test_name)
    # test it one way (printing default off)
    set_log_file(test_name)
    set_log_level('WARNING')
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1)
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose=False)
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose='WARNING')
    assert_true(open(test_name).readlines() == [])
    # SHOULD print
    evoked = Evoked(fname_evoked, setno=1, verbose=True)
    new_log_file = open(test_name, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    new_log_file.close()
    set_log_file(None)  # Need to do this to close the old file
    os.remove(test_name)

    # now go the other way (printing default on)
    set_log_file(test_name)
    set_log_level('INFO')
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose='WARNING')
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose=False)
    assert_true(open(test_name).readlines() == [])
    # SHOULD print
    evoked = Evoked(fname_evoked, setno=1)
    new_log_file = open(test_name, 'r')
    old_log_file = open(fname_log, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    # check to make sure appending works (and as default, raises a warning)
    with warnings.catch_warnings(True) as w:
        set_log_file(test_name, overwrite=False)
        assert len(w) == 0
        set_log_file(test_name)
        assert len(w) == 1
    evoked = Evoked(fname_evoked, setno=1)
    new_log_file = open(test_name, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines_2)

    # make sure overwriting works
    set_log_file(test_name, overwrite=True)
    # this line needs to be called to actually do some logging
    evoked = Evoked(fname_evoked, setno=1)
    del evoked
    new_log_file = open(test_name, 'r')
    new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)


def test_config():
    """Test mne-python config file support"""
    key = '_MNE_PYTHON_CONFIG_TESTING'
    value = '123456'
    old_val = os.getenv(key, None)
    os.environ[key] = value
    assert_true(get_config(key) == value)
    del os.environ[key]
    # catch the warning about it being a non-standard config key
    with warnings.catch_warnings(True) as w:
        set_config(key, None)
        assert_true(len(w) == 1)
    assert_true(get_config(key) is None)
    assert_raises(KeyError, get_config, key, raise_error=True)
    set_config(key, value)
    assert_true(get_config(key) == value)
    set_config(key, None)
    if old_val is not None:
        os.environ[key] = old_val


def test_show_fiff():
    """Test show_fiff
    """
    # this is not exhaustive, but hopefully bugs will be found in use
    info = show_fiff(fname_evoked)
    keys = ['FIFF_EPOCH', 'FIFFB_HPI_COIL', 'FIFFB_PROJ_ITEM',
            'FIFFB_PROCESSED_DATA', 'FIFFB_EVOKED', 'FIFF_NAVE',
            'FIFF_EPOCH']
    assert_true(all([key in info for key in keys]))
    info = show_fiff(fname_raw, read_limit=1024)


@deprecated('message')
def deprecated_func():
    pass


def test_deprecated():
    """Test deprecated function
    """
    with warnings.catch_warnings(True) as w:
        deprecated_func()
    assert_true(len(w) == 1)


def test_fetch_file():
    """Test file downloading
    """
    # Skipping test if no internet connection available
    try:
        urllib2.urlopen("http://github.com", timeout=1)
    except urllib2.URLError:
        from nose.plugins.skip import SkipTest
        raise SkipTest('No internet connection, skipping download test.')
    url = "http://github.com/mne-tools/mne-python/blob/master/README.rst"
    archive_name = op.join(tempdir, "download_test")
    _fetch_file(url, archive_name, print_destination=False)
