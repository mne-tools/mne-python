from numpy.testing import assert_equal, assert_array_equal
from nose.tools import assert_true, assert_raises, assert_not_equal
from copy import deepcopy
import os.path as op
import numpy as np
import os
import warnings
from mne.externals.six.moves import urllib

from mne.utils import (set_log_level, set_log_file, _TempDir,
                       get_config, set_config, deprecated, _fetch_file,
                       sum_squared, requires_mem_gb, estimate_rank,
                       _url_to_local_path, sizeof_fmt,
                       _check_type_picks, object_hash, object_diff,
                       requires_good_network)
from mne.io import show_fiff
from mne import Evoked


warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')
fname_raw = op.join(base_dir, 'test_raw.fif')
fname_log = op.join(base_dir, 'test-ave.log')
fname_log_2 = op.join(base_dir, 'test-ave-2.log')
tempdir = _TempDir()
test_name = op.join(tempdir, 'test.log')


def clean_lines(lines):
    # Function to scrub filenames for checking logging output (in test_logging)
    return [l if 'Reading ' not in l else 'Reading test file' for l in lines]


def test_hash():
    """Test dictionary hashing and comparison functions"""
    # does hashing all of these types work:
    # {dict, list, tuple, ndarray, str, float, int, None}
    d0 = dict(a=dict(a=0.1, b='fo', c=1), b=[1, 'b'], c=(), d=np.ones(3))
    d0[1] = None
    d0[2.] = b'123'

    d1 = deepcopy(d0)
    print(object_diff(d0, d1))
    assert_equal(object_hash(d0), object_hash(d1))

    # change values slightly
    d1['data'] = np.ones(3, int)
    assert_not_equal(object_hash(d0), object_hash(d1))

    d1 = deepcopy(d0)
    print(object_diff(d0, d1))
    assert_equal(object_hash(d0), object_hash(d1))
    d1['a']['a'] = 0.11
    object_diff(d0, d1)
    assert_not_equal(object_hash(d0), object_hash(d1))

    d1 = deepcopy(d0)
    print(object_diff(d0, d1))
    assert_equal(object_hash(d0), object_hash(d1))
    d1[1] = 2
    object_diff(d0, d1)
    assert_not_equal(object_hash(d0), object_hash(d1))
    # generators (and other types) not supported
    d1[1] = (x for x in d0)
    assert_raises(RuntimeError, object_hash, d1)


def test_tempdir():
    """Test TempDir
    """
    tempdir2 = _TempDir()
    assert_true(op.isdir(tempdir2))
    tempdir2.cleanup()
    assert_true(not op.isdir(tempdir2))


def test_estimate_rank():
    """Test rank estimation
    """
    data = np.eye(10)
    data[0, 0] = 0
    assert_equal(estimate_rank(data), 9)


def test_logging():
    """Test logging (to file)
    """
    with open(fname_log, 'r') as old_log_file:
        old_lines = clean_lines(old_log_file.readlines())
    with open(fname_log_2, 'r') as old_log_file_2:
        old_lines_2 = clean_lines(old_log_file_2.readlines())

    if op.isfile(test_name):
        os.remove(test_name)
    # test it one way (printing default off)
    set_log_file(test_name)
    set_log_level('WARNING')
    # should NOT print
    evoked = Evoked(fname_evoked, condition=1)
    with open(test_name) as fid:
        assert_true(fid.readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, condition=1, verbose=False)
    with open(test_name) as fid:
        assert_true(fid.readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, condition=1, verbose='WARNING')
    with open(test_name) as fid:
        assert_true(fid.readlines() == [])
    # SHOULD print
    evoked = Evoked(fname_evoked, condition=1, verbose=True)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    set_log_file(None)  # Need to do this to close the old file
    os.remove(test_name)

    # now go the other way (printing default on)
    set_log_file(test_name)
    set_log_level('INFO')
    # should NOT print
    evoked = Evoked(fname_evoked, condition=1, verbose='WARNING')
    with open(test_name) as fid:
        assert_true(fid.readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, condition=1, verbose=False)
    with open(test_name) as fid:
        assert_true(fid.readlines() == [])
    # SHOULD print
    evoked = Evoked(fname_evoked, condition=1)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    with open(fname_log, 'r') as old_log_file:
        assert_equal(new_lines, old_lines)
    # check to make sure appending works (and as default, raises a warning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        set_log_file(test_name, overwrite=False)
        assert len(w) == 0
        set_log_file(test_name)
        assert len(w) == 1
    evoked = Evoked(fname_evoked, condition=1)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines_2)

    # make sure overwriting works
    set_log_file(test_name, overwrite=True)
    # this line needs to be called to actually do some logging
    evoked = Evoked(fname_evoked, condition=1)
    del evoked
    with open(test_name, 'r') as new_log_file:
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        set_config(key, None, home_dir=tempdir)
    assert_true(len(w) == 1)
    assert_true(get_config(key, home_dir=tempdir) is None)
    assert_raises(KeyError, get_config, key, raise_error=True)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        set_config(key, value, home_dir=tempdir)
        assert_true(get_config(key, home_dir=tempdir) == value)
        set_config(key, None, home_dir=tempdir)
    if old_val is not None:
        os.environ[key] = old_val
    # Check if get_config with no input returns all config
    key = 'MNE_PYTHON_TESTING_KEY'
    config = {key: value}
    with warnings.catch_warnings(record=True):  # non-standard key
        warnings.simplefilter('always')
        set_config(key, value, home_dir=tempdir)
    assert_equal(get_config(home_dir=tempdir), config)


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


@deprecated('message')
class deprecated_class(object):
    def __init__(self):
        pass


@requires_mem_gb(10000)
def big_mem_func():
    pass


@requires_mem_gb(0)
def no_mem_func():
    pass


def test_deprecated():
    """Test deprecated function
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        deprecated_func()
    assert_true(len(w) == 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        deprecated_class()
    assert_true(len(w) == 1)


def test_requires_mem_gb():
    """Test requires memory function
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            big_mem_func()
        assert_true(len(w) == 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            no_mem_func()
        assert_true(len(w) == 0)
    except:
        try:
            import psutil
            msg = ('psutil version %s exposes unexpected API' %
                   psutil.__version__)
        except ImportError:
            msg = 'Could not import psutil'
        from nose.plugins.skip import SkipTest
        SkipTest(msg)


@requires_good_network
def test_fetch_file():
    """Test file downloading
    """
    # Skipping test if no internet connection available
    try:
        urllib.request.urlopen("http://github.com", timeout=2)
    except:
        from nose.plugins.skip import SkipTest
        raise SkipTest('No internet connection, skipping download test.')

    urls = ['http://github.com/mne-tools/mne-python/blob/master/README.rst',
            'ftp://surfer.nmr.mgh.harvard.edu/pub/data/bert.recon.md5sum.txt']
    for url in urls:
        archive_name = op.join(tempdir, "download_test")
        _fetch_file(url, archive_name, print_destination=False)
        assert_raises(Exception, _fetch_file, 'NOT_AN_ADDRESS',
                      op.join(tempdir, 'test'))
        resume_name = op.join(tempdir, "download_resume")
        # touch file
        with open(resume_name + '.part', 'w'):
            os.utime(resume_name + '.part', None)
        _fetch_file(url, resume_name, print_destination=False, resume=True)


def test_sum_squared():
    """Test optimized sum of squares
    """
    X = np.random.randint(0, 50, (3, 3))
    assert_equal(np.sum(X ** 2), sum_squared(X))


def test_sizeof_fmt():
    """Test sizeof_fmt
    """
    assert_equal(sizeof_fmt(0), '0 bytes')
    assert_equal(sizeof_fmt(1), '1 byte')
    assert_equal(sizeof_fmt(1000), '1000 bytes')


def test_url_to_local_path():
    """Test URL to local path
    """
    assert_equal(_url_to_local_path('http://google.com/home/why.html', '.'),
                 op.join('.', 'home', 'why.html'))


def test_check_type_picks():
    """Test checking type integrity checks of picks
    """
    picks = np.arange(12)
    assert_array_equal(picks, _check_type_picks(picks))
    picks = list(range(12))
    assert_array_equal(np.array(picks), _check_type_picks(picks))
    picks = None
    assert_array_equal(None, _check_type_picks(picks))
    picks = ['a', 'b']
    assert_raises(ValueError, _check_type_picks, picks)
    picks = 'b'
    assert_raises(ValueError, _check_type_picks, picks)
