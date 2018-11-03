from copy import deepcopy
import os.path as op
import os
import sys
import webbrowser

import numpy as np
import warnings
import pytest
from scipy import sparse
from numpy.testing import assert_equal, assert_array_equal, assert_allclose

from mne import read_evokeds, open_docs
from mne.datasets import testing
from mne.externals.six.moves import StringIO
from mne.io import show_fiff, read_raw_fif
from mne.epochs import _segment_raw
from mne.parallel import parallel_func
from mne.time_frequency import tfr_morlet
from mne.utils import (set_log_level, set_log_file, _TempDir,
                       get_config, set_config, deprecated, _fetch_file,
                       sum_squared, estimate_rank, _reg_pinv,
                       _url_to_local_path, sizeof_fmt, _check_subject,
                       _check_type_picks, object_hash, object_diff,
                       requires_good_network, run_tests_if_main, md5sum,
                       ArgvSetter, _memory_usage, check_random_state,
                       _check_mayavi_version, requires_mayavi, traits_test,
                       set_memmap_min_size, _get_stim_channel, _check_fname,
                       create_slices, _time_mask, random_permutation,
                       _get_call_line, compute_corr, sys_info, verbose,
                       check_fname, get_config_path, warn,
                       object_size, buggy_mkl_svd, _get_inst_data,
                       copy_doc, copy_function_doc_to_method_doc, ProgressBar,
                       linkcode_resolve, array_split_idx, filter_out_warnings)


base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')
fname_raw = op.join(base_dir, 'test_raw.fif')
fname_log = op.join(base_dir, 'test-ave.log')
fname_log_2 = op.join(base_dir, 'test-ave-2.log')

data_path = testing.data_path(download=False)
fname_fsaverage_trans = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                                'fsaverage-trans.fif')


def clean_lines(lines=[]):
    """Scrub filenames for checking logging output (in test_logging)."""
    return [l if 'Reading ' not in l else 'Reading test file' for l in lines]


def test_buggy_mkl():
    """Test decorator for buggy MKL issues."""
    from unittest import SkipTest

    @buggy_mkl_svd
    def foo(a, b):
        raise np.linalg.LinAlgError('SVD did not converge')
    with pytest.warns(RuntimeWarning, match='convergence error'):
        pytest.raises(SkipTest, foo, 1, 2)

    @buggy_mkl_svd
    def bar(c, d, e):
        raise RuntimeError('SVD did not converge')
    pytest.raises(RuntimeError, bar, 1, 2, 3)


def test_sys_info():
    """Test info-showing utility."""
    out = StringIO()
    sys_info(fid=out)
    out = out.getvalue()
    assert ('numpy:' in out)


def test_get_call_line():
    """Test getting a call line."""
    @verbose
    def foo(verbose=None):
        return _get_call_line(in_verbose=True)

    for v in (None, True):
        my_line = foo(verbose=v)  # testing
        assert_equal(my_line, 'my_line = foo(verbose=v)  # testing')

    def bar():
        return _get_call_line(in_verbose=False)

    my_line = bar()  # testing more
    assert_equal(my_line, 'my_line = bar()  # testing more')


def test_object_size():
    """Test object size estimation."""
    assert (object_size(np.ones(10, np.float32)) <
            object_size(np.ones(10, np.float64)))
    for lower, upper, obj in ((0, 60, ''),
                              (0, 30, 1),
                              (0, 30, 1.),
                              (0, 70, 'foo'),
                              (0, 150, np.ones(0)),
                              (0, 150, np.int32(1)),
                              (150, 500, np.ones(20)),
                              (100, 400, dict()),
                              (400, 1000, dict(a=np.ones(50))),
                              (200, 900, sparse.eye(20, format='csc')),
                              (200, 900, sparse.eye(20, format='csr'))):
        size = object_size(obj)
        assert lower < size < upper, \
            '%s < %s < %s:\n%s' % (lower, size, upper, obj)


def test_get_inst_data():
    """Test _get_inst_data."""
    raw = read_raw_fif(fname_raw)
    raw.crop(tmax=1.)
    assert_equal(_get_inst_data(raw), raw._data)
    raw.pick_channels(raw.ch_names[:2])

    epochs = _segment_raw(raw, 0.5)
    assert_equal(_get_inst_data(epochs), epochs._data)

    evoked = epochs.average()
    assert_equal(_get_inst_data(evoked), evoked.data)

    evoked.crop(tmax=0.1)
    picks = list(range(2))
    freqs = np.array([50., 55.])
    n_cycles = 3
    tfr = tfr_morlet(evoked, freqs, n_cycles, return_itc=False, picks=picks)
    assert_equal(_get_inst_data(tfr), tfr.data)

    pytest.raises(TypeError, _get_inst_data, 'foo')


def test_misc():
    """Test misc utilities."""
    assert_equal(_memory_usage(-1)[0], -1)
    assert_equal(_memory_usage((clean_lines, [], {}))[0], -1)
    assert_equal(_memory_usage(clean_lines)[0], -1)
    pytest.raises(ValueError, check_random_state, 'foo')
    pytest.raises(ValueError, set_memmap_min_size, 1)
    pytest.raises(ValueError, set_memmap_min_size, 'foo')
    pytest.raises(TypeError, get_config, 1)
    pytest.raises(TypeError, set_config, 1)
    pytest.raises(TypeError, set_config, 'foo', 1)
    pytest.raises(TypeError, _get_stim_channel, 1, None)
    pytest.raises(TypeError, _get_stim_channel, [1], None)
    pytest.raises(TypeError, _check_fname, 1)
    pytest.raises(IOError, check_fname, 'foo', 'tets-dip.x', (), ('.fif',))
    pytest.raises(ValueError, _check_subject, None, None)
    pytest.raises(TypeError, _check_subject, None, 1)
    pytest.raises(TypeError, _check_subject, 1, None)


@requires_mayavi
@traits_test
def test_check_mayavi():
    """Test mayavi version check."""
    pytest.raises(RuntimeError, _check_mayavi_version, '100.0.0')


def test_run_tests_if_main():
    """Test run_tests_if_main functionality."""
    x = []

    def test_a():
        x.append(True)

    @pytest.mark.skipif(True)
    def test_b():
        return

    try:
        __name__ = '__main__'
        run_tests_if_main(measure_mem=False)  # dual meas causes problems

        def test_c():
            raise RuntimeError

        try:
            __name__ = '__main__'
            run_tests_if_main(measure_mem=False)  # dual meas causes problems
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Error not raised')
    finally:
        del __name__
    assert (len(x) == 2)
    assert (x[0] and x[1])


def test_hash():
    """Test dictionary hashing and comparison functions."""
    # does hashing all of these types work:
    # {dict, list, tuple, ndarray, str, float, int, None}
    d0 = dict(a=dict(a=0.1, b='fo', c=1), b=[1, 'b'], c=(), d=np.ones(3),
              e=None)
    d0[1] = None
    d0[2.] = b'123'

    d1 = deepcopy(d0)
    assert (len(object_diff(d0, d1)) == 0)
    assert (len(object_diff(d1, d0)) == 0)
    assert_equal(object_hash(d0), object_hash(d1))

    # change values slightly
    d1['data'] = np.ones(3, int)
    d1['d'][0] = 0
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert_equal(object_hash(d0), object_hash(d1))
    d1['a']['a'] = 0.11
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert_equal(object_hash(d0), object_hash(d1))
    d1['a']['d'] = 0  # non-existent key
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert_equal(object_hash(d0), object_hash(d1))
    d1['b'].append(0)  # different-length lists
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert_equal(object_hash(d0), object_hash(d1))
    d1['e'] = 'foo'  # non-None
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    d2 = deepcopy(d0)
    d1['e'] = StringIO()
    d2['e'] = StringIO()
    d2['e'].write('foo')
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)

    d1 = deepcopy(d0)
    d1[1] = 2
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    # generators (and other types) not supported
    d1 = deepcopy(d0)
    d2 = deepcopy(d0)
    d1[1] = (x for x in d0)
    d2[1] = (x for x in d0)
    pytest.raises(RuntimeError, object_diff, d1, d2)
    pytest.raises(RuntimeError, object_hash, d1)

    x = sparse.eye(2, 2, format='csc')
    y = sparse.eye(2, 2, format='csr')
    assert ('type mismatch' in object_diff(x, y))
    y = sparse.eye(2, 2, format='csc')
    assert_equal(len(object_diff(x, y)), 0)
    y[1, 1] = 2
    assert ('elements' in object_diff(x, y))
    y = sparse.eye(3, 3, format='csc')
    assert ('shape' in object_diff(x, y))
    y = 0
    assert ('type mismatch' in object_diff(x, y))

    # smoke test for gh-4796
    assert object_hash(np.int64(1)) != 0
    assert object_hash(np.bool_(True)) != 0


def test_md5sum():
    """Test md5sum calculation."""
    tempdir = _TempDir()
    fname1 = op.join(tempdir, 'foo')
    fname2 = op.join(tempdir, 'bar')
    with open(fname1, 'wb') as fid:
        fid.write(b'abcd')
    with open(fname2, 'wb') as fid:
        fid.write(b'efgh')
    assert_equal(md5sum(fname1), md5sum(fname1, 1))
    assert_equal(md5sum(fname2), md5sum(fname2, 1024))
    assert (md5sum(fname1) != md5sum(fname2))


def test_tempdir():
    """Test TempDir."""
    tempdir2 = _TempDir()
    assert (op.isdir(tempdir2))
    x = str(tempdir2)
    del tempdir2
    assert (not op.isdir(x))


def test_estimate_rank():
    """Test rank estimation."""
    data = np.eye(10)
    assert_array_equal(estimate_rank(data, return_singular=True)[1],
                       np.ones(10))
    data[0, 0] = 0
    assert_equal(estimate_rank(data), 9)
    pytest.raises(ValueError, estimate_rank, data, 'foo')


def test_logging():
    """Test logging (to file)."""
    pytest.raises(ValueError, set_log_level, 'foo')
    tempdir = _TempDir()
    test_name = op.join(tempdir, 'test.log')
    with open(fname_log, 'r') as old_log_file:
        # [:-1] used to strip an extra "No baseline correction applied"
        old_lines = clean_lines(old_log_file.readlines())
        old_lines.pop(-1)
    with open(fname_log_2, 'r') as old_log_file_2:
        old_lines_2 = clean_lines(old_log_file_2.readlines())
        old_lines_2.pop(14)
        old_lines_2.pop(-1)

    if op.isfile(test_name):
        os.remove(test_name)
    # test it one way (printing default off)
    set_log_file(test_name)
    set_log_level('WARNING')
    # should NOT print
    evoked = read_evokeds(fname_evoked, condition=1)
    with open(test_name) as fid:
        assert (fid.readlines() == [])
    # should NOT print
    evoked = read_evokeds(fname_evoked, condition=1, verbose=False)
    with open(test_name) as fid:
        assert (fid.readlines() == [])
    # should NOT print
    evoked = read_evokeds(fname_evoked, condition=1, verbose='WARNING')
    with open(test_name) as fid:
        assert (fid.readlines() == [])
    # SHOULD print
    evoked = read_evokeds(fname_evoked, condition=1, verbose=True)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    set_log_file(None)  # Need to do this to close the old file
    os.remove(test_name)

    # now go the other way (printing default on)
    set_log_file(test_name)
    set_log_level('INFO')
    # should NOT print
    evoked = read_evokeds(fname_evoked, condition=1, verbose='WARNING')
    with open(test_name) as fid:
        assert (fid.readlines() == [])
    # should NOT print
    evoked = read_evokeds(fname_evoked, condition=1, verbose=False)
    with open(test_name) as fid:
        assert (fid.readlines() == [])
    # SHOULD print
    evoked = read_evokeds(fname_evoked, condition=1)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)
    # check to make sure appending works (and as default, raises a warning)
    set_log_file(test_name, overwrite=False)
    with pytest.warns(RuntimeWarning, match='appended to the file'):
        set_log_file(test_name)
    evoked = read_evokeds(fname_evoked, condition=1)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines_2)

    # make sure overwriting works
    set_log_file(test_name, overwrite=True)
    # this line needs to be called to actually do some logging
    evoked = read_evokeds(fname_evoked, condition=1)
    del evoked
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert_equal(new_lines, old_lines)


@testing.requires_testing_data
def test_datasets():
    """Test dataset config."""
    # gh-4192
    data_path = testing.data_path(download=False)
    os.environ['MNE_DATASETS_TESTING_PATH'] = op.dirname(data_path)
    assert testing.data_path(download=False) == data_path


def test_config():
    """Test mne-python config file support."""
    tempdir = _TempDir()
    key = '_MNE_PYTHON_CONFIG_TESTING'
    value = '123456'
    value2 = '123'
    old_val = os.getenv(key, None)
    os.environ[key] = value
    assert (get_config(key) == value)
    del os.environ[key]
    # catch the warning about it being a non-standard config key
    assert (len(set_config(None, None)) > 10)  # tuple of valid keys
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
    # Check if get_config with key=None returns all config
    key = 'MNE_PYTHON_TESTING_KEY'
    assert key not in get_config(home_dir=tempdir)
    with pytest.warns(RuntimeWarning, match='non-standard'):
        set_config(key, value, home_dir=tempdir)
    assert_equal(get_config(home_dir=tempdir)[key], value)
    old_val = os.environ.get(key)
    try:  # os.environ should take precedence over config file
        os.environ[key] = value2
        assert_equal(get_config(home_dir=tempdir)[key], value2)
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


@testing.requires_testing_data
def test_show_fiff():
    """Test show_fiff."""
    # this is not exhaustive, but hopefully bugs will be found in use
    info = show_fiff(fname_evoked)
    keys = ['FIFF_EPOCH', 'FIFFB_HPI_COIL', 'FIFFB_PROJ_ITEM',
            'FIFFB_PROCESSED_DATA', 'FIFFB_EVOKED', 'FIFF_NAVE',
            'FIFF_EPOCH']
    assert (all(key in info for key in keys))
    info = show_fiff(fname_raw, read_limit=1024)
    assert ('COORD_TRANS' in show_fiff(fname_fsaverage_trans))


@deprecated('message')
def deprecated_func():
    pass


@deprecated('message')
class deprecated_class(object):

    def __init__(self):
        pass


def test_deprecated():
    """Test deprecated function."""
    pytest.deprecated_call(deprecated_func)
    pytest.deprecated_call(deprecated_class)


def test_how_to_deal_with_warnings():
    """Test filter some messages out of warning records."""
    with pytest.warns(UserWarning, match='bb') as w:
        warnings.warn("aa warning", UserWarning)
        warnings.warn("bb warning", UserWarning)
        warnings.warn("bb warning", RuntimeWarning)
        warnings.warn("aa warning", UserWarning)
    filter_out_warnings(w, category=UserWarning, match='aa')
    filter_out_warnings(w, category=RuntimeWarning)
    assert len(w) == 1


def _test_fetch(url):
    """Test URL retrieval."""
    tempdir = _TempDir()
    with ArgvSetter(disable_stderr=False):  # to capture stdout
        archive_name = op.join(tempdir, "download_test")
        _fetch_file(url, archive_name, timeout=30., verbose=False,
                    resume=False)
        pytest.raises(Exception, _fetch_file, 'NOT_AN_ADDRESS',
                      op.join(tempdir, 'test'), verbose=False)
        resume_name = op.join(tempdir, "download_resume")
        # touch file
        with open(resume_name + '.part', 'w'):
            os.utime(resume_name + '.part', None)
        _fetch_file(url, resume_name, resume=True, timeout=30.,
                    verbose=False)
        pytest.raises(ValueError, _fetch_file, url, archive_name,
                      hash_='a', verbose=False)
        pytest.raises(RuntimeError, _fetch_file, url, archive_name,
                      hash_='a' * 32, verbose=False)


@requires_good_network
def test_fetch_file_html():
    """Test file downloading over http."""
    _test_fetch('http://google.com')


def test_sum_squared():
    """Test optimized sum of squares."""
    X = np.random.RandomState(0).randint(0, 50, (3, 3))
    assert_equal(np.sum(X ** 2), sum_squared(X))


def test_sizeof_fmt():
    """Test sizeof_fmt."""
    assert_equal(sizeof_fmt(0), '0 bytes')
    assert_equal(sizeof_fmt(1), '1 byte')
    assert_equal(sizeof_fmt(1000), '1000 bytes')


def test_url_to_local_path():
    """Test URL to local path."""
    assert_equal(_url_to_local_path('http://google.com/home/why.html', '.'),
                 op.join('.', 'home', 'why.html'))


def test_check_type_picks():
    """Test checking type integrity checks of picks."""
    picks = np.arange(12)
    assert_array_equal(picks, _check_type_picks(picks))
    picks = list(range(12))
    assert_array_equal(np.array(picks), _check_type_picks(picks))
    picks = None
    assert_array_equal(None, _check_type_picks(picks))
    picks = ['a', 'b']
    pytest.raises(TypeError, _check_type_picks, picks)
    picks = 'b'
    pytest.raises(TypeError, _check_type_picks, picks)


def test_compute_corr():
    """Test Anscombe's Quartett."""
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y = np.array([[8.04, 6.95, 7.58, 8.81, 8.33, 9.96,
                   7.24, 4.26, 10.84, 4.82, 5.68],
                  [9.14, 8.14, 8.74, 8.77, 9.26, 8.10,
                   6.13, 3.10, 9.13, 7.26, 4.74],
                  [7.46, 6.77, 12.74, 7.11, 7.81, 8.84,
                   6.08, 5.39, 8.15, 6.42, 5.73],
                  [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
                  [6.58, 5.76, 7.71, 8.84, 8.47, 7.04,
                   5.25, 12.50, 5.56, 7.91, 6.89]])

    r = compute_corr(x, y.T)
    r2 = np.array([np.corrcoef(x, y[i])[0, 1]
                   for i in range(len(y))])
    assert_allclose(r, r2)
    pytest.raises(ValueError, compute_corr, [1, 2], [])


def test_create_slices():
    """Test checking the create of time create_slices."""
    # Test that create_slices default provide an empty list
    assert (create_slices(0, 0) == [])
    # Test that create_slice return correct number of slices
    assert (len(create_slices(0, 100)) == 100)
    # Test with non-zero start parameters
    assert (len(create_slices(50, 100)) == 50)
    # Test slices' length with non-zero start and window_width=2
    assert (len(create_slices(0, 100, length=2)) == 50)
    # Test slices' length with manual slice separation
    assert (len(create_slices(0, 100, step=10)) == 10)
    # Test slices' within length for non-consecutive samples
    assert (len(create_slices(0, 500, length=50, step=10)) == 46)
    # Test that slices elements start, stop and step correctly
    slices = create_slices(0, 10)
    assert (slices[0].start == 0)
    assert (slices[0].step == 1)
    assert (slices[0].stop == 1)
    assert (slices[-1].stop == 10)
    # Same with larger window width
    slices = create_slices(0, 9, length=3)
    assert (slices[0].start == 0)
    assert (slices[0].step == 1)
    assert (slices[0].stop == 3)
    assert (slices[-1].stop == 9)
    # Same with manual slices' separation
    slices = create_slices(0, 9, length=3, step=1)
    assert (len(slices) == 7)
    assert (slices[0].step == 1)
    assert (slices[0].stop == 3)
    assert (slices[-1].start == 6)
    assert (slices[-1].stop == 9)


def test_time_mask():
    """Test safe time masking."""
    N = 10
    x = np.arange(N).astype(float)
    assert_equal(_time_mask(x, 0, N - 1).sum(), N)
    assert_equal(_time_mask(x - 1e-10, 0, N - 1, sfreq=1000.).sum(), N)
    assert_equal(_time_mask(x - 1e-10, None, N - 1, sfreq=1000.).sum(), N)
    assert_equal(_time_mask(x - 1e-10, None, None, sfreq=1000.).sum(), N)
    assert_equal(_time_mask(x - 1e-10, -np.inf, None, sfreq=1000.).sum(), N)
    assert_equal(_time_mask(x - 1e-10, None, np.inf, sfreq=1000.).sum(), N)
    # non-uniformly spaced inputs
    x = np.array([4, 10])
    assert_equal(_time_mask(x[:1], tmin=10, sfreq=1,
                            raise_error=False).sum(), 0)
    assert_equal(_time_mask(x[:1], tmin=11, tmax=12, sfreq=1,
                            raise_error=False).sum(), 0)
    assert_equal(_time_mask(x, tmin=10, sfreq=1).sum(), 1)
    assert_equal(_time_mask(x, tmin=6, sfreq=1).sum(), 1)
    assert_equal(_time_mask(x, tmin=5, sfreq=1).sum(), 1)
    assert_equal(_time_mask(x, tmin=4.5001, sfreq=1).sum(), 1)
    assert_equal(_time_mask(x, tmin=4.4999, sfreq=1).sum(), 2)
    assert_equal(_time_mask(x, tmin=4, sfreq=1).sum(), 2)
    # degenerate cases
    pytest.raises(ValueError, _time_mask, x[:1], tmin=11, tmax=12)
    pytest.raises(ValueError, _time_mask, x[:1], tmin=10, sfreq=1)


def test_random_permutation():
    """Test random permutation function."""
    n_samples = 10
    random_state = 42
    python_randperm = random_permutation(n_samples, random_state)

    # matlab output when we execute rng(42), randperm(10)
    matlab_randperm = np.array([7, 6, 5, 1, 4, 9, 10, 3, 8, 2])

    assert_array_equal(python_randperm, matlab_randperm - 1)


def test_copy_doc():
    """Test decorator for copying docstrings."""
    class A:
        def m1():
            """Docstring for m1."""
            pass

    class B:
        def m1():
            pass

    class C (A):
        @copy_doc(A.m1)
        def m1():
            pass

    assert_equal(C.m1.__doc__, 'Docstring for m1.')
    pytest.raises(ValueError, copy_doc(B.m1), C.m1)


def test_copy_function_doc_to_method_doc():
    """Test decorator for re-using function docstring as method docstrings."""
    def f1(object, a, b, c):
        """Docstring for f1.

        Parameters
        ----------
        object : object
            Some object. This description also has

            blank lines in it.
        a : int
            Parameter a
        b : int
            Parameter b
        """
        pass

    def f2(object):
        """Docstring for f2.

        Parameters
        ----------
        object : object
            Only one parameter

        Returns
        -------
        nothing.
        """
        pass

    def f3(object):
        """Docstring for f3.

        Parameters
        ----------
        object : object
            Only one parameter
        """
        pass

    def f4(object):
        """Docstring for f4."""
        pass

    def f5(object):  # noqa: D410, D411, D414
        """Docstring for f5.

        Parameters
        ----------
        Returns
        -------
        nothing.
        """
        pass

    class A:
        @copy_function_doc_to_method_doc(f1)
        def method_f1(self, a, b, c):
            pass

        @copy_function_doc_to_method_doc(f2)
        def method_f2(self):
            "method_f3 own docstring"
            pass

        @copy_function_doc_to_method_doc(f3)
        def method_f3(self):
            pass

    assert_equal(
        A.method_f1.__doc__,
        """Docstring for f1.

        Parameters
        ----------
        a : int
            Parameter a
        b : int
            Parameter b
        """
    )

    assert_equal(
        A.method_f2.__doc__,
        """Docstring for f2.

        Returns
        -------
        nothing.
        method_f3 own docstring"""
    )

    assert_equal(A.method_f3.__doc__, 'Docstring for f3.\n\n        ')
    pytest.raises(ValueError, copy_function_doc_to_method_doc(f4), A.method_f1)
    pytest.raises(ValueError, copy_function_doc_to_method_doc(f5), A.method_f1)


def test_progressbar():
    """Test progressbar class."""
    a = np.arange(10)
    pbar = ProgressBar(a)
    assert_equal(a, pbar.iterable)
    assert_equal(10, pbar.max_value)

    pbar = ProgressBar(10)
    assert_equal(10, pbar.max_value)
    assert (pbar.iterable is None)

    # Make sure that non-iterable input raises an error
    def iter_func(a):
        for ii in a:
            pass
    pytest.raises(ValueError, iter_func, ProgressBar(20))


def myfun(x):
    """Check url."""
    assert 'martinos' in x


def _identity(x):
    return x


def test_progressbar_parallel_basic(capsys):
    """Test ProgressBar with parallel computing, basic version."""
    assert capsys.readouterr().out == ''
    parallel, p_fun, _ = parallel_func(_identity, total=10, n_jobs=1,
                                       verbose=True)
    out = parallel(p_fun(x) for x in range(10))
    assert out == list(range(10))
    assert '100.00%' in capsys.readouterr().out


def _identity_block(x, pb):
    for ii in range(len(x)):
        pb.update(ii + 1)
    return x


def test_progressbar_parallel_advanced(capsys):
    """Test ProgressBar with parallel computing, advanced version."""
    assert capsys.readouterr().out == ''
    # This must be "1" because "capsys" won't get stdout properly otherwise
    parallel, p_fun, _ = parallel_func(_identity_block, n_jobs=1,
                                       verbose=False)
    arr = np.arange(10)
    with ProgressBar(len(arr), verbose_bool=True) as pb:
        out = parallel(p_fun(x, pb.subset(pb_idx))
                       for pb_idx, x in array_split_idx(arr, 2))
        assert op.isfile(pb._mmap_fname)
        sum_ = np.memmap(pb._mmap_fname, dtype='bool', mode='r',
                         shape=10).sum()
        assert sum_ == len(arr)
    assert not op.isfile(pb._mmap_fname), '__exit__ not called?'
    out = np.concatenate(out)
    assert_array_equal(out, arr)
    assert '100.00%' in capsys.readouterr().out


def _identity_block_wide(x, pb):
    for ii in range(len(x)):
        for jj in range(2):
            pb.update(ii * 2 + jj + 1)
    return x, pb.idx


def test_progressbar_parallel_more(capsys):
    """Test ProgressBar with parallel computing, advanced version."""
    assert capsys.readouterr().out == ''
    # This must be "1" because "capsys" won't get stdout properly otherwise
    parallel, p_fun, _ = parallel_func(_identity_block_wide, n_jobs=1,
                                       verbose=False)
    arr = np.arange(10)
    with ProgressBar(len(arr) * 2, verbose_bool=True) as pb:
        out = parallel(p_fun(x, pb.subset(pb_idx))
                       for pb_idx, x in array_split_idx(arr, 2, n_per_split=2))
        idxs = np.concatenate([o[1] for o in out])
        assert_array_equal(idxs, np.arange(len(arr) * 2))
        out = np.concatenate([o[0] for o in out])
        assert op.isfile(pb._mmap_fname)
        sum_ = np.memmap(pb._mmap_fname, dtype='bool', mode='r',
                         shape=len(arr) * 2).sum()
        assert sum_ == len(arr) * 2
    assert not op.isfile(pb._mmap_fname), '__exit__ not called?'
    assert '100.00%' in capsys.readouterr().out


def test_open_docs():
    """Test doc launching."""
    old_tab = webbrowser.open_new_tab
    try:
        # monkey patch temporarily to prevent tabs from actually spawning
        webbrowser.open_new_tab = myfun
        open_docs()
        open_docs('tutorials', 'dev')
        open_docs('examples', 'stable')
        pytest.raises(ValueError, open_docs, 'foo')
        pytest.raises(ValueError, open_docs, 'api', 'foo')
    finally:
        webbrowser.open_new_tab = old_tab


def test_linkcode_resolve():
    """Test linkcode resolving."""
    ex = '' if sys.version[0] == '2' else '#L'
    url = linkcode_resolve('py', dict(module='mne', fullname='Epochs'))
    assert '/mne/epochs.py' + ex in url
    url = linkcode_resolve('py', dict(module='mne',
                                      fullname='compute_covariance'))
    assert '/mne/cov.py' + ex in url
    url = linkcode_resolve('py', dict(module='mne',
                                      fullname='convert_forward_solution'))
    assert '/mne/forward/forward.py' + ex in url
    url = linkcode_resolve('py', dict(module='mne',
                                      fullname='datasets.sample.data_path'))
    assert '/mne/datasets/sample/sample.py' + ex in url


def test_warn(capsys):
    """Test the smart warn() function."""
    with pytest.warns(RuntimeWarning, match='foo'):
        warn('foo')
    captured = capsys.readouterr()
    assert captured.out == ''  # gh-5592
    assert captured.err == ''  # this is because pytest.warns took it already


def test_reg_pinv():
    """Test regularization and inversion of covariance matrix."""
    # create rank-deficient array
    a = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])

    # Test if rank-deficient matrix without regularization throws
    # specific warning
    with pytest.warns(RuntimeWarning, match='deficient'):
        _reg_pinv(a, reg=0.)

    # Test inversion with explicit rank
    a_inv_np = np.linalg.pinv(a)
    a_inv_mne, loading_factor, rank = _reg_pinv(a, rank=2)
    assert loading_factor == 0
    assert rank == 2
    assert_array_equal(a_inv_np, a_inv_mne)

    # Test inversion with automatic rank detection
    a_inv_mne, _, estimated_rank = _reg_pinv(a, rank=None)
    assert_array_equal(a_inv_np, a_inv_mne)
    assert estimated_rank == 2

    # Test adding regularization
    a_inv_mne, loading_factor, estimated_rank = _reg_pinv(a, reg=2)
    # Since A has a diagonal of all ones, loading_factor should equal the
    # regularization parameter
    assert loading_factor == 2
    # The estimated rank should be that of the non-regularized matrix
    assert estimated_rank == 2
    # Test result against the NumPy version
    a_inv_np = np.linalg.pinv(a + loading_factor * np.eye(3))
    assert_array_equal(a_inv_np, a_inv_mne)

    # Test setting rcond
    a_inv_np = np.linalg.pinv(a, rcond=0.5)
    a_inv_mne, _, estimated_rank = _reg_pinv(a, rcond=0.5)
    assert_array_equal(a_inv_np, a_inv_mne)
    assert estimated_rank == 1

    # Test inverting an all zero cov
    a_inv, loading_factor, estimated_rank = _reg_pinv(np.zeros((3, 3)), reg=2)
    assert_array_equal(a_inv, 0)
    assert loading_factor == 0
    assert estimated_rank == 0


run_tests_if_main()
