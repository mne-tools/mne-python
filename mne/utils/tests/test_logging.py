import os
import os.path as op
import warnings

import pytest

from mne import read_evokeds
from mne.utils import (warn, set_log_level, set_log_file, filter_out_warnings,
                       )

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')
fname_log = op.join(base_dir, 'test-ave.log')
fname_log_2 = op.join(base_dir, 'test-ave-2.log')


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


def clean_lines(lines=[]):
    """Scrub filenames for checking logging output (in test_logging)."""
    return [l if 'Reading ' not in l else 'Reading test file' for l in lines]


def test_logging(tmpdir):
    """Test logging (to file)."""
    pytest.raises(ValueError, set_log_level, 'foo')
    tempdir = str(tmpdir)
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
    assert new_lines == old_lines
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
    assert new_lines == old_lines
    # check to make sure appending works (and as default, raises a warning)
    set_log_file(test_name, overwrite=False)
    with pytest.warns(RuntimeWarning, match='appended to the file'):
        set_log_file(test_name)
    evoked = read_evokeds(fname_evoked, condition=1)
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert new_lines == old_lines_2

    # make sure overwriting works
    set_log_file(test_name, overwrite=True)
    # this line needs to be called to actually do some logging
    evoked = read_evokeds(fname_evoked, condition=1)
    del evoked
    with open(test_name, 'r') as new_log_file:
        new_lines = clean_lines(new_log_file.readlines())
    assert new_lines == old_lines


def test_warn(capsys):
    """Test the smart warn() function."""
    with pytest.warns(RuntimeWarning, match='foo'):
        warn('foo')
    captured = capsys.readouterr()
    assert captured.out == ''  # gh-5592
    assert captured.err == ''  # this is because pytest.warns took it already
