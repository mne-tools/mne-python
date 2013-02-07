import numpy as np
from os import path as op

from numpy.testing import assert_array_equal
from nose.tools import assert_true
from mne.fiff.pick import pick_channels_regexp, pick_types
from mne.fiff import Raw

base_dir = op.join(op.dirname(__file__), 'data')
fif_fname = op.join(base_dir, 'test_raw.fif')


def test_pick_channels_regexp():
    """Test pick with regular expression
    """
    ch_names = ['MEG 2331', 'MEG 2332', 'MEG 2333']
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...1'), [0])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...[2-3]'), [1, 2])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG *'), [0, 1, 2])


def test_pick_types_slice():
    """Test pick_types slice support
    """
    raw = Raw(fif_fname)
    picks = pick_types(raw.info, meg=True, eeg=False, exclude=[])
    picks_slice = pick_types(raw.info, meg=True, eeg=False, exclude=[],
                             return_slice=True)
    inds = np.arange(len(picks) + 1)
    assert_true(isinstance(picks_slice, slice))
    assert_array_equal(inds[picks], inds[picks_slice])
