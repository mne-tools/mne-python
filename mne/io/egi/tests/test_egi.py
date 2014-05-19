# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license


import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true, assert_raises, assert_equal

from mne import find_events
from mne.io import read_raw_egi
from mne.io.egi import _combine_triggers
from mne import pick_types
from mne.io import Raw
from mne.utils import _TempDir

warnings.simplefilter('always')  # enable b/c these tests throw warnings
tempdir = _TempDir()

base_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
egi_fname = op.join(base_dir, 'test_egi.raw')


def test_io_egi():
    """Test importing EGI simple binary files"""
    # test default
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', category=RuntimeWarning)
        _ = read_raw_egi(egi_fname, include=None)
        assert_equal(len(w), 1)
        assert_true(w[0].category == RuntimeWarning)
        msg = 'Did not find any event code with more than one event.'
        assert_true(msg in '%s' % w[0].message)

    include = ['TRSP', 'XXX1']
    raw = read_raw_egi(egi_fname, include=include)

    _ = repr(raw)
    _ = repr(raw.info)  # analysis:ignore, noqa

    assert_equal('eeg' in raw, True)
    out_fname = op.join(tempdir, 'test_egi_raw.fif')
    raw.save(out_fname)

    raw2 = Raw(out_fname, preload=True)
    data1, times1 = raw[:10, :]
    data2, times2 = raw2[:10, :]

    assert_array_almost_equal(data1, data2)
    assert_array_almost_equal(times1, times2)

    eeg_chan = [c for c in raw.ch_names if 'EEG' in c]
    assert_equal(len(eeg_chan), 256)
    picks = pick_types(raw.info, eeg=True)
    assert_equal(len(picks), 256)
    assert_equal('STI 014' in raw.ch_names, True)

    events = find_events(raw, stim_channel='STI 014')
    assert_equal(len(events), 2)  # ground truth
    assert_equal(np.unique(events[:, 1])[0], 0)
    assert_true(np.unique(events[:, 0])[0] != 0)
    assert_true(np.unique(events[:, 2])[0] != 0)
    triggers = np.array([[0, 1, 1, 0], [0, 0, 1, 0]])

    # test trigger functionality
    assert_raises(RuntimeError, _combine_triggers, triggers, None)
    triggers = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    events_ids = [12, 24]
    new_trigger = _combine_triggers(triggers, events_ids)
    assert_array_equal(np.unique(new_trigger), np.unique([0, 12, 24]))

    assert_raises(ValueError, read_raw_egi, egi_fname,
                  include=['Foo'])
    assert_raises(ValueError, read_raw_egi, egi_fname,
                  exclude=['Bar'])
    for ii, k in enumerate(include, 1):
        assert_true(k in raw.event_id)
        assert_true(raw.event_id[k] == ii)
