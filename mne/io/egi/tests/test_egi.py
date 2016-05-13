# -*- coding: utf-8 -*-
# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license


import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_true, assert_raises, assert_equal

from mne import find_events, pick_types
from mne.io import read_raw_egi
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.egi import _combine_triggers
from mne.utils import run_tests_if_main

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
egi_fname = op.join(base_dir, 'test_egi.raw')
egi_txt_fname = op.join(base_dir, 'test_egi.txt')


def test_io_egi():
    """Test importing EGI simple binary files"""
    # test default
    with open(egi_txt_fname) as fid:
        data = np.loadtxt(fid)
    t = data[0]
    data = data[1:]
    data *= 1e-6  # μV

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw = read_raw_egi(egi_fname, include=None)
        assert_true('RawEGI' in repr(raw))
        assert_equal(len(w), 1)
        assert_true(w[0].category == RuntimeWarning)
        msg = 'Did not find any event code with more than one event.'
        assert_true(msg in '%s' % w[0].message)
    data_read, t_read = raw[:256]
    assert_allclose(t_read, t)
    assert_allclose(data_read, data, atol=1e-10)

    include = ['TRSP', 'XXX1']
    with warnings.catch_warnings(record=True):  # preload=None
        raw = _test_raw_reader(read_raw_egi, input_fname=egi_fname,
                               include=include)

    assert_equal('eeg' in raw, True)

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
    triggers = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    events_ids = [12, 24]
    new_trigger = _combine_triggers(triggers, events_ids)
    assert_array_equal(np.unique(new_trigger), np.unique([0, 12, 24]))

    assert_raises(ValueError, read_raw_egi, egi_fname, include=['Foo'],
                  preload=False)
    assert_raises(ValueError, read_raw_egi, egi_fname, exclude=['Bar'],
                  preload=False)
    for ii, k in enumerate(include, 1):
        assert_true(k in raw.event_id)
        assert_true(raw.event_id[k] == ii)

run_tests_if_main()
