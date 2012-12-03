# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.fiff import Raw
from mne.event import read_events
from mne.artifacts.stim import eliminate_stim_artifact

data_path = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')


def test_stim_elim():
    """Test eliminate stim artifact"""
    raw = Raw(raw_fname, preload=True)
    events = read_events(event_fname)
    event_idx = np.where(events[:, 2] == 1)[0][0]
    tidx = events[event_idx, 0] - raw.first_samp

    raw = eliminate_stim_artifact(raw, events, event_id=1, tmin=-0.005,
                                  tmax=0.01, mode='linear')
    data, times = raw[:, tidx - 3:tidx + 5]
    diff_data0 = np.diff(data[0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))
    raw = eliminate_stim_artifact(raw, events, event_id=1, tmin=-0.005,
                                  tmax=0.01, mode='window')
    data, times = raw[:, tidx:tidx + 1]
    assert_true(np.all(data) == 0.)
