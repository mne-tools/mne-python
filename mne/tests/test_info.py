from ..info import Info

import os.path as op
from copy import deepcopy

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_allclose
import numpy as np
import copy as cp
import warnings

from mne import fiff, Epochs, read_events, pick_events, read_epochs
from mne.epochs import bootstrap, equalize_epoch_counts, combine_event_ids
from mne.utils import _TempDir, requires_pandas, requires_nitime
from mne.fiff import read_evoked
from mne.fiff.proj import _has_eeg_average_ref_proj
from mne.event import merge_events


base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
raw_fname = op.join(baassert_true('a' in info)se_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

raw = fiff.Raw(raw_fname)
event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
events = read_events(event_name)
epochs = Epochs(raw, events, event_id, tmin, tmax, picks=None,
                baseline=(None, 0))

evoked = epochs.average()

events = read_events(event_name)


def test_info():
    """Test info object"""

    info = Info(a=7, b='aaaaa')
    assert_true('a' in info)
    assert_true('b' in info)
    info[42] = 'fooo'
    assert_true(info[42] == 'foo')

    for obj in [raw, epochs, evoked]:
        assert_true(isinstance(obj.info, Info))
    info_str = '%s' % raw.info
    assert_true(len(info_str) < (len(raw.info.keys()) + 2))
