# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op
from copy import deepcopy

from nose.tools import assert_true, assert_equal, assert_raises

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose)
import numpy as np
import copy as cp
import warnings

from mne import fiff, Epochs, read_events, pick_events, read_epochs
from mne.epochs import bootstrap, equalize_epoch_counts, combine_event_ids
from mne.utils import _TempDir, requires_pandas, requires_nitime
from mne.fiff import read_evoked
from mne.fiff.channels import ContainsMixin
from mne.fiff.proj import _has_eeg_average_ref_proj
from mne.event import merge_events
from mne.externals.six.moves import zip

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
raw = fiff.Raw(raw_fname, add_eeg_ref=False)
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=True,
                        ecg=True, eog=True, include=['STI 014'],
                        exclude='bads')

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

tempdir = _TempDir()


def test_epochs_bad_baseline():
    """Test Epochs initialization with bad baseline parameters
    """
    assert_raises(ValueError, Epochs, raw, events, None, -0.1, 0.3, (-0.2, 0))
    assert_raises(ValueError, Epochs, raw, events, None, -0.1, 0.3, (0, 0.4))


def test_epoch_combine_ids():
    """Test combining event ids in epochs compared to events
    """
    for preload in [False]:
        epochs = Epochs(raw, events, {'a': 1, 'b': 2, 'c': 3,
                                      'd': 4, 'e': 5, 'f': 32},
                        tmin, tmax, picks=picks, preload=preload)
        events_new = merge_events(events, [1, 2], 12)
        epochs_new = combine_event_ids(epochs, ['a', 'b'], {'ab': 12})
        assert_array_equal(events_new, epochs_new.events)
        # should probably add test + functionality for non-replacement XXX


def test_read_epochs_bad_events():
    """Test epochs when events are at the beginning or the end of the file
    """
    # Event at the beginning
    epochs = Epochs(raw, np.array([[raw.first_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))
    evoked = epochs.average()

    epochs = Epochs(raw, np.array([[raw.first_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))
    epochs.drop_bad_epochs()
    evoked = epochs.average()

    # Event at the end
    epochs = Epochs(raw, np.array([[raw.last_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))
    evoked = epochs.average()
    assert evoked


def test_read_write_epochs():
    """Test epochs from raw files with IO as fif file
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    evoked = epochs.average()
    data = epochs.get_data()

    epochs_no_id = Epochs(raw, pick_events(events, include=event_id),
                          None, tmin, tmax, picks=picks,
                          baseline=(None, 0))
    assert_array_equal(data, epochs_no_id.get_data())

    eog_picks = fiff.pick_types(raw.info, meg=False, eeg=False, stim=False,
                                eog=True, exclude='bads')
    eog_ch_names = [raw.ch_names[k] for k in eog_picks]
    epochs.drop_channels(eog_ch_names)
    assert_true(len(epochs.info['chs']) == len(epochs.ch_names)
                == epochs.get_data().shape[1])
    data_no_eog = epochs.get_data
    events1 = events[events[:, 2] == event_id]

    # Bound checks
    assert_raises(IndexError, epochs.drop_epochs, [len(epochs.events)])
    assert_raises(IndexError, epochs.drop_epochs, [-1])
    assert_raises(ValueError, epochs.drop_epochs, [[1, 2], [3, 4]])

    # Test selection attribute
    assert_array_equal(epochs.selection,
                       np.where(events[:, 2] == event_id)[0])
    assert_equal(len(epochs.drop_log), len(events))
    assert_true(all(epochs.drop_log[k] == ['IGNORED']
                 for k in set(range(len(events))) - set(epochs.selection)))

    selection = epochs.selection.copy()
    epochs.drop_epochs([2, 4], reason='d')
    assert_equal(len(epochs.drop_log), len(events))
    assert_equal([epochs.drop_log[k] for k in selection[[2, 4]]], [['d'],['d']])
    assert_array_equal(events[epochs.selection], events1[[0, 1, 3, 5, 6]])
    assert_array_equal(events[epochs[3:].selection], events1[[5, 6]])
    assert_array_equal(events[epochs['1'].selection], events1[[0, 1, 3, 5, 6]])


def test_drop_epochs_mult():
    """Test that subselecting epochs or making less epochs is equivalent"""
    for preload in [True, False]:
        epochs1 = Epochs(raw, events, {'a': 1, 'b': 2},
                         tmin, tmax, picks=picks, reject=reject,
                         preload=preload)['a']
        epochs2 = Epochs(raw, events, {'a': 1},
                         tmin, tmax, picks=picks, reject=reject,
                         preload=preload)

        if preload:
            # In the preload case you cannot know the bads if already ignored
            assert_equal(len(epochs1.drop_log), len(epochs2.drop_log))
            for d1, d2 in zip(epochs1.drop_log, epochs2.drop_log):
                if d1 == ['IGNORED']:
                    assert_true(d2 == ['IGNORED'])
                if d1 != ['IGNORED'] and d1 != []:
                    assert_true((d2 == d1) or (d2 == ['IGNORED']))
                if d1 == []:
                    assert_true(d2 == [])
            assert_array_equal(epochs1.events, epochs2.events)
            assert_array_equal(epochs1.selection, epochs2.selection)
        else:
            # In the non preload is should be exactly the same
            assert_equal(epochs1.drop_log, epochs2.drop_log)
            assert_array_equal(epochs1.events, epochs2.events)
            assert_array_equal(epochs1.selection, epochs2.selection)


def test_contains():
    """Test membership API"""

    tests = [(('mag', False), ('grad', 'eeg')),
             (('grad', False), ('mag', 'eeg')),
             ((False, True), ('grad', 'mag'))]
    
    for (meg, eeg), others in tests:
        picks_contains = fiff.pick_types(raw.info, meg=meg, eeg=eeg)
        epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax,
                        picks=picks_contains, reject=None,
                        preload=False)
        test = 'eeg' if eeg is True else meg
        assert_true(test in epochs)
        assert_true(not any(o in epochs for o in others))
    
    assert_raises(ValueError, epochs.__contains__, 'foo')
    assert_raises(ValueError, epochs.__contains__, 1)


def test_drop_channels_mixin():
    """Test channels-dropping functionality
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    drop_ch = epochs.ch_names[:3]
    ch_names = epochs.ch_names[3:]
    epochs.drop_channels(drop_ch)
    assert_equal(ch_names, epochs.ch_names)
    assert_equal(len(ch_names), epochs.get_data().shape[1])
