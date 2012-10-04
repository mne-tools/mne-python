# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import copy as cp

from mne import fiff, Epochs, read_events, pick_events
from mne.epochs import bootstrap

raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test_raw.fif')
event_name = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test-eve.fif')
evoked_nf_name = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
raw = fiff.Raw(raw_fname)
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=True,
                        ecg=True, eog=True, include=['STI 014'])

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)


def test_read_epochs():
    """Reading epochs from raw files
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    epochs.average()
    data = epochs.get_data()

    epochs_no_id = Epochs(raw, pick_events(events, include=event_id),
                          None, tmin, tmax, picks=picks,
                          baseline=(None, 0))
    assert_array_equal(data, epochs_no_id.get_data())

    eog_picks = fiff.pick_types(raw.info, meg=False, eeg=False, stim=False,
                                eog=True)
    epochs.drop_picks(eog_picks)
    data_no_eog = epochs.get_data()
    assert_true(data.shape[1] == (data_no_eog.shape[1] + len(eog_picks)))


def test_epochs_proj():
    """Test handling projection (apply proj in Raw or in Epochs)
    """
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more
    this_picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=True,
                                 eog=True, exclude=exclude)
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    baseline=(None, 0), proj=True)
    epochs.average()
    data = epochs.get_data()

    raw_proj = fiff.Raw(raw_fname, proj=True)
    epochs_no_proj = Epochs(raw_proj, events[:4], event_id, tmin, tmax,
                            picks=this_picks, baseline=(None, 0), proj=False)
    epochs_no_proj.average()
    data_no_proj = epochs_no_proj.get_data()
    assert_array_almost_equal(data, data_no_proj, decimal=8)


def test_evoked_arithmetic():
    """Arithmetic of evoked data"""
    epochs1 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    evoked1 = epochs1.average()
    epochs2 = Epochs(raw, events[4:8], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    evoked2 = epochs2.average()
    epochs = Epochs(raw, events[:8], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    evoked = epochs.average()
    evoked_sum = evoked1 + evoked2
    assert_array_equal(evoked.data, evoked_sum.data)
    assert_array_equal(evoked.times, evoked_sum.times)
    assert_true(evoked_sum.nave == (evoked1.nave + evoked2.nave))
    evoked_diff = evoked1 - evoked1
    assert_array_equal(np.zeros_like(evoked.data), evoked_diff.data)


def test_reject_epochs():
    """Test of epochs rejection
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0),
                        reject=reject, flat=flat)
    n_events = len(epochs.events)
    data = epochs.get_data()
    n_clean_epochs = len(data)
    # Should match
    # mne_process_raw --raw test_raw.fif --projoff \
    #   --saveavetag -ave --ave test.ave --filteroff
    assert_true(n_events > n_clean_epochs)
    assert_true(n_clean_epochs == 3)
    assert_true(epochs.drop_log == [[], [], [], ['MEG 2443'], \
                                  ['MEG 2443'], ['MEG 2443'], ['MEG 2443']])


def test_preload_epochs():
    """Test preload of epochs
    """
    epochs_preload = Epochs(raw, events[:16], event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0), preload=True,
                        reject=reject, flat=flat)
    data_preload = epochs_preload.get_data()

    epochs = Epochs(raw, events[:16], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=False,
                        reject=reject, flat=flat)
    data = epochs.get_data()
    assert_array_equal(data_preload, data)
    assert_array_equal(epochs_preload.average().data, epochs.average().data)


def test_indexing_slicing():
    """Test of indexing and slicing operations
    """
    epochs = Epochs(raw, events[:20], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=False,
                    reject=reject, flat=flat)

    data_normal = epochs.get_data()

    n_good_events = data_normal.shape[0]

    # indices for slicing
    start_index = 1
    end_index = n_good_events - 1

    assert((end_index - start_index) > 0)

    for preload in [True, False]:
        epochs2 = Epochs(raw, events[:20], event_id, tmin, tmax,
                         picks=picks, baseline=(None, 0), preload=preload,
                         reject=reject, flat=flat)

        if not preload:
            epochs2.drop_bad_epochs()

        # using slicing
        epochs2_sliced = epochs2[start_index:end_index]

        data_epochs2_sliced = epochs2_sliced.get_data()
        assert_array_equal(data_epochs2_sliced,
                           data_normal[start_index:end_index])

        # using indexing
        pos = 0
        for idx in range(start_index, end_index):
            data = epochs2_sliced[pos].get_data()
            assert_array_equal(data[0], data_normal[idx])
            pos += 1

        # using indexing with an int
        data = epochs2[data_epochs2_sliced.shape[0]].get_data()
        assert_array_equal(data, data_normal[[idx]])

        # using indexing with an array
        idx = np.random.randint(0, data_epochs2_sliced.shape[0], 10)
        data = epochs2[idx].get_data()
        assert_array_equal(data, data_normal[idx])

        # using indexing with a list of indices
        idx = [0]
        data = epochs2[idx].get_data()
        assert_array_equal(data, data_normal[idx])
        idx = [0, 1]
        data = epochs2[idx].get_data()
        assert_array_equal(data, data_normal[idx])


def test_comparision_with_c():
    """Test of average obtained vs C code
    """
    c_evoked = fiff.Evoked(evoked_nf_name, setno=0)
    epochs = Epochs(raw, events, event_id, tmin, tmax,
                        baseline=None, preload=True,
                        reject=None, flat=None)
    evoked = epochs.average()
    sel = fiff.pick_channels(c_evoked.ch_names, evoked.ch_names)
    evoked_data = evoked.data
    c_evoked_data = c_evoked.data[sel]

    assert_true(evoked.nave == c_evoked.nave)
    assert_array_almost_equal(evoked_data, c_evoked_data, 10)
    assert_array_almost_equal(evoked.times, c_evoked.times, 12)


def test_crop():
    """Test of crop of epochs
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=False,
                    reject=reject, flat=flat)
    data_normal = epochs.get_data()

    epochs2 = Epochs(raw, events[:5], event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)

    # indices for slicing
    tmin_window = tmin + 0.1
    tmax_window = tmax - 0.1
    tmask = (epochs.times >= tmin_window) & (epochs.times <= tmax_window)
    assert_true(tmin_window > tmin)
    assert_true(tmax_window < tmax)
    epochs3 = epochs2.crop(tmin_window, tmax_window, copy=True)
    data3 = epochs3.get_data()
    epochs2.crop(tmin_window, tmax_window)
    data2 = epochs2.get_data()
    assert_array_equal(data2, data_normal[:, :, tmask])
    assert_array_equal(data3, data_normal[:, :, tmask])

def test_resample():
    """Test of resample of epochs
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)
    data_normal = cp.deepcopy(epochs.get_data())
    times_normal = cp.deepcopy(epochs.times)
    sfreq_normal = epochs.info['sfreq']
    # upsample by 2
    epochs.resample(sfreq_normal*2)
    data_up = cp.deepcopy(epochs.get_data())
    times_up = cp.deepcopy(epochs.times)
    sfreq_up = epochs.info['sfreq']
    # downsamply by 2, which should match
    epochs.resample(sfreq_normal)
    data_new = cp.deepcopy(epochs.get_data())
    times_new = cp.deepcopy(epochs.times)
    sfreq_new = epochs.info['sfreq']

    assert_true(data_up.shape[2] == 2*data_normal.shape[2])
    assert_true(sfreq_up == 2*sfreq_normal)
    assert_true(sfreq_new == sfreq_normal)
    assert_true(len(times_up) == 2*len(times_normal))
    assert_array_almost_equal(times_new, times_normal, 10)
    assert_true(data_up.shape[2] == 2*data_normal.shape[2])
    assert_array_almost_equal(data_new, data_normal, 2)

def test_bootstrap():
    """Test of bootstrapping of epochs
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)
    epochs2 = bootstrap(epochs, random_state=0)
    assert_true(len(epochs2.events) == len(epochs.events))
    assert_true(epochs._data.shape == epochs2._data.shape)
