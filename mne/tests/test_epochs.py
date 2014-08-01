# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
from copy import deepcopy

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose)
import numpy as np
import copy as cp
import warnings

from mne import (io, Epochs, read_events, pick_events, read_epochs,
                 equalize_channels, pick_types, pick_channels, read_evokeds,
                 write_evokeds)
from mne.epochs import (bootstrap, equalize_epoch_counts, combine_event_ids,
                        add_channels_epochs, EpochsArray)
from mne.utils import (_TempDir, requires_pandas, requires_nitime,
                       clean_warning_registry)

from mne.io.meas_info import create_info
from mne.io.proj import _has_eeg_average_ref_proj
from mne.event import merge_events
from mne.io.constants import FIFF
from mne.externals.six.moves import zip
from mne.externals.six.moves import cPickle as pickle


warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
raw = io.Raw(raw_fname, add_eeg_ref=False)
events = read_events(event_name)
picks = pick_types(raw.info, meg=True, eeg=True, stim=True,
                   ecg=True, eog=True, include=['STI 014'],
                   exclude='bads')

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

tempdir = _TempDir()

clean_warning_registry()  # really clean warning stack


def test_epochs_hash():
    """Test epoch hashing
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax)
    assert_raises(RuntimeError, epochs.__hash__)
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True)
    assert_equal(hash(epochs), hash(epochs))
    epochs_2 = Epochs(raw, events, event_id, tmin, tmax, preload=True)
    assert_equal(hash(epochs), hash(epochs_2))
    # do NOT use assert_equal here, failing output is terrible
    assert_true(pickle.dumps(epochs) == pickle.dumps(epochs_2))

    epochs_2._data[0, 0, 0] -= 1
    assert_not_equal(hash(epochs), hash(epochs_2))


def test_event_ordering():
    """Test event order"""
    events2 = events.copy()
    np.random.shuffle(events2)
    for ii, eve in enumerate([events, events2]):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Epochs(raw, eve, event_id, tmin, tmax,
                   baseline=(None, 0), reject=reject, flat=flat)
            assert_equal(len(w), ii)
            if ii > 0:
                assert_true('chronologically' in '%s' % w[-1].message)


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
    with warnings.catch_warnings(record=True):
        evoked = epochs.average()

    epochs = Epochs(raw, np.array([[raw.first_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))
    epochs.drop_bad_epochs()
    with warnings.catch_warnings(record=True):
        evoked = epochs.average()

    # Event at the end
    epochs = Epochs(raw, np.array([[raw.last_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks, baseline=(None, 0))

    with warnings.catch_warnings(record=True):
        evoked = epochs.average()
        assert evoked
    warnings.resetwarnings()


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

    eog_picks = pick_types(raw.info, meg=False, eeg=False, stim=False,
                           eog=True, exclude='bads')
    eog_ch_names = [raw.ch_names[k] for k in eog_picks]
    epochs.drop_channels(eog_ch_names)
    assert_true(len(epochs.info['chs']) == len(epochs.ch_names)
                == epochs.get_data().shape[1])
    data_no_eog = epochs.get_data()
    assert_true(data.shape[1] == (data_no_eog.shape[1] + len(eog_picks)))

    # test decim kwarg
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs_dec = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            baseline=(None, 0), decim=4)
        assert_equal(len(w), 1)

    data_dec = epochs_dec.get_data()
    assert_array_equal(data[:, :, epochs_dec._decim_idx], data_dec)

    evoked_dec = epochs_dec.average()
    assert_array_equal(evoked.data[:, epochs_dec._decim_idx], evoked_dec.data)

    n = evoked.data.shape[1]
    n_dec = evoked_dec.data.shape[1]
    n_dec_min = n // 4
    assert_true(n_dec_min <= n_dec <= n_dec_min + 1)
    assert_true(evoked_dec.info['sfreq'] == evoked.info['sfreq'] / 4)

    # test IO
    epochs.save(op.join(tempdir, 'test-epo.fif'))
    epochs_read = read_epochs(op.join(tempdir, 'test-epo.fif'))

    assert_array_almost_equal(epochs_read.get_data(), epochs.get_data())
    assert_array_equal(epochs_read.times, epochs.times)
    assert_array_almost_equal(epochs_read.average().data, evoked.data)
    assert_equal(epochs_read.proj, epochs.proj)
    bmin, bmax = epochs.baseline
    if bmin is None:
        bmin = epochs.times[0]
    if bmax is None:
        bmax = epochs.times[-1]
    baseline = (bmin, bmax)
    assert_array_almost_equal(epochs_read.baseline, baseline)
    assert_array_almost_equal(epochs_read.tmin, epochs.tmin, 2)
    assert_array_almost_equal(epochs_read.tmax, epochs.tmax, 2)
    assert_equal(epochs_read.event_id, epochs.event_id)

    epochs.event_id.pop('1')
    epochs.event_id.update({'a:a': 1})  # test allow for ':' in key
    epochs.save(op.join(tempdir, 'foo-epo.fif'))
    epochs_read2 = read_epochs(op.join(tempdir, 'foo-epo.fif'))
    assert_equal(epochs_read2.event_id, epochs.event_id)

    # add reject here so some of the epochs get dropped
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject)
    epochs.save(op.join(tempdir, 'test-epo.fif'))
    # ensure bad events are not saved
    epochs_read3 = read_epochs(op.join(tempdir, 'test-epo.fif'))
    assert_array_equal(epochs_read3.events, epochs.events)
    data = epochs.get_data()
    assert_true(epochs_read3.events.shape[0] == data.shape[0])

    # test copying loaded one (raw property)
    epochs_read4 = epochs_read3.copy()
    assert_array_almost_equal(epochs_read4.get_data(), data)
    # test equalizing loaded one (drop_log property)
    epochs_read4.equalize_event_counts(epochs.event_id)

    epochs.drop_epochs([1, 2], reason='can we recover orig ID?')
    epochs.save('test-epo.fif')
    epochs_read5 = read_epochs('test-epo.fif')
    assert_array_equal(epochs_read5.selection, epochs.selection)
    assert_array_equal(epochs_read5.drop_log, epochs.drop_log)

    # Test that one can drop channels on read file
    epochs_read5.drop_channels(epochs_read5.ch_names[:1])

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        epochs.save(epochs_badname)
        read_epochs(epochs_badname)
    assert_true(len(w) == 2)


def test_epochs_proj():
    """Test handling projection (apply proj in Raw or in Epochs)
    """
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more
    this_picks = pick_types(raw.info, meg=True, eeg=False, stim=True,
                            eog=True, exclude=exclude)
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    baseline=(None, 0), proj=True)
    assert_true(all(p['active'] is True for p in epochs.info['projs']))
    evoked = epochs.average()
    assert_true(all(p['active'] is True for p in evoked.info['projs']))
    data = epochs.get_data()

    raw_proj = io.Raw(raw_fname, proj=True)
    epochs_no_proj = Epochs(raw_proj, events[:4], event_id, tmin, tmax,
                            picks=this_picks, baseline=(None, 0), proj=False)

    data_no_proj = epochs_no_proj.get_data()
    assert_true(all(p['active'] is True for p in epochs_no_proj.info['projs']))
    evoked_no_proj = epochs_no_proj.average()
    assert_true(all(p['active'] is True for p in evoked_no_proj.info['projs']))
    assert_true(epochs_no_proj.proj is True)  # as projs are active from Raw

    assert_array_almost_equal(data, data_no_proj, decimal=8)

    # make sure we can exclude avg ref
    this_picks = pick_types(raw.info, meg=True, eeg=True, stim=True,
                            eog=True, exclude=exclude)
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    baseline=(None, 0), proj=True, add_eeg_ref=True)
    assert_true(_has_eeg_average_ref_proj(epochs.info['projs']))
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    baseline=(None, 0), proj=True, add_eeg_ref=False)
    assert_true(not _has_eeg_average_ref_proj(epochs.info['projs']))


def test_evoked_arithmetic():
    """Test arithmetic of evoked data
    """
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


def test_evoked_io_from_epochs():
    """Test IO of evoked data made from epochs
    """
    # offset our tmin so we don't get exactly a zero value when decimating
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs = Epochs(raw, events[:4], event_id, tmin + 0.011, tmax,
                        picks=picks, baseline=(None, 0), decim=5)
    assert_true(len(w) == 1)
    evoked = epochs.average()
    evoked.save(op.join(tempdir, 'evoked-ave.fif'))
    evoked2 = read_evokeds(op.join(tempdir, 'evoked-ave.fif'))[0]
    assert_allclose(evoked.data, evoked2.data, rtol=1e-4, atol=1e-20)
    assert_allclose(evoked.times, evoked2.times, rtol=1e-4,
                    atol=1 / evoked.info['sfreq'])

    # now let's do one with negative time
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs = Epochs(raw, events[:4], event_id, 0.1, tmax,
                        picks=picks, baseline=(0.1, 0.2), decim=5)
    evoked = epochs.average()
    evoked.save(op.join(tempdir, 'evoked-ave.fif'))
    evoked2 = read_evokeds(op.join(tempdir, 'evoked-ave.fif'))[0]
    assert_allclose(evoked.data, evoked2.data, rtol=1e-4, atol=1e-20)
    assert_allclose(evoked.times, evoked2.times, rtol=1e-4, atol=1e-20)

    # should be equivalent to a cropped original
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs = Epochs(raw, events[:4], event_id, -0.2, tmax,
                        picks=picks, baseline=(0.1, 0.2), decim=5)
    evoked = epochs.average()
    evoked.crop(0.099, None)
    assert_allclose(evoked.data, evoked2.data, rtol=1e-4, atol=1e-20)
    assert_allclose(evoked.times, evoked2.times, rtol=1e-4, atol=1e-20)


def test_evoked_standard_error():
    """Test calculation and read/write of standard error
    """
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    evoked = [epochs.average(), epochs.standard_error()]
    write_evokeds(op.join(tempdir, 'evoked-ave.fif'), evoked)
    evoked2 = read_evokeds(op.join(tempdir, 'evoked-ave.fif'), [0, 1])
    evoked3 = [read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 'Unknown'),
               read_evokeds(op.join(tempdir, 'evoked-ave.fif'), 'Unknown',
                            kind='standard_error')]
    for evoked_new in [evoked2, evoked3]:
        assert_true(evoked_new[0]._aspect_kind ==
                    FIFF.FIFFV_ASPECT_AVERAGE)
        assert_true(evoked_new[0].kind == 'average')
        assert_true(evoked_new[1]._aspect_kind ==
                    FIFF.FIFFV_ASPECT_STD_ERR)
        assert_true(evoked_new[1].kind == 'standard_error')
        for ave, ave2 in zip(evoked, evoked_new):
            assert_array_almost_equal(ave.data, ave2.data)
            assert_array_almost_equal(ave.times, ave2.times)
            assert_equal(ave.nave, ave2.nave)
            assert_equal(ave._aspect_kind, ave2._aspect_kind)
            assert_equal(ave.kind, ave2.kind)
            assert_equal(ave.last, ave2.last)
            assert_equal(ave.first, ave2.first)


def test_reject_epochs():
    """Test of epochs rejection
    """
    events1 = events[events[:, 2] == event_id]
    epochs = Epochs(raw, events1,
                    event_id, tmin, tmax, baseline=(None, 0),
                    reject=reject, flat=flat)
    assert_raises(RuntimeError, len, epochs)
    n_events = len(epochs.events)
    data = epochs.get_data()
    n_clean_epochs = len(data)
    # Should match
    # mne_process_raw --raw test_raw.fif --projoff \
    #   --saveavetag -ave --ave test.ave --filteroff
    assert_true(n_events > n_clean_epochs)
    assert_true(n_clean_epochs == 3)
    assert_true(epochs.drop_log == [[], [], [], ['MEG 2443'], ['MEG 2443'],
                                    ['MEG 2443'], ['MEG 2443']])

    # Ensure epochs are not dropped based on a bad channel
    raw_2 = raw.copy()
    raw_2.info['bads'] = ['MEG 2443']
    reject_crazy = dict(grad=1000e-15, mag=4e-15, eeg=80e-9, eog=150e-9)
    epochs = Epochs(raw_2, events1, event_id, tmin, tmax, baseline=(None, 0),
                    reject=reject_crazy, flat=flat)
    epochs.drop_bad_epochs()

    assert_true(all(['MEG 2442' in e for e in epochs.drop_log]))
    assert_true(all(['MEG 2443' not in e for e in epochs.drop_log]))

    epochs = Epochs(raw, events1, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, flat=flat,
                    reject_tmin=0., reject_tmax=.1)
    data = epochs.get_data()
    n_clean_epochs = len(data)
    assert_true(n_clean_epochs == 7)
    assert_true(len(epochs) == 7)
    assert_true(epochs.times[epochs._reject_time][0] >= 0.)
    assert_true(epochs.times[epochs._reject_time][-1] <= 0.1)


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
    assert_array_almost_equal(epochs_preload.average().data,
                              epochs.average().data, 18)


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
    c_evoked = read_evokeds(evoked_nf_name, condition=0)
    epochs = Epochs(raw, events, event_id, tmin, tmax,
                    baseline=None, preload=True,
                    reject=None, flat=None)
    evoked = epochs.average()
    sel = pick_channels(c_evoked.ch_names, evoked.ch_names)
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
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)
    data_normal = cp.deepcopy(epochs.get_data())
    times_normal = cp.deepcopy(epochs.times)
    sfreq_normal = epochs.info['sfreq']
    # upsample by 2
    epochs.resample(sfreq_normal * 2, npad=0)
    data_up = cp.deepcopy(epochs.get_data())
    times_up = cp.deepcopy(epochs.times)
    sfreq_up = epochs.info['sfreq']
    # downsamply by 2, which should match
    epochs.resample(sfreq_normal, npad=0)
    data_new = cp.deepcopy(epochs.get_data())
    times_new = cp.deepcopy(epochs.times)
    sfreq_new = epochs.info['sfreq']
    assert_true(data_up.shape[2] == 2 * data_normal.shape[2])
    assert_true(sfreq_up == 2 * sfreq_normal)
    assert_true(sfreq_new == sfreq_normal)
    assert_true(len(times_up) == 2 * len(times_normal))
    assert_array_almost_equal(times_new, times_normal, 10)
    assert_true(data_up.shape[2] == 2 * data_normal.shape[2])
    assert_array_almost_equal(data_new, data_normal, 5)

    # use parallel
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)
    epochs.resample(sfreq_normal * 2, n_jobs=2, npad=0)
    assert_true(np.allclose(data_up, epochs._data, rtol=1e-8, atol=1e-16))


def test_detrend():
    """Test detrending of epochs
    """
    # test first-order
    epochs_1 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                      baseline=None, detrend=1)
    epochs_2 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                      baseline=None, detrend=None)
    data_picks = pick_types(epochs_1.info, meg=True, eeg=True,
                            exclude='bads')
    evoked_1 = epochs_1.average()
    evoked_2 = epochs_2.average()
    evoked_2.detrend(1)
    # Due to roundoff these won't be exactly equal, but they should be close
    assert_true(np.allclose(evoked_1.data, evoked_2.data,
                            rtol=1e-8, atol=1e-20))

    # test zeroth-order case
    for preload in [True, False]:
        epochs_1 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                          baseline=(None, None), preload=preload)
        epochs_2 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                          baseline=None, preload=preload, detrend=0)
        a = epochs_1.get_data()
        b = epochs_2.get_data()
        # All data channels should be almost equal
        assert_true(np.allclose(a[:, data_picks, :], b[:, data_picks, :],
                                rtol=1e-16, atol=1e-20))
        # There are non-M/EEG channels that should not be equal:
        assert_true(not np.allclose(a, b))


def test_bootstrap():
    """Test of bootstrapping of epochs
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)
    epochs2 = bootstrap(epochs, random_state=0)
    assert_true(len(epochs2.events) == len(epochs.events))
    assert_true(epochs._data.shape == epochs2._data.shape)


def test_epochs_copy():
    """Test copy epochs
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)
    copied = epochs.copy()
    assert_array_equal(epochs._data, copied._data)

    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=False,
                    reject=reject, flat=flat)
    copied = epochs.copy()
    data = epochs.get_data()
    copied_data = copied.get_data()
    assert_array_equal(data, copied_data)


def test_iter_evoked():
    """Test the iterator for epochs -> evoked
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))

    for ii, ev in enumerate(epochs.iter_evoked()):
        x = ev.data
        y = epochs.get_data()[ii, :, :]
        assert_array_equal(x, y)


def test_subtract_evoked():
    """Test subtraction of Evoked from Epochs
    """
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))

    # make sure subraction fails if data channels are missing
    assert_raises(ValueError, epochs.subtract_evoked,
                  epochs.average(picks[:5]))

    # do the subraction using the default argument
    epochs.subtract_evoked()

    # apply SSP now
    epochs.apply_proj()

    # use preloading and SSP from the start
    epochs2 = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), preload=True, proj=True)

    evoked = epochs2.average()
    epochs2.subtract_evoked(evoked)

    # this gives the same result
    assert_allclose(epochs.get_data(), epochs2.get_data())

    # if we compute the evoked response after subtracting it we get zero
    zero_evoked = epochs.average()
    data = zero_evoked.data
    assert_array_almost_equal(data, np.zeros_like(data), decimal=20)


@requires_nitime
def test_epochs_to_nitime():
    """Test test_to_nitime
    """
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=reject, flat=flat)

    picks2 = [0, 3]

    epochs_ts = epochs.to_nitime(picks=None, epochs_idx=[0],
                                 collapse=True, copy=True)
    assert_true(epochs_ts.ch_names == epochs.ch_names)

    epochs_ts = epochs.to_nitime(picks=picks2, epochs_idx=None,
                                 collapse=True, copy=True)
    assert_true(epochs_ts.ch_names == [epochs.ch_names[k] for k in picks2])

    epochs_ts = epochs.to_nitime(picks=None, epochs_idx=[0],
                                 collapse=False, copy=False)
    assert_true(epochs_ts.ch_names == epochs.ch_names)

    epochs_ts = epochs.to_nitime(picks=picks2, epochs_idx=None,
                                 collapse=False, copy=False)
    assert_true(epochs_ts.ch_names == [epochs.ch_names[k] for k in picks2])


def test_epoch_eq():
    """Test epoch count equalization and condition combining
    """
    # equalizing epochs objects
    epochs_1 = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    epochs_2 = Epochs(raw, events, event_id_2, tmin, tmax, picks=picks)
    epochs_1.drop_bad_epochs()  # make sure drops are logged
    assert_true(len([l for l in epochs_1.drop_log if not l]) ==
                len(epochs_1.events))
    drop_log1 = epochs_1.drop_log = [[] for _ in range(len(epochs_1.events))]
    drop_log2 = [[] if l == ['EQUALIZED_COUNT'] else l for l in
                 epochs_1.drop_log]
    assert_true(drop_log1 == drop_log2)
    assert_true(len([l for l in epochs_1.drop_log if not l]) ==
                len(epochs_1.events))
    assert_true(epochs_1.events.shape[0] != epochs_2.events.shape[0])
    equalize_epoch_counts([epochs_1, epochs_2], method='mintime')
    assert_true(epochs_1.events.shape[0] == epochs_2.events.shape[0])
    epochs_3 = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    epochs_4 = Epochs(raw, events, event_id_2, tmin, tmax, picks=picks)
    equalize_epoch_counts([epochs_3, epochs_4], method='truncate')
    assert_true(epochs_1.events.shape[0] == epochs_3.events.shape[0])
    assert_true(epochs_3.events.shape[0] == epochs_4.events.shape[0])

    # equalizing conditions
    epochs = Epochs(raw, events, {'a': 1, 'b': 2, 'c': 3, 'd': 4},
                    tmin, tmax, picks=picks, reject=reject)
    epochs.drop_bad_epochs()  # make sure drops are logged
    assert_true(len([l for l in epochs.drop_log if not l]) ==
                len(epochs.events))
    drop_log1 = deepcopy(epochs.drop_log)
    old_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    epochs.equalize_event_counts(['a', 'b'], copy=False)
    # undo the eq logging
    drop_log2 = [[] if l == ['EQUALIZED_COUNT'] else l for l in
                 epochs.drop_log]
    assert_true(drop_log1 == drop_log2)

    assert_true(len([l for l in epochs.drop_log if not l]) ==
                len(epochs.events))
    new_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    assert_true(new_shapes[0] == new_shapes[1])
    assert_true(new_shapes[2] == new_shapes[2])
    assert_true(new_shapes[3] == new_shapes[3])
    # now with two conditions collapsed
    old_shapes = new_shapes
    epochs.equalize_event_counts([['a', 'b'], 'c'], copy=False)
    new_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    assert_true(new_shapes[0] + new_shapes[1] == new_shapes[2])
    assert_true(new_shapes[3] == old_shapes[3])
    assert_raises(KeyError, epochs.equalize_event_counts, [1, 'a'])

    # now let's combine conditions
    old_shapes = new_shapes
    epochs = epochs.equalize_event_counts([['a', 'b'], ['c', 'd']])[0]
    new_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    assert_true(old_shapes[0] + old_shapes[1] == new_shapes[0] + new_shapes[1])
    assert_true(new_shapes[0] + new_shapes[1] == new_shapes[2] + new_shapes[3])
    assert_raises(ValueError, combine_event_ids, epochs, ['a', 'b'],
                  {'ab': 1})

    combine_event_ids(epochs, ['a', 'b'], {'ab': 12}, copy=False)
    caught = 0
    for key in ['a', 'b']:
        try:
            epochs[key]
        except KeyError:
            caught += 1
    assert_raises(Exception, caught == 2)
    assert_true(not np.any(epochs.events[:, 2] == 1))
    assert_true(not np.any(epochs.events[:, 2] == 2))
    epochs = combine_event_ids(epochs, ['c', 'd'], {'cd': 34})
    assert_true(np.all(np.logical_or(epochs.events[:, 2] == 12,
                                     epochs.events[:, 2] == 34)))
    assert_true(epochs['ab'].events.shape[0] == old_shapes[0] + old_shapes[1])
    assert_true(epochs['ab'].events.shape[0] == epochs['cd'].events.shape[0])


def test_access_by_name():
    """Test accessing epochs by event name and on_missing for rare events
    """
    assert_raises(ValueError, Epochs, raw, events, {1: 42, 2: 42}, tmin,
                  tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, {'a': 'spam', 2: 'eggs'},
                  tmin, tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, {'a': 'spam', 2: 'eggs'},
                  tmin, tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, 'foo', tmin, tmax,
                  picks=picks)
    # Test accessing non-existent events (assumes 12345678 does not exist)
    event_id_illegal = dict(aud_l=1, does_not_exist=12345678)
    assert_raises(ValueError, Epochs, raw, events, event_id_illegal,
                  tmin, tmax)
    # Test on_missing
    assert_raises(ValueError, Epochs, raw, events, 1, tmin, tmax,
                  on_missing='foo')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Epochs(raw, events, event_id_illegal, tmin, tmax, on_missing='warning')
        nw = len(w)
        assert_true(1 <= nw <= 2)
        Epochs(raw, events, event_id_illegal, tmin, tmax, on_missing='ignore')
        assert_equal(len(w), nw)
    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks)
    assert_raises(KeyError, epochs.__getitem__, 'bar')

    data = epochs['a'].get_data()
    event_a = events[events[:, 2] == 1]
    assert_true(len(data) == len(event_a))

    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks,
                    preload=True)
    assert_raises(KeyError, epochs.__getitem__, 'bar')
    epochs.save(op.join(tempdir, 'test-epo.fif'))
    epochs2 = read_epochs(op.join(tempdir, 'test-epo.fif'))

    for ep in [epochs, epochs2]:
        data = ep['a'].get_data()
        event_a = events[events[:, 2] == 1]
        assert_true(len(data) == len(event_a))

    assert_array_equal(epochs2['a'].events, epochs['a'].events)

    epochs3 = Epochs(raw, events, {'a': 1, 'b': 2, 'c': 3, 'd': 4},
                     tmin, tmax, picks=picks, preload=True)
    assert_equal(list(sorted(epochs3[('a', 'b')].event_id.values())),
                 [1, 2])
    epochs4 = epochs['a']
    epochs5 = epochs3['a']
    assert_array_equal(epochs4.events, epochs5.events)
    # 20 is our tolerance because epochs are written out as floats
    assert_array_almost_equal(epochs4.get_data(), epochs5.get_data(), 20)
    epochs6 = epochs3[['a', 'b']]
    assert_true(all(np.logical_or(epochs6.events[:, 2] == 1,
                                  epochs6.events[:, 2] == 2)))
    assert_array_equal(epochs.events, epochs6.events)
    assert_array_almost_equal(epochs.get_data(), epochs6.get_data(), 20)


@requires_pandas
def test_as_data_frame():
    """Test epochs Pandas exporter"""
    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks)
    assert_raises(ValueError, epochs.as_data_frame, index=['foo', 'bar'])
    assert_raises(ValueError, epochs.as_data_frame, index='qux')
    assert_raises(ValueError, epochs.as_data_frame, np.arange(400))
    df = epochs.as_data_frame()
    data = np.hstack(epochs.get_data())
    assert_true((df.columns == epochs.ch_names).all())
    assert_array_equal(df.values[:, 0], data[0] * 1e13)
    assert_array_equal(df.values[:, 2], data[2] * 1e15)
    for ind in ['time', ['condition', 'time'], ['condition', 'time', 'epoch']]:
        df = epochs.as_data_frame(index=ind)
        assert_true(df.index.names == ind if isinstance(ind, list) else [ind])
        # test that non-indexed data were present as categorial variables
        df.reset_index().columns[:3] == ['condition', 'epoch', 'time']


def test_epochs_proj_mixin():
    """Test SSP proj methods from ProjMixin class
    """
    for proj in [True, False]:
        epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), proj=proj)

        assert_true(all(p['active'] == proj for p in epochs.info['projs']))

        # test adding / deleting proj
        if proj:
            epochs.get_data()
            assert_true(all(p['active'] == proj for p in epochs.info['projs']))
            assert_raises(ValueError, epochs.add_proj, epochs.info['projs'][0],
                          {'remove_existing': True})
            assert_raises(ValueError, epochs.add_proj, 'spam')
            assert_raises(ValueError, epochs.del_proj, 0)
        else:
            projs = deepcopy(epochs.info['projs'])
            n_proj = len(epochs.info['projs'])
            epochs.del_proj(0)
            assert_true(len(epochs.info['projs']) == n_proj - 1)
            epochs.add_proj(projs, remove_existing=False)
            assert_true(len(epochs.info['projs']) == 2 * n_proj - 1)
            epochs.add_proj(projs, remove_existing=True)
            assert_true(len(epochs.info['projs']) == n_proj)

    # catch no-gos.
    # wrong proj argument
    assert_raises(ValueError, Epochs, raw, events[:4], event_id, tmin, tmax,
                  picks=picks, baseline=(None, 0), proj='crazy')
    # delayed without reject params
    assert_raises(RuntimeError, Epochs, raw, events[:4], event_id, tmin, tmax,
                  picks=picks, baseline=(None, 0), proj='delayed', reject=None)

    for preload in [True, False]:
        epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), proj='delayed', preload=preload,
                        add_eeg_ref=True, verbose=True, reject=reject)
        epochs2 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                         baseline=(None, 0), proj=True, preload=preload,
                         add_eeg_ref=True, reject=reject)
        assert_allclose(epochs.copy().apply_proj().get_data()[0],
                        epochs2.get_data()[0])

        # make sure data output is constant across repeated calls
        # e.g. drop bads
        assert_array_equal(epochs.get_data(), epochs.get_data())
        assert_array_equal(epochs2.get_data(), epochs2.get_data())

    # test epochs.next calls
    data = epochs.get_data().copy()
    data2 = np.array([e for e in epochs])
    assert_array_equal(data, data2)

    # cross application from processing stream 1 to 2
    epochs.apply_proj()
    assert_array_equal(epochs._projector, epochs2._projector)
    assert_allclose(epochs._data, epochs2.get_data())

    # test mixin against manual application
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=None, proj=False, add_eeg_ref=True)
    data = epochs.get_data().copy()
    epochs.apply_proj()
    assert_allclose(np.dot(epochs._projector, data[0]), epochs._data[0])


def test_drop_epochs():
    """Test dropping of epochs.
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
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
    n_events = len(epochs.events)
    epochs.drop_epochs([2, 4], reason='d')
    assert_equal(epochs.drop_log_stats(), 2. / n_events * 100)
    assert_equal(len(epochs.drop_log), len(events))
    assert_equal([epochs.drop_log[k]
                  for k in selection[[2, 4]]], [['d'], ['d']])
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
        picks_contains = pick_types(raw.info, meg=meg, eeg=eeg)
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
    # here without picks to get additional coverage
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=None,
                    baseline=(None, 0))
    drop_ch = epochs.ch_names[:3]
    ch_names = epochs.ch_names[3:]

    ch_names_orig = epochs.ch_names
    dummy = epochs.drop_channels(drop_ch, copy=True)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, epochs.ch_names)
    assert_equal(len(ch_names_orig), epochs.get_data().shape[1])

    epochs.drop_channels(drop_ch)
    assert_equal(ch_names, epochs.ch_names)
    assert_equal(len(ch_names), epochs.get_data().shape[1])


def test_pick_channels_mixin():
    """Test channel-picking functionality
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    ch_names = epochs.ch_names[:3]

    ch_names_orig = epochs.ch_names
    dummy = epochs.pick_channels(ch_names, copy=True)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, epochs.ch_names)
    assert_equal(len(ch_names_orig), epochs.get_data().shape[1])

    epochs.pick_channels(ch_names)
    assert_equal(ch_names, epochs.ch_names)
    assert_equal(len(ch_names), epochs.get_data().shape[1])


def test_equalize_channels():
    """Test equalization of channels
    """
    epochs1 = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), proj=False)
    epochs2 = epochs1.copy()
    ch_names = epochs1.ch_names[2:]
    epochs1.drop_channels(epochs1.ch_names[:1])
    epochs2.drop_channels(epochs2.ch_names[1:2])
    my_comparison = [epochs1, epochs2]
    equalize_channels(my_comparison)
    for e in my_comparison:
        assert_equal(ch_names, e.ch_names)


def test_illegal_event_id():
    """Test handling of invalid events ids"""
    event_id_illegal = dict(aud_l=1, does_not_exist=12345678)

    assert_raises(ValueError, Epochs, raw, events, event_id_illegal, tmin,
                  tmax, picks=picks, baseline=(None, 0), proj=False)


def test_add_channels_epochs():
    """Test adding channels"""

    def make_epochs(picks):
        return Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0),
                      reject=None, preload=True, proj=False, picks=picks)

    picks = pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    picks_meg = pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')

    epochs = make_epochs(picks=picks)
    epochs_meg = make_epochs(picks=picks_meg)
    epochs_eeg = make_epochs(picks=picks_eeg)

    epochs2 = add_channels_epochs([epochs_meg, epochs_eeg])

    assert_equal(len(epochs.info['projs']), len(epochs2.info['projs']))
    assert_equal(len(epochs.info.keys()), len(epochs2.info.keys()))

    data1 = epochs.get_data()
    data2 = epochs2.get_data()
    data3 = np.concatenate([e.get_data() for e in
                            [epochs_meg, epochs_eeg]], axis=1)
    assert_array_equal(data1.shape, data2.shape)
    assert_array_equal(data1, data3)  # XXX unrelated bug? this crashes
                                      # when proj == True
    assert_array_equal(data1, data2)

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['meas_date'] += 10
    add_channels_epochs([epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs2.info['filename'] = epochs2.info['filename'].upper()
    epochs2 = add_channels_epochs([epochs_meg, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.events[3, 2] -= 1
    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg, epochs_eeg[:2]])

    epochs_meg.info['chs'].pop(0)
    assert_raises(RuntimeError, add_channels_epochs,
                  [epochs_meg, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['sfreq'] = None
    assert_raises(RuntimeError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['sfreq'] += 10
    assert_raises(RuntimeError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['ch_names'][1] = epochs_meg2.info['ch_names'][0]
    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['dev_head_t']['to'] += 1
    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['dev_head_t']['to'] += 1
    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['expimenter'] = 'foo'
    assert_raises(RuntimeError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.preload = False
    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.tmin += 0.4
    assert_raises(NotImplementedError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.tmin += 0.5
    assert_raises(NotImplementedError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.baseline = None
    assert_raises(NotImplementedError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.event_id['b'] = 2
    assert_raises(NotImplementedError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])


def test_array_epochs():
    """Test creating epochs from array
    """

    # creating
    rng = np.random.RandomState(42)
    data = rng.random_sample((10, 20, 300))
    sfreq = 1e3
    ch_names = ['EEG %03d' % (i + 1) for i in range(20)]
    types = ['eeg'] * 20
    info = create_info(ch_names, sfreq, types)
    events = np.c_[np.arange(1, 600, 60),
                   np.zeros(10),
                   [1, 2] * 5]
    event_id = {'a': 1, 'b': 2}
    epochs = EpochsArray(data, info, events=events, event_id=event_id,
                         tmin=-.2)

    # saving
    temp_fname = op.join(tempdir, 'test-epo.fif')
    epochs.save(temp_fname)
    epochs2 = read_epochs(temp_fname)
    data2 = epochs2.get_data()
    assert_allclose(data, data2)
    assert_allclose(epochs.times, epochs2.times)
    assert_equal(epochs.event_id, epochs2.event_id)
    assert_array_equal(epochs.events, epochs2.events)

    # plotting
    import matplotlib
    matplotlib.use('Agg')  # for testing don't use X server
    epochs[0].plot()

    # indexing
    assert_array_equal(np.unique(epochs['a'].events[:, 2]), np.array([1]))
    assert_equal(len(epochs[:2]), 2)
    data[0, 5, 150] = 3000
    data[1, :, :] = 0
    data[2, 5, 210] = 3000
    data[3, 5, 260] = 0
    epochs = EpochsArray(data, info, events=events, event_id=event_id,
                         tmin=0, reject=dict(eeg=1000), flat=dict(eeg=1e-1),
                         reject_tmin=0.1, reject_tmax=0.2)
    assert_equal(len(epochs), len(events) - 2)
    assert_equal(epochs.drop_log[0], ['EEG 006'])
    assert_equal(len(events), len(epochs.selection))
