# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import copy as cp
import warnings

from mne import fiff, Epochs, read_events, pick_events, read_epochs
from mne.epochs import bootstrap, equalize_epoch_counts, combine_event_ids
from mne.utils import _TempDir, requires_pandas, requires_nitime
from mne.fiff import read_evoked

base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
raw = fiff.Raw(raw_fname)
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=True,
                        ecg=True, eog=True, include=['STI 014'])

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)

tempdir = _TempDir()


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
                                eog=True)
    epochs.drop_picks(eog_picks)
    data_no_eog = epochs.get_data()
    assert_true(data.shape[1] == (data_no_eog.shape[1] + len(eog_picks)))

    # test decim kwarg
    with warnings.catch_warnings(record=True) as w:
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
    epochs.event_id.update({'a': 1})
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

    raw_proj = fiff.Raw(raw_fname, proj_active=True)
    epochs_no_proj = Epochs(raw_proj, events[:4], event_id, tmin, tmax,
                            picks=this_picks, baseline=(None, 0), proj=False)
    epochs_no_proj.average()
    data_no_proj = epochs_no_proj.get_data()
    assert_array_almost_equal(data, data_no_proj, decimal=8)


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


def test_evoked_standard_error():
    """Test calculation and read/write of standard error
    """
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    evoked = [epochs.average(), epochs.standard_error()]
    fiff.write_evoked(op.join(tempdir, 'evoked.fif'), evoked)
    evoked2 = read_evoked(op.join(tempdir, 'evoked.fif'), [0, 1])
    evoked3 = [read_evoked(op.join(tempdir, 'evoked.fif'), 'Unknown'),
               read_evoked(op.join(tempdir, 'evoked.fif'), 'Unknown',
                           kind='standard_error')]
    for evoked_new in [evoked2, evoked3]:
        assert_true(evoked_new[0]._aspect_kind == fiff.FIFF.FIFFV_ASPECT_AVERAGE)
        assert_true(evoked_new[0].kind == 'average')
        assert_true(evoked_new[1]._aspect_kind == fiff.FIFF.FIFFV_ASPECT_STD_ERR)
        assert_true(evoked_new[1].kind == 'standard_error')
        for ave, ave2 in zip(evoked, evoked_new):
            assert_array_almost_equal(ave.data, ave2.data)
            assert_array_almost_equal(ave.times, ave2.times)
            assert_equal(ave.nave, ave2.nave)
            assert_equal(ave._aspect_kind, ave2._aspect_kind)
            print ave._aspect_kind
            print ave2._aspect_kind
            assert_equal(ave.kind, ave2.kind)
            assert_equal(ave.last, ave2.last)
            assert_equal(ave.first, ave2.first)


def test_reject_epochs():
    """Test of epochs rejection
    """
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, flat=flat)
    n_events = len(epochs.events)
    data = epochs.get_data()
    n_clean_epochs = len(data)
    # Should match
    # mne_process_raw --raw test_raw.fif --projoff \
    #   --saveavetag -ave --ave test.ave --filteroff
    assert_true(n_events > n_clean_epochs)
    assert_true(n_clean_epochs == 3)
    assert_true(epochs.drop_log == [[], [], [], ['MEG 2443'],
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
    assert_array_equal(data_up, epochs._data)


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
                    tmin, tmax, picks=picks)
    old_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    epochs.equalize_event_counts(['a', 'b'], copy=False)
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
    assert_raises(caught == 2)
    assert_true(not np.any(epochs.events[:, 2] == 1))
    assert_true(not np.any(epochs.events[:, 2] == 2))
    epochs = combine_event_ids(epochs, ['c', 'd'], {'cd': 34})
    assert_true(np.all(np.logical_or(epochs.events[:, 2] == 12,
                                     epochs.events[:, 2] == 34)))
    assert_true(epochs['ab'].events.shape[0] == old_shapes[0] + old_shapes[1])
    assert_true(epochs['ab'].events.shape[0] == epochs['cd'].events.shape[0])


def test_access_by_name():
    """Test accessing epochs by event name
    """
    assert_raises(ValueError, Epochs, raw, events, {1: 42, 2: 42}, tmin,
                  tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, {'a': 'spam', 2: 'eggs'},
                  tmin, tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, {'a': 'spam', 2: 'eggs'},
                  tmin, tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, 'foo', tmin, tmax,
                  picks=picks)
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
    """Test Pandas exporter"""
    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks)
    assert_raises(ValueError, epochs.as_data_frame, index=['foo', 'bar'])
    assert_raises(ValueError, epochs.as_data_frame, index='qux')
    assert_raises(ValueError, epochs.as_data_frame, np.arange(400))
    df = epochs.as_data_frame()
    data = np.hstack(epochs.get_data())
    assert_true((df.columns[1:] == epochs.ch_names).all())
    assert_true(df.index.names == ['epoch', 'time'])
    assert_array_equal(df.values[:, 1], data[0] * 1e13)
    assert_array_equal(df.values[:, 3], data[2] * 1e15)
