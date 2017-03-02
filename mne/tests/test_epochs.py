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
from scipy import fftpack
import matplotlib

from mne import (Epochs, Annotations, read_events, pick_events, read_epochs,
                 equalize_channels, pick_types, pick_channels, read_evokeds,
                 write_evokeds, create_info, make_fixed_length_events,
                 combine_evoked)
from mne.baseline import rescale
from mne.preprocessing import maxwell_filter
from mne.epochs import (
    bootstrap, equalize_epoch_counts, combine_event_ids, add_channels_epochs,
    EpochsArray, concatenate_epochs, BaseEpochs, average_movements)
from mne.utils import (_TempDir, requires_pandas, slow_test,
                       run_tests_if_main, requires_version)
from mne.chpi import read_head_pos, head_pos_to_trans_rot_t

from mne.io import RawArray, read_raw_fif
from mne.io.proj import _has_eeg_average_ref_proj
from mne.event import merge_events
from mne.io.constants import FIFF
from mne.externals.six import text_type
from mne.externals.six.moves import zip, cPickle as pickle
from mne.datasets import testing
from mne.tests.common import assert_meg_snr, assert_naming

matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = testing.data_path(download=False)
fname_raw_move = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
fname_raw_movecomp_sss = op.join(
    data_path, 'SSS', 'test_move_anon_movecomp_raw_sss.fif')
fname_raw_move_pos = op.join(data_path, 'SSS', 'test_move_anon_raw.pos')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = np.int64(2)  # to test non Python int types
rng = np.random.RandomState(42)


def _get_data(preload=False):
    """Get data."""
    raw = read_raw_fif(raw_fname, preload=preload)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, eeg=True, stim=True,
                       ecg=True, eog=True, include=['STI 014'],
                       exclude='bads')
    return raw, events, picks

reject = dict(grad=1000e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
flat = dict(grad=1e-15, mag=1e-15)


def test_hierarchical():
    """Test hierarchical access."""
    raw, events, picks = _get_data()
    event_id = {'a/1': 1, 'a/2': 2, 'b/1': 3, 'b/2': 4}
    epochs = Epochs(raw, events, event_id, preload=True)
    epochs_a1 = epochs['a/1']
    epochs_a2 = epochs['a/2']
    epochs_b1 = epochs['b/1']
    epochs_b2 = epochs['b/2']
    epochs_a = epochs['a']
    assert_equal(len(epochs_a), len(epochs_a1) + len(epochs_a2))
    epochs_b = epochs['b']
    assert_equal(len(epochs_b), len(epochs_b1) + len(epochs_b2))
    epochs_1 = epochs['1']
    assert_equal(len(epochs_1), len(epochs_a1) + len(epochs_b1))
    epochs_2 = epochs['2']
    assert_equal(len(epochs_2), len(epochs_a2) + len(epochs_b2))
    epochs_all = epochs[('1', '2')]
    assert_equal(len(epochs), len(epochs_all))
    assert_array_equal(epochs.get_data(), epochs_all.get_data())


@slow_test
@testing.requires_testing_data
def test_average_movements():
    """Test movement averaging algorithm."""
    # usable data
    crop = 0., 10.
    origin = (0., 0., 0.04)
    raw = read_raw_fif(fname_raw_move, allow_maxshield='yes')
    raw.info['bads'] += ['MEG2443']  # mark some bad MEG channel
    raw.crop(*crop).load_data()
    raw.filter(None, 20)
    events = make_fixed_length_events(raw, event_id)
    picks = pick_types(raw.info, meg=True, eeg=True, stim=True,
                       ecg=True, eog=True, exclude=())
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks, proj=False,
                    preload=True)
    epochs_proj = Epochs(raw, events[:1], event_id, tmin, tmax, picks=picks,
                         proj=True, preload=True)
    raw_sss_stat = maxwell_filter(raw, origin=origin, regularize=None,
                                  bad_condition='ignore')
    del raw
    epochs_sss_stat = Epochs(raw_sss_stat, events, event_id, tmin, tmax,
                             picks=picks, proj=False)
    evoked_sss_stat = epochs_sss_stat.average()
    del raw_sss_stat, epochs_sss_stat
    head_pos = read_head_pos(fname_raw_move_pos)
    trans = epochs.info['dev_head_t']['trans']
    head_pos_stat = (np.array([trans[:3, 3]]),
                     np.array([trans[:3, :3]]),
                     np.array([0.]))

    # SSS-based
    assert_raises(TypeError, average_movements, epochs, None)
    evoked_move_non = average_movements(epochs, head_pos=head_pos,
                                        weight_all=False, origin=origin)
    evoked_move_all = average_movements(epochs, head_pos=head_pos,
                                        weight_all=True, origin=origin)
    evoked_stat_all = average_movements(epochs, head_pos=head_pos_stat,
                                        weight_all=True, origin=origin)
    evoked_std = epochs.average()
    for ev in (evoked_move_non, evoked_move_all, evoked_stat_all):
        assert_equal(ev.nave, evoked_std.nave)
        assert_equal(len(ev.info['bads']), 0)
    # substantial changes to MEG data
    for ev in (evoked_move_non, evoked_stat_all):
        assert_meg_snr(ev, evoked_std, 0., 0.1)
        assert_raises(AssertionError, assert_meg_snr,
                      ev, evoked_std, 1., 1.)
    meg_picks = pick_types(evoked_std.info, meg=True, exclude=())
    assert_allclose(evoked_move_non.data[meg_picks],
                    evoked_move_all.data[meg_picks], atol=1e-20)
    # compare to averaged movecomp version (should be fairly similar)
    raw_sss = read_raw_fif(fname_raw_movecomp_sss)
    raw_sss.crop(*crop).load_data()
    raw_sss.filter(None, 20)
    picks_sss = pick_types(raw_sss.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, exclude=())
    assert_array_equal(picks, picks_sss)
    epochs_sss = Epochs(raw_sss, events, event_id, tmin, tmax,
                        picks=picks_sss, proj=False)
    evoked_sss = epochs_sss.average()
    assert_equal(evoked_std.nave, evoked_sss.nave)
    # this should break the non-MEG channels
    assert_raises(AssertionError, assert_meg_snr,
                  evoked_sss, evoked_move_all, 0., 0.)
    assert_meg_snr(evoked_sss, evoked_move_non, 0.02, 2.6)
    assert_meg_snr(evoked_sss, evoked_stat_all, 0.05, 3.2)
    # these should be close to numerical precision
    assert_allclose(evoked_sss_stat.data, evoked_stat_all.data, atol=1e-20)

    # pos[0] > epochs.events[0] uses dev_head_t, so make it equivalent
    destination = deepcopy(epochs.info['dev_head_t'])
    x = head_pos_to_trans_rot_t(head_pos[1])
    epochs.info['dev_head_t']['trans'][:3, :3] = x[1]
    epochs.info['dev_head_t']['trans'][:3, 3] = x[0]
    assert_raises(AssertionError, assert_allclose,
                  epochs.info['dev_head_t']['trans'],
                  destination['trans'])
    evoked_miss = average_movements(epochs, head_pos=head_pos[2:],
                                    origin=origin, destination=destination)
    assert_allclose(evoked_miss.data, evoked_move_all.data,
                    atol=1e-20)
    assert_allclose(evoked_miss.info['dev_head_t']['trans'],
                    destination['trans'])

    # degenerate cases
    destination['to'] = destination['from']  # bad dest
    assert_raises(RuntimeError, average_movements, epochs, head_pos,
                  origin=origin, destination=destination)
    assert_raises(TypeError, average_movements, 'foo', head_pos=head_pos)
    assert_raises(RuntimeError, average_movements, epochs_proj,
                  head_pos=head_pos)  # prj


def test_reject():
    """Test epochs rejection."""
    raw, events, picks = _get_data()
    # cull the list just to contain the relevant event
    events = events[events[:, 2] == event_id, :]
    selection = np.arange(3)
    drop_log = [[]] * 3 + [['MEG 2443']] * 4
    assert_raises(TypeError, pick_types, raw)
    picks_meg = pick_types(raw.info, meg=True, eeg=False)
    assert_raises(TypeError, Epochs, raw, events, event_id, tmin, tmax,
                  picks=picks, preload=False, reject='foo')
    assert_raises(ValueError, Epochs, raw, events, event_id, tmin, tmax,
                  picks=picks_meg, preload=False, reject=dict(eeg=1.))
    # this one is okay because it's not actually requesting rejection
    Epochs(raw, events, event_id, tmin, tmax, picks=picks_meg,
           preload=False, reject=dict(eeg=np.inf))
    for val in (None, -1):  # protect against older MNE-C types
        for kwarg in ('reject', 'flat'):
            assert_raises(ValueError, Epochs, raw, events, event_id,
                          tmin, tmax, picks=picks_meg, preload=False,
                          **{kwarg: dict(grad=val)})
    assert_raises(KeyError, Epochs, raw, events, event_id, tmin, tmax,
                  picks=picks, preload=False, reject=dict(foo=1.))

    data_7 = dict()
    keep_idx = [0, 1, 2]
    for preload in (True, False):
        for proj in (True, False, 'delayed'):
            # no rejection
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            preload=preload)
            assert_raises(ValueError, epochs.drop_bad, reject='foo')
            epochs.drop_bad()
            assert_equal(len(epochs), len(events))
            assert_array_equal(epochs.selection, np.arange(len(events)))
            assert_array_equal(epochs.drop_log, [[]] * 7)
            if proj not in data_7:
                data_7[proj] = epochs.get_data()
            assert_array_equal(epochs.get_data(), data_7[proj])

            # with rejection
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            reject=reject, preload=preload)
            epochs.drop_bad()
            assert_equal(len(epochs), len(events) - 4)
            assert_array_equal(epochs.selection, selection)
            assert_array_equal(epochs.drop_log, drop_log)
            assert_array_equal(epochs.get_data(), data_7[proj][keep_idx])

            # rejection post-hoc
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            preload=preload)
            epochs.drop_bad()
            assert_equal(len(epochs), len(events))
            assert_array_equal(epochs.get_data(), data_7[proj])
            epochs.drop_bad(reject)
            assert_equal(len(epochs), len(events) - 4)
            assert_equal(len(epochs), len(epochs.get_data()))
            assert_array_equal(epochs.selection, selection)
            assert_array_equal(epochs.drop_log, drop_log)
            assert_array_equal(epochs.get_data(), data_7[proj][keep_idx])

            # rejection twice
            reject_part = dict(grad=1100e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            reject=reject_part, preload=preload)
            epochs.drop_bad()
            assert_equal(len(epochs), len(events) - 1)
            epochs.drop_bad(reject)
            assert_equal(len(epochs), len(events) - 4)
            assert_array_equal(epochs.selection, selection)
            assert_array_equal(epochs.drop_log, drop_log)
            assert_array_equal(epochs.get_data(), data_7[proj][keep_idx])

            # ensure that thresholds must become more stringent, not less
            assert_raises(ValueError, epochs.drop_bad, reject_part)
            assert_equal(len(epochs), len(events) - 4)
            assert_array_equal(epochs.get_data(), data_7[proj][keep_idx])
            epochs.drop_bad(flat=dict(mag=1.))
            assert_equal(len(epochs), 0)
            assert_raises(ValueError, epochs.drop_bad,
                          flat=dict(mag=0.))

            # rejection of subset of trials (ensure array ownership)
            reject_part = dict(grad=1100e-12, mag=4e-12, eeg=80e-6, eog=150e-6)
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            reject=None, preload=preload)
            epochs = epochs[:-1]
            epochs.drop_bad(reject=reject)
            assert_equal(len(epochs), len(events) - 4)
            assert_array_equal(epochs.get_data(), data_7[proj][keep_idx])

        # rejection on annotations
        sfreq = raw.info['sfreq']
        onsets = [(event[0] - raw.first_samp) / sfreq for event in
                  events[::2][:3]]
        onsets[0] = onsets[0] + tmin - 0.499  # tmin < 0
        onsets[1] = onsets[1] + tmax - 0.001
        first_time = (raw.info['meas_date'][0] + raw.info['meas_date'][1] *
                      0.000001 + raw.first_samp / sfreq)
        for orig_time in [None, first_time]:
            raw.annotations = Annotations(onsets, [0.5, 0.5, 0.5], 'BAD',
                                          orig_time)
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=[0],
                            reject=None, preload=preload)
            epochs.drop_bad()
            assert_equal(len(events) - 3, len(epochs.events))
            assert_equal(epochs.drop_log[0][0], 'BAD')
            assert_equal(epochs.drop_log[2][0], 'BAD')
            assert_equal(epochs.drop_log[4][0], 'BAD')
        raw.annotations = None


def test_decim():
    """Test epochs decimation."""
    # First with EpochsArray
    dec_1, dec_2 = 2, 3
    decim = dec_1 * dec_2
    n_epochs, n_channels, n_times = 5, 10, 20
    sfreq = 1000.
    sfreq_new = sfreq / decim
    data = rng.randn(n_epochs, n_channels, n_times)
    events = np.array([np.arange(n_epochs), [0] * n_epochs, [1] * n_epochs]).T
    info = create_info(n_channels, sfreq, 'eeg')
    info['lowpass'] = sfreq_new / float(decim)
    epochs = EpochsArray(data, info, events)
    data_epochs = epochs.copy().decimate(decim).get_data()
    data_epochs_2 = epochs.copy().decimate(decim, offset=1).get_data()
    data_epochs_3 = epochs.decimate(dec_1).decimate(dec_2).get_data()
    assert_array_equal(data_epochs, data[:, :, ::decim])
    assert_array_equal(data_epochs_2, data[:, :, 1::decim])
    assert_array_equal(data_epochs, data_epochs_3)

    # Now let's do it with some real data
    raw, events, picks = _get_data()
    events = events[events[:, 2] == 1][:2]
    raw.load_data().pick_channels([raw.ch_names[pick] for pick in picks[::30]])
    raw.info.normalize_proj()
    del picks
    sfreq_new = raw.info['sfreq'] / decim
    raw.info['lowpass'] = sfreq_new / 12.  # suppress aliasing warnings
    assert_raises(ValueError, epochs.decimate, -1)
    assert_raises(ValueError, epochs.decimate, 2, offset=-1)
    assert_raises(ValueError, epochs.decimate, 2, offset=2)
    for this_offset in range(decim):
        epochs = Epochs(raw, events, event_id,
                        tmin=-this_offset / raw.info['sfreq'], tmax=tmax)
        idx_offsets = np.arange(decim) + this_offset
        for offset, idx_offset in zip(np.arange(decim), idx_offsets):
            expected_times = epochs.times[idx_offset::decim]
            expected_data = epochs.get_data()[:, :, idx_offset::decim]
            must_have = offset / float(epochs.info['sfreq'])
            assert_true(np.isclose(must_have, expected_times).any())
            ep_decim = epochs.copy().decimate(decim, offset)
            assert_true(np.isclose(must_have, ep_decim.times).any())
            assert_allclose(ep_decim.times, expected_times)
            assert_allclose(ep_decim.get_data(), expected_data)
            assert_equal(ep_decim.info['sfreq'], sfreq_new)

    # More complex cases
    epochs = Epochs(raw, events, event_id, tmin, tmax)
    expected_data = epochs.get_data()[:, :, ::decim]
    expected_times = epochs.times[::decim]
    for preload in (True, False):
        # at init
        epochs = Epochs(raw, events, event_id, tmin, tmax, decim=decim,
                        preload=preload)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)

        # split between init and afterward
        epochs = Epochs(raw, events, event_id, tmin, tmax, decim=dec_1,
                        preload=preload).decimate(dec_2)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)
        epochs = Epochs(raw, events, event_id, tmin, tmax, decim=dec_2,
                        preload=preload).decimate(dec_1)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)

        # split between init and afterward, with preload in between
        epochs = Epochs(raw, events, event_id, tmin, tmax, decim=dec_1,
                        preload=preload)
        epochs.load_data()
        epochs = epochs.decimate(dec_2)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)
        epochs = Epochs(raw, events, event_id, tmin, tmax, decim=dec_2,
                        preload=preload)
        epochs.load_data()
        epochs = epochs.decimate(dec_1)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)

        # decimate afterward
        epochs = Epochs(raw, events, event_id, tmin, tmax,
                        preload=preload).decimate(decim)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)

        # decimate afterward, with preload in between
        epochs = Epochs(raw, events, event_id, tmin, tmax, preload=preload)
        epochs.load_data()
        epochs.decimate(decim)
        assert_allclose(epochs.get_data(), expected_data)
        assert_allclose(epochs.get_data(), expected_data)
        assert_equal(epochs.info['sfreq'], sfreq_new)
        assert_array_equal(epochs.times, expected_times)


def test_base_epochs():
    """Test base epochs class."""
    raw = _get_data()[0]
    epochs = BaseEpochs(raw.info, None, np.ones((1, 3), int),
                        event_id, tmin, tmax)
    assert_raises(NotImplementedError, epochs.get_data)
    # events with non integers
    assert_raises(ValueError, BaseEpochs, raw.info, None,
                  np.ones((1, 3), float), event_id, tmin, tmax)
    assert_raises(ValueError, BaseEpochs, raw.info, None,
                  np.ones((1, 3, 2), int), event_id, tmin, tmax)


@requires_version('scipy', '0.14')
def test_savgol_filter():
    """Test savgol filtering."""
    h_freq = 10.
    raw, events = _get_data()[:2]
    epochs = Epochs(raw, events, event_id, tmin, tmax)
    assert_raises(RuntimeError, epochs.savgol_filter, 10.)
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True)
    freqs = fftpack.fftfreq(len(epochs.times), 1. / epochs.info['sfreq'])
    data = np.abs(fftpack.fft(epochs.get_data()))
    match_mask = np.logical_and(freqs >= 0, freqs <= h_freq / 2.)
    mismatch_mask = np.logical_and(freqs >= h_freq * 2, freqs < 50.)
    epochs.savgol_filter(h_freq)
    data_filt = np.abs(fftpack.fft(epochs.get_data()))
    # decent in pass-band
    assert_allclose(np.mean(data[:, :, match_mask], 0),
                    np.mean(data_filt[:, :, match_mask], 0),
                    rtol=1e-4, atol=1e-2)
    # suppression in stop-band
    assert_true(np.mean(data[:, :, mismatch_mask]) >
                np.mean(data_filt[:, :, mismatch_mask]) * 5)


def test_epochs_hash():
    """Test epoch hashing."""
    raw, events = _get_data()[:2]
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
    """Test event order."""
    raw, events = _get_data()[:2]
    events2 = events.copy()
    rng.shuffle(events2)
    for ii, eve in enumerate([events, events2]):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Epochs(raw, eve, event_id, tmin, tmax,
                   reject=reject, flat=flat)
            assert_equal(len(w), ii)
            if ii > 0:
                assert_true('chronologically' in '%s' % w[-1].message)
    # Duplicate events should be an error...
    events2 = events[[0, 0]]
    events2[:, 2] = [1, 2]
    assert_raises(RuntimeError, Epochs, raw, events2, event_id=None)
    # But only if duplicates are actually used by event_id
    assert_equal(len(Epochs(raw, events2, event_id=dict(a=1), preload=True)),
                 1)


def test_epochs_bad_baseline():
    """Test Epochs initialization with bad baseline parameters."""
    raw, events = _get_data()[:2]
    assert_raises(ValueError, Epochs, raw, events, None, -0.1, 0.3, (-0.2, 0))
    assert_raises(ValueError, Epochs, raw, events, None, -0.1, 0.3, (0, 0.4))
    assert_raises(ValueError, Epochs, raw, events, None, -0.1, 0.3, (0.1, 0))
    assert_raises(ValueError, Epochs, raw, events, None, 0.1, 0.3, (None, 0))
    assert_raises(ValueError, Epochs, raw, events, None, -0.3, -0.1, (0, None))
    epochs = Epochs(raw, events, None, 0.1, 0.3, baseline=None)
    assert_raises(RuntimeError, epochs.apply_baseline, (0.1, 0.2))
    epochs.load_data()
    assert_raises(ValueError, epochs.apply_baseline, (None, 0))
    assert_raises(ValueError, epochs.apply_baseline, (0, None))
    # put some rescale options here, too
    data = np.arange(100, dtype=float)
    assert_raises(ValueError, rescale, data, times=data, baseline=(-2, -1))
    rescale(data.copy(), times=data, baseline=(2, 2))  # ok
    assert_raises(ValueError, rescale, data, times=data, baseline=(2, 1))
    assert_raises(ValueError, rescale, data, times=data, baseline=(100, 101))


def test_epoch_combine_ids():
    """Test combining event ids in epochs compared to events."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, {'a': 1, 'b': 2, 'c': 3,
                                  'd': 4, 'e': 5, 'f': 32},
                    tmin, tmax, picks=picks, preload=False)
    events_new = merge_events(events, [1, 2], 12)
    epochs_new = combine_event_ids(epochs, ['a', 'b'], {'ab': 12})
    assert_equal(epochs_new['ab'].name, 'ab')
    assert_array_equal(events_new, epochs_new.events)
    # should probably add test + functionality for non-replacement XXX


def test_epoch_multi_ids():
    """Test epoch selection via multiple/partial keys."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, {'a/b/a': 1, 'a/b/b': 2, 'a/c': 3,
                                  'b/d': 4, 'a_b': 5},
                    tmin, tmax, picks=picks, preload=False)
    epochs_regular = epochs['a/b']
    epochs_reverse = epochs['b/a']
    epochs_multi = epochs[['a/b/a', 'a/b/b']]
    assert_array_equal(epochs_multi.events, epochs_regular.events)
    assert_array_equal(epochs_reverse.events, epochs_regular.events)
    assert_allclose(epochs_multi.get_data(), epochs_regular.get_data())
    assert_allclose(epochs_reverse.get_data(), epochs_regular.get_data())


def test_read_epochs_bad_events():
    """Test epochs when events are at the beginning or the end of the file."""
    raw, events, picks = _get_data()
    # Event at the beginning
    epochs = Epochs(raw, np.array([[raw.first_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks)
    with warnings.catch_warnings(record=True):
        evoked = epochs.average()

    epochs = Epochs(raw, np.array([[raw.first_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks)
    assert_true(repr(epochs))  # test repr
    epochs.drop_bad()
    assert_true(repr(epochs))
    with warnings.catch_warnings(record=True):
        evoked = epochs.average()

    # Event at the end
    epochs = Epochs(raw, np.array([[raw.last_samp, 0, event_id]]),
                    event_id, tmin, tmax, picks=picks)

    with warnings.catch_warnings(record=True):
        evoked = epochs.average()
        assert evoked
    warnings.resetwarnings()


@slow_test
def test_read_write_epochs():
    """Test epochs from raw files with IO as fif file."""
    raw, events, picks = _get_data(preload=True)
    tempdir = _TempDir()
    temp_fname = op.join(tempdir, 'test-epo.fif')
    temp_fname_no_bl = op.join(tempdir, 'test_no_bl-epo.fif')
    baseline = (None, 0)
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, preload=True)
    epochs_orig = epochs.copy()
    epochs_no_bl = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                          baseline=None, preload=True)
    assert_true(epochs_no_bl.baseline is None)
    evoked = epochs.average()
    data = epochs.get_data()

    # Bad tmin/tmax parameters
    assert_raises(ValueError, Epochs, raw, events, event_id, tmax, tmin,
                  baseline=None)

    epochs_no_id = Epochs(raw, pick_events(events, include=event_id),
                          None, tmin, tmax, picks=picks)
    assert_array_equal(data, epochs_no_id.get_data())

    eog_picks = pick_types(raw.info, meg=False, eeg=False, stim=False,
                           eog=True, exclude='bads')
    eog_ch_names = [raw.ch_names[k] for k in eog_picks]
    epochs.drop_channels(eog_ch_names)
    assert_true(len(epochs.info['chs']) == len(epochs.ch_names) ==
                epochs.get_data().shape[1])
    data_no_eog = epochs.get_data()
    assert_true(data.shape[1] == (data_no_eog.shape[1] + len(eog_picks)))

    # test decim kwarg
    with warnings.catch_warnings(record=True) as w:
        # decim with lowpass
        warnings.simplefilter('always')
        epochs_dec = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            decim=2)
        assert_equal(len(w), 1)

        # decim without lowpass
        epochs_dec.info['lowpass'] = None
        epochs_dec.decimate(2)
        assert_equal(len(w), 2)

    data_dec = epochs_dec.get_data()
    assert_allclose(data[:, :, epochs_dec._decim_slice], data_dec, rtol=1e-7,
                    atol=1e-12)

    evoked_dec = epochs_dec.average()
    assert_allclose(evoked.data[:, epochs_dec._decim_slice],
                    evoked_dec.data, rtol=1e-12, atol=1e-17)

    n = evoked.data.shape[1]
    n_dec = evoked_dec.data.shape[1]
    n_dec_min = n // 4
    assert_true(n_dec_min <= n_dec <= n_dec_min + 1)
    assert_true(evoked_dec.info['sfreq'] == evoked.info['sfreq'] / 4)

    # Test event access on non-preloaded data (#2345)

    # due to reapplication of the proj matrix, this is our quality limit
    # for some tests
    tols = dict(atol=1e-3, rtol=1e-20)

    raw, events, picks = _get_data()
    events[::2, 1] = 1
    events[1::2, 2] = 2
    event_ids = dict(a=1, b=2)
    for proj in (True, 'delayed', False):
        epochs = Epochs(raw, events, event_ids, tmin, tmax, picks=picks,
                        proj=proj, reject=reject)
        assert_equal(epochs.proj, proj if proj != 'delayed' else False)
        data1 = epochs.get_data()
        epochs2 = epochs.copy().apply_proj()
        assert_equal(epochs2.proj, True)
        data2 = epochs2.get_data()
        assert_allclose(data1, data2, **tols)
        epochs.save(temp_fname)
        epochs_read = read_epochs(temp_fname, preload=False)
        assert_allclose(epochs.get_data(), epochs_read.get_data(), **tols)
        assert_allclose(epochs['a'].get_data(),
                        epochs_read['a'].get_data(), **tols)
        assert_allclose(epochs['b'].get_data(),
                        epochs_read['b'].get_data(), **tols)

    # ensure we don't leak file descriptors
    epochs_read = read_epochs(temp_fname, preload=False)
    epochs_copy = epochs_read.copy()
    del epochs_read
    epochs_copy.get_data()
    with warnings.catch_warnings(record=True) as w:
        del epochs_copy
    assert_equal(len(w), 0)

    # test IO
    for preload in (False, True):
        epochs = epochs_orig.copy()
        epochs.save(temp_fname)
        epochs_no_bl.save(temp_fname_no_bl)
        epochs_read = read_epochs(temp_fname, preload=preload)
        epochs_no_bl.save(temp_fname_no_bl)
        epochs_read = read_epochs(temp_fname)
        epochs_no_bl_read = read_epochs(temp_fname_no_bl)
        assert_raises(ValueError, epochs.apply_baseline, baseline=[1, 2, 3])
        epochs_with_bl = epochs_no_bl_read.copy().apply_baseline(baseline)
        assert_true(isinstance(epochs_with_bl, BaseEpochs))
        assert_true(epochs_with_bl.baseline == baseline)
        assert_true(epochs_no_bl_read.baseline != baseline)
        assert_true(str(epochs_read).startswith('<Epochs'))

        epochs_no_bl_read.apply_baseline(baseline)
        assert_array_equal(epochs_no_bl_read.times, epochs.times)
        assert_array_almost_equal(epochs_read.get_data(), epochs.get_data())
        assert_array_almost_equal(epochs.get_data(),
                                  epochs_no_bl_read.get_data())
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
        epochs_read2 = read_epochs(op.join(tempdir, 'foo-epo.fif'),
                                   preload=preload)
        assert_equal(epochs_read2.event_id, epochs.event_id)

        # add reject here so some of the epochs get dropped
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        reject=reject)
        epochs.save(temp_fname)
        # ensure bad events are not saved
        epochs_read3 = read_epochs(temp_fname, preload=preload)
        assert_array_equal(epochs_read3.events, epochs.events)
        data = epochs.get_data()
        assert_true(epochs_read3.events.shape[0] == data.shape[0])

        # test copying loaded one (raw property)
        epochs_read4 = epochs_read3.copy()
        assert_array_almost_equal(epochs_read4.get_data(), data)
        # test equalizing loaded one (drop_log property)
        epochs_read4.equalize_event_counts(epochs.event_id)

        epochs.drop([1, 2], reason='can we recover orig ID?')
        epochs.save(temp_fname)
        epochs_read5 = read_epochs(temp_fname, preload=preload)
        assert_array_equal(epochs_read5.selection, epochs.selection)
        assert_equal(len(epochs_read5.selection), len(epochs_read5.events))
        assert_array_equal(epochs_read5.drop_log, epochs.drop_log)

        if preload:
            # Test that one can drop channels on read file
            epochs_read5.drop_channels(epochs_read5.ch_names[:1])

        # test warnings on bad filenames
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            epochs_badname = op.join(tempdir, 'test-bad-name.fif.gz')
            epochs.save(epochs_badname)
            read_epochs(epochs_badname, preload=preload)
        assert_naming(w, 'test_epochs.py', 2)

        # test loading epochs with missing events
        epochs = Epochs(raw, events, dict(foo=1, bar=999), tmin, tmax,
                        picks=picks, on_missing='ignore')
        epochs.save(temp_fname)
        epochs_read = read_epochs(temp_fname, preload=preload)
        assert_allclose(epochs.get_data(), epochs_read.get_data(), **tols)
        assert_array_equal(epochs.events, epochs_read.events)
        assert_equal(set(epochs.event_id.keys()),
                     set(text_type(x) for x in epochs_read.event_id.keys()))

        # test saving split epoch files
        epochs.save(temp_fname, split_size='7MB')
        epochs_read = read_epochs(temp_fname, preload=preload)
        assert_allclose(epochs.get_data(), epochs_read.get_data(), **tols)
        assert_array_equal(epochs.events, epochs_read.events)
        assert_array_equal(epochs.selection, epochs_read.selection)
        assert_equal(epochs.drop_log, epochs_read.drop_log)

        # Test that having a single time point works
        epochs.load_data().crop(0, 0)
        assert_equal(len(epochs.times), 1)
        assert_equal(epochs.get_data().shape[-1], 1)
        epochs.save(temp_fname)
        epochs_read = read_epochs(temp_fname, preload=preload)
        assert_equal(len(epochs_read.times), 1)
        assert_equal(epochs.get_data().shape[-1], 1)


def test_epochs_proj():
    """Test handling projection (apply proj in Raw or in Epochs)."""
    tempdir = _TempDir()
    raw, events, picks = _get_data()
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more
    this_picks = pick_types(raw.info, meg=True, eeg=False, stim=True,
                            eog=True, exclude=exclude)
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    proj=True)
    assert_true(all(p['active'] is True for p in epochs.info['projs']))
    evoked = epochs.average()
    assert_true(all(p['active'] is True for p in evoked.info['projs']))
    data = epochs.get_data()

    raw_proj = read_raw_fif(raw_fname).apply_proj()
    epochs_no_proj = Epochs(raw_proj, events[:4], event_id, tmin, tmax,
                            picks=this_picks, proj=False)

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
                    proj=True)
    epochs.set_eeg_reference().apply_proj()
    assert_true(_has_eeg_average_ref_proj(epochs.info['projs']))
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    proj=True)
    assert_true(not _has_eeg_average_ref_proj(epochs.info['projs']))

    # make sure we don't add avg ref when a custom ref has been applied
    raw.info['custom_ref_applied'] = True
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    proj=True)
    assert_true(not _has_eeg_average_ref_proj(epochs.info['projs']))

    # From GH#2200:
    # This has no problem
    proj = raw.info['projs']
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=this_picks,
                    proj=False)
    epochs.info['projs'] = []
    data = epochs.copy().add_proj(proj).apply_proj().get_data()
    # save and reload data
    fname_epo = op.join(tempdir, 'temp-epo.fif')
    epochs.save(fname_epo)  # Save without proj added
    epochs_read = read_epochs(fname_epo)
    epochs_read.add_proj(proj)
    epochs_read.apply_proj()  # This used to bomb
    data_2 = epochs_read.get_data()  # Let's check the result
    assert_allclose(data, data_2, atol=1e-15, rtol=1e-3)

    # adding EEG ref (GH #2727)
    raw = read_raw_fif(raw_fname)
    raw.add_proj([], remove_existing=True)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']
    picks = pick_types(raw.info, meg=False, eeg=True, stim=True, eog=False,
                       exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    preload=True)
    epochs.pick_channels(['EEG 001', 'EEG 002'])
    assert_equal(len(epochs), 7)  # sufficient for testing
    temp_fname = op.join(tempdir, 'test-epo.fif')
    epochs.save(temp_fname)
    for preload in (True, False):
        epochs = read_epochs(temp_fname, proj=False, preload=preload)
        epochs.set_eeg_reference().apply_proj()
        assert_allclose(epochs.get_data().mean(axis=1), 0, atol=1e-15)
        epochs = read_epochs(temp_fname, proj=False, preload=preload)
        epochs.set_eeg_reference()
        assert_raises(AssertionError, assert_allclose,
                      epochs.get_data().mean(axis=1), 0., atol=1e-15)
        epochs.apply_proj()
        assert_allclose(epochs.get_data().mean(axis=1), 0, atol=1e-15)


def test_evoked_arithmetic():
    """Test arithmetic of evoked data."""
    raw, events, picks = _get_data()
    epochs1 = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks)
    evoked1 = epochs1.average()
    epochs2 = Epochs(raw, events[4:8], event_id, tmin, tmax, picks=picks)
    evoked2 = epochs2.average()
    epochs = Epochs(raw, events[:8], event_id, tmin, tmax, picks=picks)
    evoked = epochs.average()
    evoked_sum = combine_evoked([evoked1, evoked2], weights='nave')
    assert_array_equal(evoked.data, evoked_sum.data)
    assert_array_equal(evoked.times, evoked_sum.times)
    assert_equal(evoked_sum.nave, evoked1.nave + evoked2.nave)
    evoked_diff = combine_evoked([evoked1, evoked1], weights=[1, -1])
    assert_array_equal(np.zeros_like(evoked.data), evoked_diff.data)


def test_evoked_io_from_epochs():
    """Test IO of evoked data made from epochs."""
    tempdir = _TempDir()
    raw, events, picks = _get_data()
    # offset our tmin so we don't get exactly a zero value when decimating
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs = Epochs(raw, events[:4], event_id, tmin + 0.011, tmax,
                        picks=picks, decim=5)
    assert_true(len(w) == 1)
    evoked = epochs.average()
    evoked.info['proj_name'] = ''  # Test that empty string shortcuts to None.
    evoked.save(op.join(tempdir, 'evoked-ave.fif'))
    evoked2 = read_evokeds(op.join(tempdir, 'evoked-ave.fif'))[0]
    assert_equal(evoked2.info['proj_name'], None)
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
    """Test calculation and read/write of standard error."""
    raw, events, picks = _get_data()
    tempdir = _TempDir()
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks)
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
    """Test of epochs rejection."""
    raw, events, picks = _get_data()
    events1 = events[events[:, 2] == event_id]
    epochs = Epochs(raw, events1, event_id, tmin, tmax,
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
    epochs = Epochs(raw_2, events1, event_id, tmin, tmax,
                    reject=reject_crazy, flat=flat)
    epochs.drop_bad()

    assert_true(all('MEG 2442' in e for e in epochs.drop_log))
    assert_true(all('MEG 2443' not in e for e in epochs.drop_log))

    # Invalid reject_tmin/reject_tmax/detrend
    assert_raises(ValueError, Epochs, raw, events1, event_id, tmin, tmax,
                  reject_tmin=1., reject_tmax=0)
    assert_raises(ValueError, Epochs, raw, events1, event_id, tmin, tmax,
                  reject_tmin=tmin - 1, reject_tmax=1.)
    assert_raises(ValueError, Epochs, raw, events1, event_id, tmin, tmax,
                  reject_tmin=0., reject_tmax=tmax + 1)

    epochs = Epochs(raw, events1, event_id, tmin, tmax, picks=picks,
                    reject=reject, flat=flat, reject_tmin=0., reject_tmax=.1)
    data = epochs.get_data()
    n_clean_epochs = len(data)
    assert_true(n_clean_epochs == 7)
    assert_true(len(epochs) == 7)
    assert_true(epochs.times[epochs._reject_time][0] >= 0.)
    assert_true(epochs.times[epochs._reject_time][-1] <= 0.1)

    # Invalid data for _is_good_epoch function
    epochs = Epochs(raw, events1, event_id, tmin, tmax)
    assert_equal(epochs._is_good_epoch(None), (False, ['NO_DATA']))
    assert_equal(epochs._is_good_epoch(np.zeros((1, 1))),
                 (False, ['TOO_SHORT']))
    data = epochs[0].get_data()[0]
    assert_equal(epochs._is_good_epoch(data), (True, None))


def test_preload_epochs():
    """Test preload of epochs."""
    raw, events, picks = _get_data()
    epochs_preload = Epochs(raw, events[:16], event_id, tmin, tmax,
                            picks=picks, preload=True,
                            reject=reject, flat=flat)
    data_preload = epochs_preload.get_data()

    epochs = Epochs(raw, events[:16], event_id, tmin, tmax, picks=picks,
                    preload=False, reject=reject, flat=flat)
    data = epochs.get_data()
    assert_array_equal(data_preload, data)
    assert_array_almost_equal(epochs_preload.average().data,
                              epochs.average().data, 18)


def test_indexing_slicing():
    """Test of indexing and slicing operations."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:20], event_id, tmin, tmax, picks=picks,
                    reject=reject, flat=flat)

    data_normal = epochs.get_data()

    n_good_events = data_normal.shape[0]

    # indices for slicing
    start_index = 1
    end_index = n_good_events - 1

    assert((end_index - start_index) > 0)

    for preload in [True, False]:
        epochs2 = Epochs(raw, events[:20], event_id, tmin, tmax, picks=picks,
                         preload=preload, reject=reject, flat=flat)

        if not preload:
            epochs2.drop_bad()

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
        idx = rng.randint(0, data_epochs2_sliced.shape[0], 10)
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
    """Test of average obtained vs C code."""
    raw, events = _get_data()[:2]
    c_evoked = read_evokeds(evoked_nf_name, condition=0)
    epochs = Epochs(raw, events, event_id, tmin, tmax, baseline=None,
                    preload=True, proj=False)
    evoked = epochs.set_eeg_reference().apply_proj().average()
    sel = pick_channels(c_evoked.ch_names, evoked.ch_names)
    evoked_data = evoked.data
    c_evoked_data = c_evoked.data[sel]

    assert_true(evoked.nave == c_evoked.nave)
    assert_array_almost_equal(evoked_data, c_evoked_data, 10)
    assert_array_almost_equal(evoked.times, c_evoked.times, 12)


def test_crop():
    """Test of crop of epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    preload=False, reject=reject, flat=flat)
    assert_raises(RuntimeError, epochs.crop, None, 0.2)  # not preloaded
    data_normal = epochs.get_data()

    epochs2 = Epochs(raw, events[:5], event_id, tmin, tmax,
                     picks=picks, preload=True, reject=reject, flat=flat)
    with warnings.catch_warnings(record=True) as w:
        epochs2.crop(-20, 200)
    assert_true(len(w) == 2)

    # indices for slicing
    tmin_window = tmin + 0.1
    tmax_window = tmax - 0.1
    tmask = (epochs.times >= tmin_window) & (epochs.times <= tmax_window)
    assert_true(tmin_window > tmin)
    assert_true(tmax_window < tmax)
    epochs3 = epochs2.copy().crop(tmin_window, tmax_window)
    data3 = epochs3.get_data()
    epochs2.crop(tmin_window, tmax_window)
    data2 = epochs2.get_data()
    assert_array_equal(data2, data_normal[:, :, tmask])
    assert_array_equal(data3, data_normal[:, :, tmask])
    assert_array_equal(epochs.time_as_index([tmin, tmax], use_rounding=True),
                       [0, len(epochs.times) - 1])
    assert_array_equal(epochs3.time_as_index([tmin_window, tmax_window],
                                             use_rounding=True),
                       [0, len(epochs3.times) - 1])

    # test time info is correct
    epochs = EpochsArray(np.zeros((1, 1, 1000)), create_info(1, 1000., 'eeg'),
                         np.ones((1, 3), int), tmin=-0.2)
    epochs.crop(-.200, .700)
    last_time = epochs.times[-1]
    with warnings.catch_warnings(record=True):  # not LP filtered
        epochs.decimate(10)
    assert_allclose(last_time, epochs.times[-1])

    epochs = Epochs(raw, events[:5], event_id, -1, 1,
                    picks=picks, preload=True, reject=reject, flat=flat)
    # We include nearest sample, so actually a bit beyound our bounds here
    assert_allclose(epochs.tmin, -1.0006410259015925, rtol=1e-12)
    assert_allclose(epochs.tmax, 1.0006410259015925, rtol=1e-12)
    epochs_crop = epochs.copy().crop(-1, 1)
    assert_allclose(epochs.times, epochs_crop.times, rtol=1e-12)
    # Ensure we don't allow silly crops
    with warnings.catch_warnings(record=True):  # tmin/tmax out of bounds
        assert_raises(ValueError, epochs.crop, 1000, 2000)
        assert_raises(ValueError, epochs.crop, 0.1, 0)


def test_resample():
    """Test of resample of epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    preload=False, reject=reject, flat=flat)
    assert_raises(RuntimeError, epochs.resample, 100)

    epochs_o = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                      preload=True, reject=reject, flat=flat)
    epochs = epochs_o.copy()

    data_normal = cp.deepcopy(epochs.get_data())
    times_normal = cp.deepcopy(epochs.times)
    sfreq_normal = epochs.info['sfreq']
    # upsample by 2
    epochs = epochs_o.copy()
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
    epochs = epochs_o.copy()
    epochs.resample(sfreq_normal * 2, n_jobs=2, npad=0)
    assert_true(np.allclose(data_up, epochs._data, rtol=1e-8, atol=1e-16))

    # test copy flag
    epochs = epochs_o.copy()
    epochs_resampled = epochs.copy().resample(sfreq_normal * 2, npad=0)
    assert_true(epochs_resampled is not epochs)
    epochs_resampled = epochs.resample(sfreq_normal * 2, npad=0)
    assert_true(epochs_resampled is epochs)

    # test proper setting of times (#2645)
    n_trial, n_chan, n_time, sfreq = 1, 1, 10, 1000.
    data = np.zeros((n_trial, n_chan, n_time))
    events = np.zeros((n_trial, 3), int)
    info = create_info(n_chan, sfreq, 'eeg')
    epochs1 = EpochsArray(data, deepcopy(info), events)
    epochs2 = EpochsArray(data, deepcopy(info), events)
    epochs = concatenate_epochs([epochs1, epochs2])
    epochs1.resample(epochs1.info['sfreq'] // 2, npad='auto')
    epochs2.resample(epochs2.info['sfreq'] // 2, npad='auto')
    epochs = concatenate_epochs([epochs1, epochs2])
    for e in epochs1, epochs2, epochs:
        assert_equal(e.times[0], epochs.tmin)
        assert_equal(e.times[-1], epochs.tmax)
    # test that cropping after resampling works (#3296)
    this_tmin = -0.002
    epochs = EpochsArray(data, deepcopy(info), events, tmin=this_tmin)
    for times in (epochs.times, epochs._raw_times):
        assert_allclose(times, np.arange(n_time) / sfreq + this_tmin)
    epochs.resample(info['sfreq'] * 2.)
    for times in (epochs.times, epochs._raw_times):
        assert_allclose(times, np.arange(2 * n_time) / (sfreq * 2) + this_tmin)
    epochs.crop(0, None)
    for times in (epochs.times, epochs._raw_times):
        assert_allclose(times, np.arange((n_time - 2) * 2) / (sfreq * 2))
    epochs.resample(sfreq)
    for times in (epochs.times, epochs._raw_times):
        assert_allclose(times, np.arange(n_time - 2) / sfreq)


def test_detrend():
    """Test detrending of epochs."""
    raw, events, picks = _get_data()

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

    for value in ['foo', 2, False, True]:
        assert_raises(ValueError, Epochs, raw, events[:4], event_id,
                      tmin, tmax, detrend=value)


def test_bootstrap():
    """Test of bootstrapping of epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    preload=True, reject=reject, flat=flat)
    epochs2 = bootstrap(epochs, random_state=0)
    assert_true(len(epochs2.events) == len(epochs.events))
    assert_true(epochs._data.shape == epochs2._data.shape)


def test_epochs_copy():
    """Test copy epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    preload=True, reject=reject, flat=flat)
    copied = epochs.copy()
    assert_array_equal(epochs._data, copied._data)

    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    preload=False, reject=reject, flat=flat)
    copied = epochs.copy()
    data = epochs.get_data()
    copied_data = copied.get_data()
    assert_array_equal(data, copied_data)


def test_iter_evoked():
    """Test the iterator for epochs -> evoked."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks)

    for ii, ev in enumerate(epochs.iter_evoked()):
        x = ev.data
        y = epochs.get_data()[ii, :, :]
        assert_array_equal(x, y)


def test_subtract_evoked():
    """Test subtraction of Evoked from Epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks)

    # make sure subraction fails if data channels are missing
    assert_raises(ValueError, epochs.subtract_evoked,
                  epochs.average(picks[:5]))

    # do the subraction using the default argument
    epochs.subtract_evoked()

    # apply SSP now
    epochs.apply_proj()

    # use preloading and SSP from the start
    epochs2 = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                     preload=True)

    evoked = epochs2.average()
    epochs2.subtract_evoked(evoked)

    # this gives the same result
    assert_allclose(epochs.get_data(), epochs2.get_data())

    # if we compute the evoked response after subtracting it we get zero
    zero_evoked = epochs.average()
    data = zero_evoked.data
    assert_allclose(data, np.zeros_like(data), atol=1e-15)


def test_epoch_eq():
    """Test epoch count equalization and condition combining."""
    raw, events, picks = _get_data()
    # equalizing epochs objects
    epochs_1 = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    epochs_2 = Epochs(raw, events, event_id_2, tmin, tmax, picks=picks)
    epochs_1.drop_bad()  # make sure drops are logged
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
    epochs.drop_bad()  # make sure drops are logged
    assert_true(len([l for l in epochs.drop_log if not l]) ==
                len(epochs.events))
    drop_log1 = deepcopy(epochs.drop_log)
    old_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    epochs.equalize_event_counts(['a', 'b'])
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
    epochs.equalize_event_counts([['a', 'b'], 'c'])
    new_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    assert_true(new_shapes[0] + new_shapes[1] == new_shapes[2])
    assert_true(new_shapes[3] == old_shapes[3])
    assert_raises(KeyError, epochs.equalize_event_counts, [1, 'a'])

    # now let's combine conditions
    old_shapes = new_shapes
    epochs.equalize_event_counts([['a', 'b'], ['c', 'd']])
    new_shapes = [epochs[key].events.shape[0] for key in ['a', 'b', 'c', 'd']]
    assert_true(old_shapes[0] + old_shapes[1] == new_shapes[0] + new_shapes[1])
    assert_true(new_shapes[0] + new_shapes[1] == new_shapes[2] + new_shapes[3])
    assert_raises(ValueError, combine_event_ids, epochs, ['a', 'b'], {'ab': 1})

    combine_event_ids(epochs, ['a', 'b'], {'ab': 12}, copy=False)
    caught = 0
    for key in ['a', 'b']:
        try:
            epochs[key]
        except KeyError:
            caught += 1
    assert_equal(caught, 2)
    assert_true(not np.any(epochs.events[:, 2] == 1))
    assert_true(not np.any(epochs.events[:, 2] == 2))
    epochs = combine_event_ids(epochs, ['c', 'd'], {'cd': 34})
    assert_true(np.all(np.logical_or(epochs.events[:, 2] == 12,
                                     epochs.events[:, 2] == 34)))
    assert_true(epochs['ab'].events.shape[0] == old_shapes[0] + old_shapes[1])
    assert_true(epochs['ab'].events.shape[0] == epochs['cd'].events.shape[0])

    # equalizing with hierarchical tags
    epochs = Epochs(raw, events, {'a/x': 1, 'b/x': 2, 'a/y': 3, 'b/y': 4},
                    tmin, tmax, picks=picks, reject=reject)
    cond1, cond2 = ['a', ['b/x', 'b/y']], [['a/x', 'a/y'], 'b']
    es = [epochs.copy().equalize_event_counts(c)[0]
          for c in (cond1, cond2)]
    assert_array_equal(es[0].events[:, 0], es[1].events[:, 0])
    cond1, cond2 = ['a', ['b', 'b/y']], [['a/x', 'a/y'], 'x']
    for c in (cond1, cond2):  # error b/c tag and id mix/non-orthogonal tags
        assert_raises(ValueError, epochs.equalize_event_counts, c)
    assert_raises(KeyError, epochs.equalize_event_counts,
                  ["a/no_match", "b"])
    # test equalization with no events of one type
    epochs.drop(np.arange(10))
    assert_equal(len(epochs['a/x']), 0)
    assert_true(len(epochs['a/y']) > 0)
    epochs.equalize_event_counts(['a/x', 'a/y'])
    assert_equal(len(epochs['a/x']), 0)
    assert_equal(len(epochs['a/y']), 0)


def test_access_by_name():
    """Test accessing epochs by event name and on_missing for rare events."""
    tempdir = _TempDir()
    raw, events, picks = _get_data()

    # Test various invalid inputs
    assert_raises(ValueError, Epochs, raw, events, {1: 42, 2: 42}, tmin,
                  tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, {'a': 'spam', 2: 'eggs'},
                  tmin, tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, {'a': 'spam', 2: 'eggs'},
                  tmin, tmax, picks=picks)
    assert_raises(ValueError, Epochs, raw, events, 'foo', tmin, tmax,
                  picks=picks)
    assert_raises(ValueError, Epochs, raw, events, ['foo'], tmin, tmax,
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

    # Test constructing epochs with a list of ints as events
    epochs = Epochs(raw, events, [1, 2], tmin, tmax, picks=picks)
    for k, v in epochs.event_id.items():
        assert_equal(int(k), v)

    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks)
    assert_raises(KeyError, epochs.__getitem__, 'bar')

    data = epochs['a'].get_data()
    event_a = events[events[:, 2] == 1]
    assert_true(len(data) == len(event_a))

    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks,
                    preload=True)
    assert_raises(KeyError, epochs.__getitem__, 'bar')
    temp_fname = op.join(tempdir, 'test-epo.fif')
    epochs.save(temp_fname)
    epochs2 = read_epochs(temp_fname)

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

    # Make sure we preserve names
    assert_equal(epochs['a'].name, 'a')
    assert_equal(epochs[['a', 'b']]['a'].name, 'a')


@requires_pandas
def test_to_data_frame():
    """Test epochs Pandas exporter."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax, picks=picks)
    assert_raises(ValueError, epochs.to_data_frame, index=['foo', 'bar'])
    assert_raises(ValueError, epochs.to_data_frame, index='qux')
    assert_raises(ValueError, epochs.to_data_frame, np.arange(400))

    df = epochs.to_data_frame(index=['condition', 'epoch', 'time'],
                              picks=list(range(epochs.info['nchan'])))

    # Default index and picks
    df2 = epochs.to_data_frame()
    assert_equal(df.index.names, df2.index.names)
    assert_array_equal(df.columns.values, epochs.ch_names)

    data = np.hstack(epochs.get_data())
    assert_true((df.columns == epochs.ch_names).all())
    assert_array_equal(df.values[:, 0], data[0] * 1e13)
    assert_array_equal(df.values[:, 2], data[2] * 1e15)
    for ind in ['time', ['condition', 'time'], ['condition', 'time', 'epoch']]:
        df = epochs.to_data_frame(index=ind)
        assert_true(df.index.names == ind if isinstance(ind, list) else [ind])
        # test that non-indexed data were present as categorial variables
        assert_array_equal(sorted(df.reset_index().columns[:3]),
                           sorted(['time', 'condition', 'epoch']))


def test_epochs_proj_mixin():
    """Test SSP proj methods from ProjMixin class."""
    raw, events, picks = _get_data()
    for proj in [True, False]:
        epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        proj=proj)

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
            # Test that already existing projections are not added.
            epochs.add_proj(projs, remove_existing=False)
            assert_true(len(epochs.info['projs']) == n_proj)
            epochs.add_proj(projs[:-1], remove_existing=True)
            assert_true(len(epochs.info['projs']) == n_proj - 1)

    # catch no-gos.
    # wrong proj argument
    assert_raises(ValueError, Epochs, raw, events[:4], event_id, tmin, tmax,
                  picks=picks, proj='crazy')

    for preload in [True, False]:
        epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                        proj='delayed', preload=preload,
                        reject=reject).set_eeg_reference()
        epochs_proj = Epochs(
            raw, events[:4], event_id, tmin, tmax, picks=picks,
            proj=True, preload=preload,
            reject=reject).set_eeg_reference().apply_proj()

        epochs_noproj = Epochs(
            raw, events[:4], event_id, tmin, tmax, picks=picks,
            proj=False, preload=preload, reject=reject).set_eeg_reference()

        assert_allclose(epochs.copy().apply_proj().get_data(),
                        epochs_proj.get_data(), rtol=1e-10, atol=1e-25)
        assert_allclose(epochs.get_data(),
                        epochs_noproj.get_data(), rtol=1e-10, atol=1e-25)

        # make sure data output is constant across repeated calls
        # e.g. drop bads
        assert_array_equal(epochs.get_data(), epochs.get_data())
        assert_array_equal(epochs_proj.get_data(), epochs_proj.get_data())
        assert_array_equal(epochs_noproj.get_data(), epochs_noproj.get_data())

    # test epochs.next calls
    data = epochs.get_data().copy()
    data2 = np.array([e for e in epochs])
    assert_array_equal(data, data2)

    # cross application from processing stream 1 to 2
    epochs.apply_proj()
    assert_array_equal(epochs._projector, epochs_proj._projector)
    assert_allclose(epochs._data, epochs_proj.get_data())

    # test mixin against manual application
    epochs = Epochs(raw, events[:4], event_id, tmin, tmax, picks=picks,
                    baseline=None, proj=False).set_eeg_reference()
    data = epochs.get_data().copy()
    epochs.apply_proj()
    assert_allclose(np.dot(epochs._projector, data[0]), epochs._data[0])


def test_delayed_epochs():
    """Test delayed projection on Epochs."""
    raw, events, picks = _get_data()
    events = events[:10]
    picks = np.concatenate([pick_types(raw.info, meg=True, eeg=True)[::22],
                            pick_types(raw.info, meg=False, eeg=False,
                                       ecg=True, eog=True)])
    picks = np.sort(picks)
    raw.load_data().pick_channels([raw.ch_names[pick] for pick in picks])
    raw.info.normalize_proj()
    del picks
    n_epochs = 2  # number we expect after rejection
    raw.info['lowpass'] = 40.  # fake the LP info so no warnings
    for decim in (1, 3):
        proj_data = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                           reject=reject, decim=decim)
        use_tmin = proj_data.tmin
        proj_data = proj_data.get_data()
        noproj_data = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                             reject=reject, decim=decim).get_data()
        assert_equal(proj_data.shape, noproj_data.shape)
        assert_equal(proj_data.shape[0], n_epochs)
        for preload in (True, False):
            for proj in (True, False, 'delayed'):
                for ii in range(3):
                    print(decim, preload, proj, ii)
                    comp = proj_data if proj is True else noproj_data
                    if ii in (0, 1):
                        epochs = Epochs(raw, events, event_id, tmin, tmax,
                                        proj=proj, reject=reject,
                                        preload=preload, decim=decim)
                    else:
                        fake_events = np.zeros((len(comp), 3), int)
                        fake_events[:, 0] = np.arange(len(comp))
                        fake_events[:, 2] = 1
                        epochs = EpochsArray(comp, raw.info, tmin=use_tmin,
                                             event_id=1, events=fake_events,
                                             proj=proj)
                        epochs.info['sfreq'] /= decim
                        assert_equal(len(epochs), n_epochs)
                    assert_true(raw.proj is False)
                    assert_true(epochs.proj is
                                (True if proj is True else False))
                    if ii == 1:
                        epochs.load_data()
                    picks_data = pick_types(epochs.info, meg=True, eeg=True)
                    evoked = epochs.average(picks=picks_data)
                    assert_equal(evoked.nave, n_epochs, epochs.drop_log)
                    if proj is True:
                        evoked.apply_proj()
                    else:
                        assert_true(evoked.proj is False)
                    assert_array_equal(evoked.ch_names,
                                       np.array(epochs.ch_names)[picks_data])
                    assert_allclose(evoked.times, epochs.times)
                    epochs_data = epochs.get_data()
                    assert_allclose(evoked.data,
                                    epochs_data.mean(axis=0)[picks_data],
                                    rtol=1e-5, atol=1e-20)
                    assert_allclose(epochs_data, comp, rtol=1e-5, atol=1e-20)


def test_drop_epochs():
    """Test dropping of epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks)
    events1 = events[events[:, 2] == event_id]

    # Bound checks
    assert_raises(IndexError, epochs.drop, [len(epochs.events)])
    assert_raises(IndexError, epochs.drop, [-1])
    assert_raises(ValueError, epochs.drop, [[1, 2], [3, 4]])

    # Test selection attribute
    assert_array_equal(epochs.selection,
                       np.where(events[:, 2] == event_id)[0])
    assert_equal(len(epochs.drop_log), len(events))
    assert_true(all(epochs.drop_log[k] == ['IGNORED']
                for k in set(range(len(events))) - set(epochs.selection)))

    selection = epochs.selection.copy()
    n_events = len(epochs.events)
    epochs.drop([2, 4], reason='d')
    assert_equal(epochs.drop_log_stats(), 2. / n_events * 100)
    assert_equal(len(epochs.drop_log), len(events))
    assert_equal([epochs.drop_log[k]
                  for k in selection[[2, 4]]], [['d'], ['d']])
    assert_array_equal(events[epochs.selection], events1[[0, 1, 3, 5, 6]])
    assert_array_equal(events[epochs[3:].selection], events1[[5, 6]])
    assert_array_equal(events[epochs['1'].selection], events1[[0, 1, 3, 5, 6]])


def test_drop_epochs_mult():
    """Test that subselecting epochs or making less epochs is equivalent."""
    raw, events, picks = _get_data()
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
    """Test membership API."""
    raw, events = _get_data(True)[:2]
    # Add seeg channel
    seeg = RawArray(np.zeros((1, len(raw.times))),
                    create_info(['SEEG 001'], raw.info['sfreq'], 'seeg'))
    for key in ('dev_head_t', 'buffer_size_sec', 'highpass', 'lowpass',
                'dig', 'description', 'acq_pars', 'experimenter',
                'proj_name'):
        seeg.info[key] = raw.info[key]
    raw.add_channels([seeg])
    tests = [(('mag', False, False), ('grad', 'eeg', 'seeg')),
             (('grad', False, False), ('mag', 'eeg', 'seeg')),
             ((False, True, False), ('grad', 'mag', 'seeg')),
             ((False, False, True), ('grad', 'mag', 'eeg'))]

    for (meg, eeg, seeg), others in tests:
        picks_contains = pick_types(raw.info, meg=meg, eeg=eeg, seeg=seeg)
        epochs = Epochs(raw, events, {'a': 1, 'b': 2}, tmin, tmax,
                        picks=picks_contains)
        if eeg:
            test = 'eeg'
        elif seeg:
            test = 'seeg'
        else:
            test = meg
        assert_true(test in epochs)
        assert_true(not any(o in epochs for o in others))

    assert_raises(ValueError, epochs.__contains__, 'foo')
    assert_raises(ValueError, epochs.__contains__, 1)


def test_drop_channels_mixin():
    """Test channels-dropping functionality."""
    raw, events = _get_data()[:2]
    # here without picks to get additional coverage
    epochs = Epochs(raw, events, event_id, tmin, tmax, preload=True)
    drop_ch = epochs.ch_names[:3]
    ch_names = epochs.ch_names[3:]

    ch_names_orig = epochs.ch_names
    dummy = epochs.copy().drop_channels(drop_ch)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, epochs.ch_names)
    assert_equal(len(ch_names_orig), epochs.get_data().shape[1])

    epochs.drop_channels(drop_ch)
    assert_equal(ch_names, epochs.ch_names)
    assert_equal(len(ch_names), epochs.get_data().shape[1])


def test_pick_channels_mixin():
    """Test channel-picking functionality."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True)
    ch_names = epochs.ch_names[:3]
    epochs.preload = False
    assert_raises(RuntimeError, epochs.drop_channels, [ch_names[0]])
    epochs.preload = True
    ch_names_orig = epochs.ch_names
    dummy = epochs.copy().pick_channels(ch_names)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, epochs.ch_names)
    assert_equal(len(ch_names_orig), epochs.get_data().shape[1])

    epochs.pick_channels(ch_names)
    assert_equal(ch_names, epochs.ch_names)
    assert_equal(len(ch_names), epochs.get_data().shape[1])

    # Invalid picks
    assert_raises(ValueError, Epochs, raw, events, event_id, tmin, tmax,
                  picks=[])


def test_equalize_channels():
    """Test equalization of channels."""
    raw, events, picks = _get_data()
    epochs1 = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                     proj=False, preload=True)
    epochs2 = epochs1.copy()
    ch_names = epochs1.ch_names[2:]
    epochs1.drop_channels(epochs1.ch_names[:1])
    epochs2.drop_channels(epochs2.ch_names[1:2])
    my_comparison = [epochs1, epochs2]
    equalize_channels(my_comparison)
    for e in my_comparison:
        assert_equal(ch_names, e.ch_names)


def test_illegal_event_id():
    """Test handling of invalid events ids."""
    raw, events, picks = _get_data()
    event_id_illegal = dict(aud_l=1, does_not_exist=12345678)

    assert_raises(ValueError, Epochs, raw, events, event_id_illegal, tmin,
                  tmax, picks=picks, proj=False)


def test_add_channels_epochs():
    """Test adding channels"""
    raw, events, picks = _get_data()

    def make_epochs(picks, proj):
        return Epochs(raw, events, event_id, tmin, tmax, preload=True,
                      proj=proj, picks=picks)

    picks = pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    picks_meg = pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')

    for proj in (False, True):
        epochs = make_epochs(picks=picks, proj=proj)
        epochs_meg = make_epochs(picks=picks_meg, proj=proj)
        epochs_eeg = make_epochs(picks=picks_eeg, proj=proj)
        epochs.info._check_consistency()
        epochs_meg.info._check_consistency()
        epochs_eeg.info._check_consistency()

        epochs2 = add_channels_epochs([epochs_meg, epochs_eeg])

        assert_equal(len(epochs.info['projs']), len(epochs2.info['projs']))
        assert_equal(len(epochs.info.keys()), len(epochs_meg.info.keys()))
        assert_equal(len(epochs.info.keys()), len(epochs_eeg.info.keys()))
        assert_equal(len(epochs.info.keys()), len(epochs2.info.keys()))

        data1 = epochs.get_data()
        data2 = epochs2.get_data()
        data3 = np.concatenate([e.get_data() for e in
                                [epochs_meg, epochs_eeg]], axis=1)
        assert_array_equal(data1.shape, data2.shape)
        assert_allclose(data1, data3, atol=1e-25)
        assert_allclose(data1, data2, atol=1e-25)

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['meas_date'] += 10
    add_channels_epochs([epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs2 = add_channels_epochs([epochs_meg, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.events[3, 2] -= 1
    assert_raises(ValueError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    assert_raises(ValueError, add_channels_epochs,
                  [epochs_meg, epochs_eeg[:2]])

    epochs_meg.info['chs'].pop(0)
    epochs_meg.info._update_redundant()
    assert_raises(RuntimeError, add_channels_epochs, [epochs_meg, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['sfreq'] = None
    assert_raises(RuntimeError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['sfreq'] += 10
    assert_raises(RuntimeError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['chs'][1]['ch_name'] = epochs_meg2.info['ch_names'][0]
    epochs_meg2.info._update_redundant()
    assert_raises(RuntimeError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['dev_head_t']['to'] += 1
    assert_raises(ValueError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['dev_head_t']['to'] += 1
    assert_raises(ValueError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.info['expimenter'] = 'foo'
    assert_raises(RuntimeError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.preload = False
    assert_raises(ValueError, add_channels_epochs, [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.times += 0.4
    assert_raises(NotImplementedError, add_channels_epochs,
                  [epochs_meg2, epochs_eeg])

    epochs_meg2 = epochs_meg.copy()
    epochs_meg2.times += 0.5
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
    """Test creating epochs from array."""
    import matplotlib.pyplot as plt
    tempdir = _TempDir()

    # creating
    data = rng.random_sample((10, 20, 300))
    sfreq = 1e3
    ch_names = ['EEG %03d' % (i + 1) for i in range(20)]
    types = ['eeg'] * 20
    info = create_info(ch_names, sfreq, types)
    events = np.c_[np.arange(1, 600, 60),
                   np.zeros(10, int),
                   [1, 2] * 5]
    event_id = {'a': 1, 'b': 2}
    epochs = EpochsArray(data, info, events, tmin, event_id)
    assert_true(str(epochs).startswith('<EpochsArray'))
    # From GH#1963
    assert_raises(ValueError, EpochsArray, data[:-1], info, events, tmin,
                  event_id)
    assert_raises(ValueError, EpochsArray, data, info, events, tmin,
                  dict(a=1))

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
    epochs[0].plot()
    plt.close('all')

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
    assert_equal(len(epochs.drop_log), 10)
    assert_equal(len(epochs.events), len(epochs.selection))

    # baseline
    data = np.ones((10, 20, 300))
    epochs = EpochsArray(data, info, events, event_id=event_id, tmin=-.2,
                         baseline=(None, 0))
    ep_data = epochs.get_data()
    assert_array_equal(ep_data, np.zeros_like(ep_data))

    # one time point
    epochs = EpochsArray(data[:, :, :1], info, events=events,
                         event_id=event_id, tmin=0.)
    assert_allclose(epochs.times, [0.])
    assert_allclose(epochs.get_data(), data[:, :, :1])
    epochs.save(temp_fname)
    epochs_read = read_epochs(temp_fname)
    assert_allclose(epochs_read.times, [0.])
    assert_allclose(epochs_read.get_data(), data[:, :, :1])

    # event as integer (#2435)
    mask = (events[:, 2] == 1)
    data_1 = data[mask]
    events_1 = events[mask]
    epochs = EpochsArray(data_1, info, events=events_1, event_id=1, tmin=-0.2)

    # default events
    epochs = EpochsArray(data_1, info)
    assert_array_equal(epochs.events[:, 0], np.arange(len(data_1)))
    assert_array_equal(epochs.events[:, 1], np.zeros(len(data_1), int))
    assert_array_equal(epochs.events[:, 2], np.ones(len(data_1), int))


def test_concatenate_epochs():
    """Test concatenate epochs."""
    raw, events, picks = _get_data()
    epochs = Epochs(raw=raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, picks=picks)
    epochs2 = epochs.copy()
    epochs_list = [epochs, epochs2]
    epochs_conc = concatenate_epochs(epochs_list)
    assert_array_equal(
        epochs_conc.events[:, 0], np.unique(epochs_conc.events[:, 0]))

    expected_shape = list(epochs.get_data().shape)
    expected_shape[0] *= 2
    expected_shape = tuple(expected_shape)

    assert_equal(epochs_conc.get_data().shape, expected_shape)
    assert_equal(epochs_conc.drop_log, epochs.drop_log * 2)

    epochs2 = epochs.copy()
    epochs2._data = epochs2.get_data()
    epochs2.preload = True
    assert_raises(
        ValueError, concatenate_epochs,
        [epochs, epochs2.copy().drop_channels(epochs2.ch_names[:1])])

    epochs2.times = np.delete(epochs2.times, 1)
    assert_raises(
        ValueError,
        concatenate_epochs, [epochs, epochs2])

    assert_equal(epochs_conc._raw, None)

    # check if baseline is same for all epochs
    epochs2.baseline = (-0.1, None)
    assert_raises(ValueError, concatenate_epochs, [epochs, epochs2])

    # check if dev_head_t is same
    epochs2 = epochs.copy()
    concatenate_epochs([epochs, epochs2])  # should work
    epochs2.info['dev_head_t']['trans'][:3, 3] += 0.0001
    assert_raises(ValueError, concatenate_epochs, [epochs, epochs2])
    assert_raises(TypeError, concatenate_epochs, 'foo')
    assert_raises(TypeError, concatenate_epochs, [epochs, 'foo'])
    epochs2.info['dev_head_t'] = None
    assert_raises(ValueError, concatenate_epochs, [epochs, epochs2])
    epochs.info['dev_head_t'] = None
    concatenate_epochs([epochs, epochs2])  # should work

    # check that different event_id does not work:
    epochs1 = epochs.copy()
    epochs2 = epochs.copy()
    epochs1.event_id = dict(a=1)
    epochs2.event_id = dict(a=2)
    assert_raises(ValueError, concatenate_epochs, [epochs1, epochs2])


def test_add_channels():
    """Test epoch splitting / re-appending channel types."""
    raw, events, picks = _get_data()
    epoch_nopre = Epochs(
        raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
        picks=picks)
    epoch = Epochs(
        raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
        picks=picks, preload=True)
    epoch_eeg = epoch.copy().pick_types(meg=False, eeg=True)
    epoch_meg = epoch.copy().pick_types(meg=True)
    epoch_stim = epoch.copy().pick_types(meg=False, stim=True)
    epoch_eeg_meg = epoch.copy().pick_types(meg=True, eeg=True)
    epoch_new = epoch_meg.copy().add_channels([epoch_eeg, epoch_stim])
    assert_true(all(ch in epoch_new.ch_names
                    for ch in epoch_stim.ch_names + epoch_meg.ch_names))
    epoch_new = epoch_meg.copy().add_channels([epoch_eeg])

    assert_true(ch in epoch_new.ch_names for ch in epoch.ch_names)
    assert_array_equal(epoch_new._data, epoch_eeg_meg._data)
    assert_true(all(ch not in epoch_new.ch_names
                    for ch in epoch_stim.ch_names))

    # Now test errors
    epoch_badsf = epoch_eeg.copy()
    epoch_badsf.info['sfreq'] = 3.1415927
    epoch_eeg = epoch_eeg.crop(-.1, .1)

    assert_raises(AssertionError, epoch_meg.add_channels, [epoch_nopre])
    assert_raises(RuntimeError, epoch_meg.add_channels, [epoch_badsf])
    assert_raises(AssertionError, epoch_meg.add_channels, [epoch_eeg])
    assert_raises(ValueError, epoch_meg.add_channels, [epoch_meg])
    assert_raises(AssertionError, epoch_meg.add_channels, epoch_badsf)


def test_seeg_ecog():
    """Test the compatibility of the Epoch object with SEEG and ECoG data."""
    n_epochs, n_channels, n_times, sfreq = 5, 10, 20, 1000.
    data = np.ones((n_epochs, n_channels, n_times))
    events = np.array([np.arange(n_epochs), [0] * n_epochs, [1] * n_epochs]).T
    pick_dict = dict(meg=False, exclude=[])
    for key in ('seeg', 'ecog'):
        info = create_info(n_channels, sfreq, key)
        epochs = EpochsArray(data, info, events)
        pick_dict.update({key: True})
        picks = pick_types(epochs.info, **pick_dict)
        del pick_dict[key]
        assert_equal(len(picks), n_channels)


def test_default_values():
    """Test default event_id, tmax tmin values are working correctly"""
    raw, events = _get_data()[:2]
    epoch_1 = Epochs(raw, events[:1], preload=True)
    epoch_2 = Epochs(raw, events[:1], tmin=-0.2, tmax=0.5, preload=True)
    assert_equal(hash(epoch_1), hash(epoch_2))


run_tests_if_main()
