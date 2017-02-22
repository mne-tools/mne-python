# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import warnings
import os.path as op
import numpy as np

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_allclose

from mne import (pick_channels, pick_types, Epochs, read_events,
                 set_eeg_reference, set_bipolar_reference,
                 add_reference_channels)
from mne.epochs import BaseEpochs
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne.io.proj import _has_eeg_average_ref_proj, Projection
from mne.io.reference import _apply_reference
from mne.datasets import testing
from mne.utils import run_tests_if_main

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')
eve_fname = op.join(data_dir, 'sample_audvis_trunc_raw-eve.fif')
ave_fname = op.join(data_dir, 'sample_audvis_trunc-ave.fif')


def _test_reference(raw, reref, ref_data, ref_from):
    """Test whether a reference has been correctly applied."""
    # Separate EEG channels from other channel types
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    picks_other = pick_types(raw.info, meg=True, eeg=False, eog=True,
                             stim=True, exclude='bads')

    # Calculate indices of reference channesl
    picks_ref = [raw.ch_names.index(ch) for ch in ref_from]

    # Get data
    _data = raw._data
    _reref = reref._data

    # Check that the ref has been properly computed
    if ref_data is not None:
        assert_array_equal(ref_data, _data[..., picks_ref, :].mean(-2))

    # Get the raw EEG data and other channel data
    raw_eeg_data = _data[..., picks_eeg, :]
    raw_other_data = _data[..., picks_other, :]

    # Get the rereferenced EEG data
    reref_eeg_data = _reref[..., picks_eeg, :]
    reref_other_data = _reref[..., picks_other, :]

    # Check that non-EEG channels are untouched
    assert_allclose(raw_other_data, reref_other_data, 1e-6, atol=1e-15)

    # Undo rereferencing of EEG channels if possible
    if ref_data is not None:
        if isinstance(raw, BaseEpochs):
            unref_eeg_data = reref_eeg_data + ref_data[:, np.newaxis, :]
        else:
            unref_eeg_data = reref_eeg_data + ref_data
        assert_allclose(raw_eeg_data, unref_eeg_data, 1e-6, atol=1e-15)


@testing.requires_testing_data
def test_apply_reference():
    """Test base function for rereferencing."""
    raw = read_raw_fif(fif_fname, preload=True)

    # Rereference raw data by creating a copy of original data
    reref, ref_data = _apply_reference(
        raw.copy(), ref_from=['EEG 001', 'EEG 002'])
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(raw, reref, ref_data, ['EEG 001', 'EEG 002'])

    # The CAR reference projection should have been removed by the function
    assert_true(not _has_eeg_average_ref_proj(reref.info['projs']))

    # Test that disabling the reference does not break anything
    reref, ref_data = _apply_reference(raw, [])
    assert_array_equal(raw._data, reref._data)

    # Test that data is modified in place when copy=False
    reref, ref_data = _apply_reference(raw, ['EEG 001', 'EEG 002'])
    assert_true(raw is reref)

    # Test re-referencing Epochs object
    raw = read_raw_fif(fif_fname, preload=False)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
    reref, ref_data = _apply_reference(
        epochs.copy(), ref_from=['EEG 001', 'EEG 002'])
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(epochs, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Test re-referencing Evoked object
    evoked = epochs.average()
    reref, ref_data = _apply_reference(
        evoked.copy(), ref_from=['EEG 001', 'EEG 002'])
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(evoked, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Referencing needs data to be preloaded
    raw_np = read_raw_fif(fif_fname, preload=False)
    assert_raises(RuntimeError, _apply_reference, raw_np, ['EEG 001'])

    # Test having inactive SSP projections that deal with channels involved
    # during re-referencing
    raw = read_raw_fif(fif_fname, preload=True)
    raw.add_proj(
        Projection(
            active=False,
            data=dict(
                col_names=['EEG 001', 'EEG 002'],
                row_names=None,
                data=[[1, 1]],
                ncol=2,
                nrow=1
            ),
            desc='test',
            kind=1,
        )
    )
    # Projection concerns channels mentioned in projector
    assert_raises(RuntimeError, _apply_reference, raw, ['EEG 001'])

    # Projection does not concern channels mentioned in projector, no error
    _apply_reference(raw, ['EEG 003'], ['EEG 004'])


@testing.requires_testing_data
def test_set_eeg_reference():
    """Test rereference eeg data."""
    raw = read_raw_fif(fif_fname, preload=True)
    raw.info['projs'] = []

    # Test setting an average reference
    assert_true(not _has_eeg_average_ref_proj(raw.info['projs']))
    reref, ref_data = set_eeg_reference(raw)
    assert_true(_has_eeg_average_ref_proj(reref.info['projs']))
    assert_true(not reref.info['projs'][0]['active'])
    assert_true(ref_data is None)
    reref.apply_proj()
    eeg_chans = [raw.ch_names[ch]
                 for ch in pick_types(raw.info, meg=False, eeg=True)]
    _test_reference(raw, reref, ref_data,
                    [ch for ch in eeg_chans if ch not in raw.info['bads']])

    # Test setting an average reference when one was already present
    with warnings.catch_warnings(record=True):
        reref, ref_data = set_eeg_reference(raw, copy=False)
    assert_true(ref_data is None)

    # Test setting an average reference on non-preloaded data
    raw_nopreload = read_raw_fif(fif_fname, preload=False)
    raw_nopreload.info['projs'] = []
    reref, ref_data = set_eeg_reference(raw_nopreload)
    assert_true(_has_eeg_average_ref_proj(reref.info['projs']))
    assert_true(not reref.info['projs'][0]['active'])

    # Rereference raw data by creating a copy of original data
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'], copy=True)
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(raw, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Test that data is modified in place when copy=False
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'],
                                        copy=False)
    assert_true(raw is reref)

    # Test moving from custom to average reference
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'])
    reref, _ = set_eeg_reference(reref)
    assert_true(_has_eeg_average_ref_proj(reref.info['projs']))
    assert_equal(reref.info['custom_ref_applied'], False)

    # When creating an average reference fails, make sure the
    # custom_ref_applied flag remains untouched.
    reref = raw.copy()
    reref.info['custom_ref_applied'] = True
    reref.pick_types(eeg=False)  # Cause making average ref fail
    assert_raises(ValueError, set_eeg_reference, reref)
    assert_true(reref.info['custom_ref_applied'])

    # Test moving from average to custom reference
    reref, ref_data = set_eeg_reference(raw)
    reref, _ = set_eeg_reference(reref, ['EEG 001', 'EEG 002'])
    assert_true(not _has_eeg_average_ref_proj(reref.info['projs']))
    assert_equal(reref.info['custom_ref_applied'], True)


@testing.requires_testing_data
def test_set_bipolar_reference():
    """Test bipolar referencing."""
    raw = read_raw_fif(fif_fname, preload=True)
    raw.apply_proj()

    reref = set_bipolar_reference(raw, 'EEG 001', 'EEG 002', 'bipolar',
                                  {'kind': FIFF.FIFFV_EOG_CH,
                                   'extra': 'some extra value'})
    assert_true(reref.info['custom_ref_applied'])

    # Compare result to a manual calculation
    a = raw.copy().pick_channels(['EEG 001', 'EEG 002'])
    a = a._data[0, :] - a._data[1, :]
    b = reref.copy().pick_channels(['bipolar'])._data[0, :]
    assert_allclose(a, b)

    # Original channels should be replaced by a virtual one
    assert_true('EEG 001' not in reref.ch_names)
    assert_true('EEG 002' not in reref.ch_names)
    assert_true('bipolar' in reref.ch_names)

    # Check channel information
    bp_info = reref.info['chs'][reref.ch_names.index('bipolar')]
    an_info = reref.info['chs'][raw.ch_names.index('EEG 001')]
    for key in bp_info:
        if key == 'loc':
            assert_array_equal(bp_info[key], 0)
        elif key == 'coil_type':
            assert_equal(bp_info[key], FIFF.FIFFV_COIL_EEG_BIPOLAR)
        elif key == 'kind':
            assert_equal(bp_info[key], FIFF.FIFFV_EOG_CH)
        else:
            assert_equal(bp_info[key], an_info[key])
    assert_equal(bp_info['extra'], 'some extra value')

    # Minimalist call
    reref = set_bipolar_reference(raw, 'EEG 001', 'EEG 002')
    assert_true('EEG 001-EEG 002' in reref.ch_names)

    # Set multiple references at once
    reref = set_bipolar_reference(
        raw,
        ['EEG 001', 'EEG 003'],
        ['EEG 002', 'EEG 004'],
        ['bipolar1', 'bipolar2'],
        [{'kind': FIFF.FIFFV_EOG_CH, 'extra': 'some extra value'},
         {'kind': FIFF.FIFFV_EOG_CH, 'extra': 'some extra value'}],
    )
    a = raw.copy().pick_channels(['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004'])
    a = np.array([a._data[0, :] - a._data[1, :],
                  a._data[2, :] - a._data[3, :]])
    b = reref.copy().pick_channels(['bipolar1', 'bipolar2'])._data
    assert_allclose(a, b)

    # Test creating a bipolar reference that doesn't involve EEG channels:
    # it should not set the custom_ref_applied flag
    reref = set_bipolar_reference(raw, 'MEG 0111', 'MEG 0112',
                                  ch_info={'kind': FIFF.FIFFV_MEG_CH})
    assert_true(not reref.info['custom_ref_applied'])
    assert_true('MEG 0111-MEG 0112' in reref.ch_names)

    # Test a battery of invalid inputs
    assert_raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', ['EEG 002', 'EEG 003'], 'bipolar')
    assert_raises(ValueError, set_bipolar_reference, raw,
                  ['EEG 001', 'EEG 002'], 'EEG 003', 'bipolar')
    assert_raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', 'EEG 002', ['bipolar1', 'bipolar2'])
    assert_raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', 'EEG 002', 'bipolar',
                  ch_info=[{'foo': 'bar'}, {'foo': 'bar'}])
    assert_raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', 'EEG 002', ch_name='EEG 003')


def _check_channel_names(inst, ref_names):
    """Check channel names."""
    if isinstance(ref_names, str):
        ref_names = [ref_names]

    # Test that the names of the reference channels are present in `ch_names`
    ref_idx = pick_channels(inst.info['ch_names'], ref_names)
    assert_true(len(ref_idx), len(ref_names))

    # Test that the names of the reference channels are present in the `chs`
    # list
    inst.info._check_consistency()  # Should raise no exceptions


@testing.requires_testing_data
def test_add_reference():
    """Test adding a reference."""
    raw = read_raw_fif(fif_fname, preload=True)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    # check if channel already exists
    assert_raises(ValueError, add_reference_channels,
                  raw, raw.info['ch_names'][0])
    # add reference channel to Raw
    raw_ref = add_reference_channels(raw, 'Ref', copy=True)
    assert_equal(raw_ref._data.shape[0], raw._data.shape[0] + 1)
    assert_array_equal(raw._data[picks_eeg, :], raw_ref._data[picks_eeg, :])
    _check_channel_names(raw_ref, 'Ref')

    orig_nchan = raw.info['nchan']
    raw = add_reference_channels(raw, 'Ref', copy=False)
    assert_array_equal(raw._data, raw_ref._data)
    assert_equal(raw.info['nchan'], orig_nchan + 1)
    _check_channel_names(raw, 'Ref')

    # for Neuromag fif's, the reference electrode location is placed in
    # elements [3:6] of each "data" electrode location
    assert_allclose(raw.info['chs'][-1]['loc'][:3],
                    raw.info['chs'][picks_eeg[0]]['loc'][3:6], 1e-6)

    ref_idx = raw.ch_names.index('Ref')
    ref_data, _ = raw[ref_idx]
    assert_array_equal(ref_data, 0)

    # add reference channel to Raw when no digitization points exist
    raw = read_raw_fif(fif_fname).crop(0, 1).load_data()
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    del raw.info['dig']

    raw_ref = add_reference_channels(raw, 'Ref', copy=True)

    assert_equal(raw_ref._data.shape[0], raw._data.shape[0] + 1)
    assert_array_equal(raw._data[picks_eeg, :], raw_ref._data[picks_eeg, :])
    _check_channel_names(raw_ref, 'Ref')

    orig_nchan = raw.info['nchan']
    raw = add_reference_channels(raw, 'Ref', copy=False)
    assert_array_equal(raw._data, raw_ref._data)
    assert_equal(raw.info['nchan'], orig_nchan + 1)
    _check_channel_names(raw, 'Ref')

    # Test adding an existing channel as reference channel
    assert_raises(ValueError, add_reference_channels, raw,
                  raw.info['ch_names'][0])

    # add two reference channels to Raw
    raw_ref = add_reference_channels(raw, ['M1', 'M2'], copy=True)
    _check_channel_names(raw_ref, ['M1', 'M2'])
    assert_equal(raw_ref._data.shape[0], raw._data.shape[0] + 2)
    assert_array_equal(raw._data[picks_eeg, :], raw_ref._data[picks_eeg, :])
    assert_array_equal(raw_ref._data[-2:, :], 0)

    raw = add_reference_channels(raw, ['M1', 'M2'], copy=False)
    _check_channel_names(raw, ['M1', 'M2'])
    ref_idx = raw.ch_names.index('M1')
    ref_idy = raw.ch_names.index('M2')
    ref_data, _ = raw[[ref_idx, ref_idy]]
    assert_array_equal(ref_data, 0)

    # add reference channel to epochs
    raw = read_raw_fif(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
    # default: proj=True, after which adding a Ref channel is prohibited
    assert_raises(RuntimeError, add_reference_channels, epochs, 'Ref')

    # create epochs in delayed mode, allowing removal of CAR when re-reffing
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True, proj='delayed')
    epochs_ref = add_reference_channels(epochs, 'Ref', copy=True)

    assert_equal(epochs_ref._data.shape[1], epochs._data.shape[1] + 1)
    _check_channel_names(epochs_ref, 'Ref')
    ref_idx = epochs_ref.ch_names.index('Ref')
    ref_data = epochs_ref.get_data()[:, ref_idx, :]
    assert_array_equal(ref_data, 0)
    picks_eeg = pick_types(epochs.info, meg=False, eeg=True)
    assert_array_equal(epochs.get_data()[:, picks_eeg, :],
                       epochs_ref.get_data()[:, picks_eeg, :])

    # add two reference channels to epochs
    raw = read_raw_fif(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    # create epochs in delayed mode, allowing removal of CAR when re-reffing
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True, proj='delayed')
    with warnings.catch_warnings(record=True):  # multiple set zero
        epochs_ref = add_reference_channels(epochs, ['M1', 'M2'], copy=True)
    assert_equal(epochs_ref._data.shape[1], epochs._data.shape[1] + 2)
    _check_channel_names(epochs_ref, ['M1', 'M2'])
    ref_idx = epochs_ref.ch_names.index('M1')
    ref_idy = epochs_ref.ch_names.index('M2')
    assert_equal(epochs_ref.info['chs'][ref_idx]['ch_name'], 'M1')
    assert_equal(epochs_ref.info['chs'][ref_idy]['ch_name'], 'M2')
    ref_data = epochs_ref.get_data()[:, [ref_idx, ref_idy], :]
    assert_array_equal(ref_data, 0)
    picks_eeg = pick_types(epochs.info, meg=False, eeg=True)
    assert_array_equal(epochs.get_data()[:, picks_eeg, :],
                       epochs_ref.get_data()[:, picks_eeg, :])

    # add reference channel to evoked
    raw = read_raw_fif(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    # create epochs in delayed mode, allowing removal of CAR when re-reffing
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True, proj='delayed')
    evoked = epochs.average()
    evoked_ref = add_reference_channels(evoked, 'Ref', copy=True)
    assert_equal(evoked_ref.data.shape[0], evoked.data.shape[0] + 1)
    _check_channel_names(evoked_ref, 'Ref')
    ref_idx = evoked_ref.ch_names.index('Ref')
    ref_data = evoked_ref.data[ref_idx, :]
    assert_array_equal(ref_data, 0)
    picks_eeg = pick_types(evoked.info, meg=False, eeg=True)
    assert_array_equal(evoked.data[picks_eeg, :],
                       evoked_ref.data[picks_eeg, :])

    # add two reference channels to evoked
    raw = read_raw_fif(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    # create epochs in delayed mode, allowing removal of CAR when re-reffing
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True, proj='delayed')
    evoked = epochs.average()
    with warnings.catch_warnings(record=True):  # multiple set zero
        evoked_ref = add_reference_channels(evoked, ['M1', 'M2'], copy=True)
    assert_equal(evoked_ref.data.shape[0], evoked.data.shape[0] + 2)
    _check_channel_names(evoked_ref, ['M1', 'M2'])
    ref_idx = evoked_ref.ch_names.index('M1')
    ref_idy = evoked_ref.ch_names.index('M2')
    ref_data = evoked_ref.data[[ref_idx, ref_idy], :]
    assert_array_equal(ref_data, 0)
    picks_eeg = pick_types(evoked.info, meg=False, eeg=True)
    assert_array_equal(evoked.data[picks_eeg, :],
                       evoked_ref.data[picks_eeg, :])

    # Test invalid inputs
    raw_np = read_raw_fif(fif_fname, preload=False)
    assert_raises(RuntimeError, add_reference_channels, raw_np, ['Ref'])
    assert_raises(ValueError, add_reference_channels, raw, 1)

run_tests_if_main()
