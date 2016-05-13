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

from mne import pick_channels, pick_types, Evoked, Epochs, read_events
from mne.epochs import _BaseEpochs
from mne.io.constants import FIFF
from mne.io import (set_eeg_reference, set_bipolar_reference,
                    add_reference_channels)
from mne.io.proj import _has_eeg_average_ref_proj
from mne.io.reference import _apply_reference
from mne.datasets import testing
from mne.io import Raw

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')
eve_fname = op.join(data_dir, 'sample_audvis_trunc_raw-eve.fif')
ave_fname = op.join(data_dir, 'sample_audvis_trunc-ave.fif')


def _test_reference(raw, reref, ref_data, ref_from):
    """Helper function to test whether a reference has been correctly
    applied."""
    # Separate EEG channels from other channel types
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    picks_other = pick_types(raw.info, meg=True, eeg=False, eog=True,
                             stim=True, exclude='bads')

    # Calculate indices of reference channesl
    picks_ref = [raw.ch_names.index(ch) for ch in ref_from]

    # Get data
    if isinstance(raw, Evoked):
        _data = raw.data
        _reref = reref.data
    else:
        _data = raw._data
        _reref = reref._data

    # Check that the ref has been properly computed
    assert_array_equal(ref_data, _data[..., picks_ref, :].mean(-2))

    # Get the raw EEG data and other channel data
    raw_eeg_data = _data[..., picks_eeg, :]
    raw_other_data = _data[..., picks_other, :]

    # Get the rereferenced EEG data
    reref_eeg_data = _reref[..., picks_eeg, :]
    reref_other_data = _reref[..., picks_other, :]

    # Undo rereferencing of EEG channels
    if isinstance(raw, _BaseEpochs):
        unref_eeg_data = reref_eeg_data + ref_data[:, np.newaxis, :]
    else:
        unref_eeg_data = reref_eeg_data + ref_data

    # Check that both EEG data and other data is the same
    assert_allclose(raw_eeg_data, unref_eeg_data, 1e-6, atol=1e-15)
    assert_allclose(raw_other_data, reref_other_data, 1e-6, atol=1e-15)


@testing.requires_testing_data
def test_apply_reference():
    """Test base function for rereferencing"""
    raw = Raw(fif_fname, preload=True)

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
    raw = Raw(fif_fname, preload=False, add_eeg_ref=False)
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

    # Test invalid input
    raw_np = Raw(fif_fname, preload=False)
    assert_raises(RuntimeError, _apply_reference, raw_np, ['EEG 001'])


@testing.requires_testing_data
def test_set_eeg_reference():
    """Test rereference eeg data"""
    raw = Raw(fif_fname, preload=True)
    raw.info['projs'] = []

    # Test setting an average reference
    assert_true(not _has_eeg_average_ref_proj(raw.info['projs']))
    reref, ref_data = set_eeg_reference(raw)
    assert_true(_has_eeg_average_ref_proj(reref.info['projs']))
    assert_true(ref_data is None)

    # Test setting an average reference when one was already present
    with warnings.catch_warnings(record=True):  # weight tables
        reref, ref_data = set_eeg_reference(raw, copy=False)
    assert_true(ref_data is None)

    # Rereference raw data by creating a copy of original data
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'], copy=True)
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(raw, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Test that data is modified in place when copy=False
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'],
                                        copy=False)
    assert_true(raw is reref)


@testing.requires_testing_data
def test_set_bipolar_reference():
    """Test bipolar referencing"""
    raw = Raw(fif_fname, preload=True)
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
    raw = Raw(fif_fname, preload=True)
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

    ref_idx = raw.ch_names.index('Ref')
    ref_data, _ = raw[ref_idx]
    assert_array_equal(ref_data, 0)

    raw = Raw(fif_fname, preload=True)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)

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
    raw = Raw(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
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
    raw = Raw(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
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
    raw = Raw(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
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
    raw = Raw(fif_fname, preload=True)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
    evoked = epochs.average()
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
    raw_np = Raw(fif_fname, preload=False)
    assert_raises(RuntimeError, add_reference_channels, raw_np, ['Ref'])
    assert_raises(ValueError, add_reference_channels, raw, 1)
