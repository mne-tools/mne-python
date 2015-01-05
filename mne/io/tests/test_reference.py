# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import warnings
import numpy as np

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_allclose

from mne import pick_types, Evoked, Epochs, read_events
from mne.io.constants import FIFF
from mne.io import set_eeg_reference, set_bipolar_reference
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
    '''Helper function to test whether a reference has been correctly
    applied'''
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
    if isinstance(raw, Epochs):
        unref_eeg_data = reref_eeg_data + np.tile(ref_data[:, np.newaxis, :],
                                                  (1, len(picks_eeg), 1))
    else:
        unref_eeg_data = reref_eeg_data + ref_data

    # Check that both EEG data and other data is the same
    assert_allclose(raw_eeg_data, unref_eeg_data, 1e-6, atol=1e-15)
    assert_allclose(raw_other_data, reref_other_data, 1e-6, atol=1e-15)

@testing.requires_testing_data
def test_apply_reference():
    '''Test base function for rereferencing'''
    raw = Raw(fif_fname, preload=True, add_eeg_ref=True)


    # Rereference raw data by creating a copy of original data
    reref, ref_data = _apply_reference(raw, ref_from=['EEG 001', 'EEG 002'],
                                       copy=True)
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(raw, reref, ref_data, ['EEG 001', 'EEG 002'])

    # The CAR reference projection should have been removed by the function
    assert_true(not _has_eeg_average_ref_proj(reref.info['projs']))

    # Test that disabling the reference does not break anything
    reref, ref_data = _apply_reference(raw, [])
    assert_array_equal(raw._data, reref._data)

    # Test that data is modified in place when copy=False
    reref, ref_data = _apply_reference(raw, ['EEG 001', 'EEG 002'],
                                       copy=False)
    assert_true(raw is reref)

    # Test re-referencing Epochs object
    raw = Raw(fif_fname, preload=False, add_eeg_ref=False)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
    reref, ref_data = _apply_reference(epochs, ref_from=['EEG 001', 'EEG 002'],
                                       copy=True)
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(epochs, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Test re-referencing Evoked object
    evoked = epochs.average()
    reref, ref_data = _apply_reference(evoked, ref_from=['EEG 001', 'EEG 002'],
                                       copy=True)
    assert_true(reref.info['custom_ref_applied'])
    _test_reference(evoked, reref, ref_data, ['EEG 001', 'EEG 002'])


@testing.requires_testing_data
def test_set_eeg_reference():
    '''Test rereference eeg data'''
    raw = Raw(fif_fname, preload=True)

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
    '''Test bipolar referencing'''
    raw = Raw(fif_fname, preload=True)
    reref = set_bipolar_reference(raw, 'EEG 001', 'EEG 002', 'bipolar',
                                  {'kind': FIFF.FIFFV_EOG_CH,
                                   'extra': 'some extra value'})
    assert_true(reref.info['custom_ref_applied'])

    # Compare result to a manual calculation
    a = raw.pick_channels(['EEG 001', 'EEG 002'], copy=True)
    a = a._data[0, :] - a._data[1, :]
    b = reref.pick_channels(['bipolar'], copy=True)._data[0, :]
    assert_allclose(a, b)

    # Original channels should be replaced by a virtual one
    assert_true('EEG 001' not in reref.ch_names)
    assert_true('EEG 002' not in reref.ch_names)
    assert_true('bipolar' in reref.ch_names)

    # Check channel information
    bp_info = reref.info['chs'][reref.ch_names.index('bipolar')]
    an_info = reref.info['chs'][raw.ch_names.index('EEG 001')]
    for key in bp_info:
        if key == 'loc' or key == 'eeg_loc':
            assert_array_equal(bp_info[key], 0)
        elif key == 'coil_type':
            assert_equal(bp_info[key], FIFF.FIFFV_COIL_EEG_BIPOLAR)
        elif key == 'kind':
            assert_equal(bp_info[key], FIFF.FIFFV_EOG_CH)
        else:
            assert_equal(bp_info[key], an_info[key])
    assert_equal(bp_info['extra'], 'some extra value')

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
