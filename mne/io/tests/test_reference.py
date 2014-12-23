# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import warnings

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_allclose

from mne.io.constants import FIFF
from mne.io import set_eeg_reference, set_bipolar_reference
from mne.io.proj import _has_eeg_average_ref_proj
from mne.io.reference import _apply_reference
from mne.datasets import testing
from mne.io import Raw
from mne import pick_types

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_apply_reference():
    """ Test base function for rereferencing"""
    raw = Raw(fif_fname, preload=True, add_eeg_ref=True)

    # Separate EEG channels from other channel types
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    picks_other = pick_types(raw.info, meg=True, eeg=False, eog=True,
                             stim=True, exclude='bads')

    # Rereference raw data by creating a copy of original data
    reref, ref_data = _apply_reference(raw, ref_from=['EEG 001', 'EEG 002'],
                                       copy=True)
    assert_true(reref.info['custom_ref_applied'])

    # The CAR reference projection should have been removed by the function
    assert_true(not _has_eeg_average_ref_proj(reref.info['projs']))

    # Check that the ref has been properly computed
    ref_ch_idx = [raw.ch_names.index(ch) for ch in ['EEG 001', 'EEG 002']]
    assert_array_equal(ref_data, raw[ref_ch_idx, :][0].mean(0))

    # Get the raw EEG data and other channel data
    raw_eeg_data = raw[picks_eeg][0]
    raw_other_data = raw[picks_other][0]

    # Get the rereferenced EEG data and channel other
    reref_eeg_data = reref[picks_eeg][0]
    # Undo rereferencing of EEG channels
    unref_eeg_data = reref_eeg_data + ref_data
    reref_other_data = reref[picks_other][0]

    # Check that both EEG data and other data is the same
    assert_allclose(raw_eeg_data, unref_eeg_data, 1e-6, atol=1e-15)
    assert_allclose(raw_other_data, reref_other_data, 1e-6, atol=1e-15)

    # Test that disabling the reference does not break anything
    reref, ref_data = _apply_reference(raw, [])

    # Test that data is modified in place when copy=False
    reref, ref_data = _apply_reference(raw, ['EEG 001', 'EEG 002'],
                                       copy=False)
    assert_true(raw is reref)


@testing.requires_testing_data
def test_set_eeg_reference():
    """ Test rereference eeg data"""
    raw = Raw(fif_fname, preload=True)

    # Rereference raw data by creating a copy of original data
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'], copy=True)
    assert_true(reref.info['custom_ref_applied'])

    # Separate EEG channels from other channel types
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    picks_other = pick_types(raw.info, meg=True, eeg=False, eog=True,
                             stim=True, exclude='bads')

    # Get the raw EEG data and other channel data
    raw_eeg_data = raw[picks_eeg][0]
    raw_other_data = raw[picks_other][0]

    # Get the rereferenced EEG data and channel other
    reref_eeg_data = reref[picks_eeg][0]
    # Undo rereferencing of EEG channels
    unref_eeg_data = reref_eeg_data + ref_data
    reref_other_data = reref[picks_other][0]

    # Check that both EEG data and other data is the same
    assert_allclose(raw_eeg_data, unref_eeg_data, 1e-6, atol=1e-15)
    assert_allclose(raw_other_data, reref_other_data, 1e-6, atol=1e-15)

    # Test that data is modified in place when copy=False
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'],
                                        copy=False)
    assert_true(raw is reref)


@testing.requires_testing_data
def test_set_bipolar_reference():
    """ Test bipolar referencing"""
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
