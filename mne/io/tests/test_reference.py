# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import itertools
import os.path as op
import numpy as np

from numpy.testing import assert_array_equal, assert_allclose, assert_equal
import pytest

from mne import (pick_channels, pick_types, Epochs, read_events,
                 set_eeg_reference, set_bipolar_reference,
                 add_reference_channels, create_info, make_sphere_model,
                 make_forward_solution, setup_volume_source_space,
                 pick_channels_forward, read_evokeds)
from mne.epochs import BaseEpochs
from mne.fixes import nullcontext
from mne.io import RawArray, read_raw_fif
from mne.io.constants import FIFF
from mne.io.proj import _has_eeg_average_ref_proj, Projection
from mne.io.reference import _apply_reference
from mne.datasets import testing
from mne.utils import run_tests_if_main, catch_logging

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')
eve_fname = op.join(data_dir, 'sample_audvis_trunc_raw-eve.fif')
ave_fname = op.join(data_dir, 'sample_audvis-ave.fif')


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
    assert (reref.info['custom_ref_applied'])
    _test_reference(raw, reref, ref_data, ['EEG 001', 'EEG 002'])

    # The CAR reference projection should have been removed by the function
    assert (not _has_eeg_average_ref_proj(reref.info['projs']))

    # Test that data is modified in place when copy=False
    reref, ref_data = _apply_reference(raw, ['EEG 001', 'EEG 002'])
    assert (raw is reref)

    # Test that disabling the reference does not change anything
    reref, ref_data = _apply_reference(raw.copy(), [])
    assert_array_equal(raw._data, reref._data)

    # Test re-referencing Epochs object
    raw = read_raw_fif(fif_fname, preload=False)
    events = read_events(eve_fname)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    epochs = Epochs(raw, events=events, event_id=1, tmin=-0.2, tmax=0.5,
                    picks=picks_eeg, preload=True)
    reref, ref_data = _apply_reference(
        epochs.copy(), ref_from=['EEG 001', 'EEG 002'])
    assert (reref.info['custom_ref_applied'])
    _test_reference(epochs, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Test re-referencing Evoked object
    evoked = epochs.average()
    reref, ref_data = _apply_reference(
        evoked.copy(), ref_from=['EEG 001', 'EEG 002'])
    assert (reref.info['custom_ref_applied'])
    _test_reference(evoked, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Referencing needs data to be preloaded
    raw_np = read_raw_fif(fif_fname, preload=False)
    pytest.raises(RuntimeError, _apply_reference, raw_np, ['EEG 001'])

    # Test having inactive SSP projections that deal with channels involved
    # during re-referencing
    raw = read_raw_fif(fif_fname, preload=True)
    raw.add_proj(
        Projection(
            active=False,
            data=dict(
                col_names=['EEG 001', 'EEG 002'],
                row_names=None,
                data=np.array([[1, 1]]),
                ncol=2,
                nrow=1
            ),
            desc='test',
            kind=1,
        )
    )
    # Projection concerns channels mentioned in projector
    with pytest.raises(RuntimeError, match='Inactive signal space'):
        _apply_reference(raw, ['EEG 001'])

    # Projection does not concern channels mentioned in projector, no error
    _apply_reference(raw, ['EEG 003'], ['EEG 004'])

    # CSD cannot be rereferenced
    raw.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
    with pytest.raises(RuntimeError, match="Cannot set.* type 'CSD'"):
        raw.set_eeg_reference()


@testing.requires_testing_data
def test_set_eeg_reference():
    """Test rereference eeg data."""
    raw = read_raw_fif(fif_fname, preload=True)
    raw.info['projs'] = []

    # Test setting an average reference projection
    assert (not _has_eeg_average_ref_proj(raw.info['projs']))
    reref, ref_data = set_eeg_reference(raw, projection=True)
    assert (_has_eeg_average_ref_proj(reref.info['projs']))
    assert (not reref.info['projs'][0]['active'])
    assert (ref_data is None)
    reref.apply_proj()
    eeg_chans = [raw.ch_names[ch]
                 for ch in pick_types(raw.info, meg=False, eeg=True)]
    _test_reference(raw, reref, ref_data,
                    [ch for ch in eeg_chans if ch not in raw.info['bads']])

    # Test setting an average reference when one was already present
    with pytest.warns(RuntimeWarning, match='untouched'):
        reref, ref_data = set_eeg_reference(raw, copy=False, projection=True)
    assert ref_data is None

    # Test setting an average reference on non-preloaded data
    raw_nopreload = read_raw_fif(fif_fname, preload=False)
    raw_nopreload.info['projs'] = []
    reref, ref_data = set_eeg_reference(raw_nopreload, projection=True)
    assert _has_eeg_average_ref_proj(reref.info['projs'])
    assert not reref.info['projs'][0]['active']

    # Rereference raw data by creating a copy of original data
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'], copy=True)
    assert reref.info['custom_ref_applied']
    _test_reference(raw, reref, ref_data, ['EEG 001', 'EEG 002'])

    # Test that data is modified in place when copy=False
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'],
                                        copy=False)
    assert raw is reref

    # Test moving from custom to average reference
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'])
    reref, _ = set_eeg_reference(reref, projection=True)
    assert _has_eeg_average_ref_proj(reref.info['projs'])
    assert not reref.info['custom_ref_applied']

    # When creating an average reference fails, make sure the
    # custom_ref_applied flag remains untouched.
    reref = raw.copy()
    reref.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    reref.pick_types(meg=True, eeg=False)  # Cause making average ref fail
    pytest.raises(ValueError, set_eeg_reference, reref, projection=True)
    assert reref.info['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_ON

    # Test moving from average to custom reference
    reref, ref_data = set_eeg_reference(raw, projection=True)
    reref, _ = set_eeg_reference(reref, ['EEG 001', 'EEG 002'])
    assert not _has_eeg_average_ref_proj(reref.info['projs'])
    assert len(reref.info['projs']) == 0
    assert reref.info['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_ON

    # Test that disabling the reference does not change the data
    assert _has_eeg_average_ref_proj(raw.info['projs'])
    reref, _ = set_eeg_reference(raw, [])
    assert_array_equal(raw._data, reref._data)
    assert not _has_eeg_average_ref_proj(reref.info['projs'])

    # make sure ref_channels=[] removes average reference projectors
    assert _has_eeg_average_ref_proj(raw.info['projs'])
    reref, _ = set_eeg_reference(raw, [])
    assert (not _has_eeg_average_ref_proj(reref.info['projs']))

    # Test that average reference gives identical results when calculated
    # via SSP projection (projection=True) or directly (projection=False)
    raw.info['projs'] = []
    reref_1, _ = set_eeg_reference(raw.copy(), projection=True)
    reref_1.apply_proj()
    reref_2, _ = set_eeg_reference(raw.copy(), projection=False)
    assert_allclose(reref_1._data, reref_2._data, rtol=1e-6, atol=1e-15)

    # Test average reference without projection
    reref, ref_data = set_eeg_reference(raw.copy(), ref_channels="average",
                                        projection=False)
    _test_reference(raw, reref, ref_data, eeg_chans)

    with pytest.raises(ValueError, match='supported for ref_channels="averag'):
        set_eeg_reference(raw, [], True, True)
    with pytest.raises(ValueError, match='supported for ref_channels="averag'):
        set_eeg_reference(raw, ['EEG 001'], True, True)


@pytest.mark.parametrize('ch_type', ('auto', 'ecog'))
def test_set_eeg_reference_ch_type(ch_type):
    """Test setting EEG reference for ECoG."""
    # gh-6454
    rng = np.random.RandomState(0)
    data = rng.randn(3, 1000)
    raw = RawArray(data, create_info(3, 1000., ['ecog'] * 2 + ['misc']))
    with catch_logging() as log:
        reref, ref_data = set_eeg_reference(raw.copy(), ch_type=ch_type,
                                            verbose=True)
    assert 'Applying a custom ECoG' in log.getvalue()
    assert reref.info['custom_ref_applied']  # gh-7350
    _test_reference(raw, reref, ref_data, ['0', '1'])
    with pytest.raises(ValueError, match='No channels supplied'):
        set_eeg_reference(raw, ch_type='eeg')


@testing.requires_testing_data
def test_set_eeg_reference_rest():
    """Test setting a REST reference."""
    raw = read_raw_fif(fif_fname).crop(0, 1).pick_types(
        meg=False, eeg=True, exclude=()).load_data()
    raw.info['bads'] = ['EEG 057']  # should be excluded
    same = [raw.ch_names.index(raw.info['bads'][0])]
    picks = np.setdiff1d(np.arange(len(raw.ch_names)), same)
    trans = None
    sphere = make_sphere_model('auto', 'auto', raw.info)
    src = setup_volume_source_space(pos=20., sphere=sphere, exclude=30.)
    assert src[0]['nuse'] == 223  # low but fast
    fwd = make_forward_solution(raw.info, trans, src, sphere)
    orig_data = raw.get_data()
    avg_data = raw.copy().set_eeg_reference('average').get_data()
    assert_array_equal(avg_data[same], orig_data[same])  # not processed
    raw.set_eeg_reference('REST', forward=fwd)
    rest_data = raw.get_data()
    assert_array_equal(rest_data[same], orig_data[same])
    # should be more similar to an avg ref than nose ref
    orig_corr = np.corrcoef(rest_data[picks].ravel(),
                            orig_data[picks].ravel())[0, 1]
    avg_corr = np.corrcoef(rest_data[picks].ravel(),
                           avg_data[picks].ravel())[0, 1]
    assert -0.6 < orig_corr < -0.5
    assert 0.1 < avg_corr < 0.2
    # and applying an avg ref after should work
    avg_after = raw.set_eeg_reference('average').get_data()
    assert_allclose(avg_after, avg_data, atol=1e-12)
    with pytest.raises(TypeError, match='forward when ref_channels="REST"'):
        raw.set_eeg_reference('REST')
    fwd_bad = pick_channels_forward(fwd, raw.ch_names[:-1])
    with pytest.raises(ValueError, match='Missing channels'):
        raw.set_eeg_reference('REST', forward=fwd_bad)
    # compare to FieldTrip
    evoked = read_evokeds(ave_fname, baseline=(None, 0))[0]
    evoked.info['bads'] = []
    evoked.pick_types(meg=False, eeg=True, exclude=())
    assert len(evoked.ch_names) == 60
    # Data obtained from FieldTrip with something like (after evoked.save'ing
    # then scipy.io.savemat'ing fwd['sol']['data']):
    # dat = ft_read_data('ft-ave.fif');
    # load('leadfield.mat', 'G');
    # dat_ref = ft_preproc_rereference(dat, 'all', 'rest', true, G);
    # sprintf('%g ', dat_ref(:, 171));
    want = np.array('-3.3265e-05 -3.2419e-05 -3.18758e-05 -3.24079e-05 -3.39801e-05 -3.40573e-05 -3.24163e-05 -3.26896e-05 -3.33814e-05 -3.54734e-05 -3.51289e-05 -3.53229e-05 -3.51532e-05 -3.53149e-05 -3.4505e-05 -3.03462e-05 -2.81848e-05 -3.08895e-05 -3.27158e-05 -3.4605e-05 -3.47728e-05 -3.2459e-05 -3.06552e-05 -2.53255e-05 -2.69671e-05 -2.83425e-05 -3.12836e-05 -3.30965e-05 -3.34099e-05 -3.32766e-05 -3.32256e-05 -3.36385e-05 -3.20796e-05 -2.7108e-05 -2.47054e-05 -2.49589e-05 -2.7382e-05 -3.09774e-05 -3.12003e-05 -3.1246e-05 -3.07572e-05 -2.64942e-05 -2.25505e-05 -2.67194e-05 -2.86e-05 -2.94903e-05 -2.96249e-05 -2.92653e-05 -2.86472e-05 -2.81016e-05 -2.69737e-05 -2.48076e-05 -3.00473e-05 -2.73404e-05 -2.60153e-05 -2.41608e-05 -2.61937e-05 -2.5539e-05 -2.47104e-05 -2.35194e-05'.split(' '), float)  # noqa: E501
    norm = np.linalg.norm(want)
    idx = np.argmin(np.abs(evoked.times - 0.083))
    assert idx == 170
    old = evoked.data[:, idx].ravel()
    exp_var = 1 - np.linalg.norm(want - old) / norm
    assert 0.006 < exp_var < 0.008
    evoked.set_eeg_reference('REST', forward=fwd)
    exp_var_old = 1 - np.linalg.norm(evoked.data[:, idx] - old) / norm
    assert 0.005 < exp_var_old <= 0.009
    exp_var = 1 - np.linalg.norm(evoked.data[:, idx] - want) / norm
    assert 0.995 < exp_var <= 1


@testing.requires_testing_data
def test_set_bipolar_reference():
    """Test bipolar referencing."""
    raw = read_raw_fif(fif_fname, preload=True)
    raw.apply_proj()

    ch_info = {'kind': FIFF.FIFFV_EOG_CH, 'extra': 'some extra value'}
    with pytest.raises(KeyError, match='key errantly present'):
        set_bipolar_reference(raw, 'EEG 001', 'EEG 002', 'bipolar', ch_info)
    ch_info.pop('extra')
    reref = set_bipolar_reference(
        raw, 'EEG 001', 'EEG 002', 'bipolar', ch_info)
    assert (reref.info['custom_ref_applied'])

    # Compare result to a manual calculation
    a = raw.copy().pick_channels(['EEG 001', 'EEG 002'])
    a = a._data[0, :] - a._data[1, :]
    b = reref.copy().pick_channels(['bipolar'])._data[0, :]
    assert_allclose(a, b)

    # Original channels should be replaced by a virtual one
    assert ('EEG 001' not in reref.ch_names)
    assert ('EEG 002' not in reref.ch_names)
    assert ('bipolar' in reref.ch_names)

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

    # Minimalist call
    reref = set_bipolar_reference(raw, 'EEG 001', 'EEG 002')
    assert ('EEG 001-EEG 002' in reref.ch_names)

    # Minimalist call with twice the same anode
    reref = set_bipolar_reference(raw,
                                  ['EEG 001', 'EEG 001', 'EEG 002'],
                                  ['EEG 002', 'EEG 003', 'EEG 003'])
    assert ('EEG 001-EEG 002' in reref.ch_names)
    assert ('EEG 001-EEG 003' in reref.ch_names)

    # Set multiple references at once
    reref = set_bipolar_reference(
        raw,
        ['EEG 001', 'EEG 003'],
        ['EEG 002', 'EEG 004'],
        ['bipolar1', 'bipolar2'],
        [{'kind': FIFF.FIFFV_EOG_CH},
         {'kind': FIFF.FIFFV_EOG_CH}],
    )
    a = raw.copy().pick_channels(['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004'])
    a = np.array([a._data[0, :] - a._data[1, :],
                  a._data[2, :] - a._data[3, :]])
    b = reref.copy().pick_channels(['bipolar1', 'bipolar2'])._data
    assert_allclose(a, b)

    # Test creating a bipolar reference that doesn't involve EEG channels:
    # it should not set the custom_ref_applied flag
    reref = set_bipolar_reference(raw, 'MEG 0111', 'MEG 0112',
                                  ch_info={'kind': FIFF.FIFFV_MEG_CH},
                                  verbose='error')
    assert (not reref.info['custom_ref_applied'])
    assert ('MEG 0111-MEG 0112'[:15] in reref.ch_names)

    # Test a battery of invalid inputs
    pytest.raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', ['EEG 002', 'EEG 003'], 'bipolar')
    pytest.raises(ValueError, set_bipolar_reference, raw,
                  ['EEG 001', 'EEG 002'], 'EEG 003', 'bipolar')
    pytest.raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', 'EEG 002', ['bipolar1', 'bipolar2'])
    pytest.raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', 'EEG 002', 'bipolar',
                  ch_info=[{'foo': 'bar'}, {'foo': 'bar'}])
    pytest.raises(ValueError, set_bipolar_reference, raw,
                  'EEG 001', 'EEG 002', ch_name='EEG 003')


def _check_channel_names(inst, ref_names):
    """Check channel names."""
    if isinstance(ref_names, str):
        ref_names = [ref_names]

    # Test that the names of the reference channels are present in `ch_names`
    ref_idx = pick_channels(inst.info['ch_names'], ref_names)
    assert len(ref_idx) == len(ref_names)

    # Test that the names of the reference channels are present in the `chs`
    # list
    inst.info._check_consistency()  # Should raise no exceptions


@testing.requires_testing_data
def test_add_reference():
    """Test adding a reference."""
    raw = read_raw_fif(fif_fname, preload=True)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)
    # check if channel already exists
    pytest.raises(ValueError, add_reference_channels,
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
    pytest.raises(ValueError, add_reference_channels, raw,
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
    pytest.raises(RuntimeError, add_reference_channels, epochs, 'Ref')

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
    with pytest.warns(RuntimeWarning, match='ignored .set to zero.'):
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
    with pytest.warns(RuntimeWarning, match='ignored .set to zero.'):
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
    raw = read_raw_fif(fif_fname, preload=False)
    with pytest.raises(RuntimeError, match='loaded'):
        add_reference_channels(raw, ['Ref'])
    raw.load_data()
    with pytest.raises(ValueError, match='Channel.*already.*'):
        add_reference_channels(raw, raw.ch_names[:1])
    with pytest.raises(TypeError, match='instance of'):
        add_reference_channels(raw, 1)


@pytest.mark.parametrize('n_ref', (1, 2))
def test_add_reorder(n_ref):
    """Test that a reference channel can be added and then data reordered."""
    # gh-8300
    raw = read_raw_fif(raw_fname).crop(0, 0.1).del_proj().pick('eeg')
    assert len(raw.ch_names) == 60
    chs = ['EEG %03d' % (60 + ii) for ii in range(1, n_ref)] + ['EEG 000']
    with pytest.raises(RuntimeError, match='preload'):
        with pytest.warns(None):  # ignore multiple warning
            add_reference_channels(raw, chs, copy=False)
    raw.load_data()
    if n_ref == 1:
        ctx = nullcontext()
    else:
        assert n_ref == 2
        ctx = pytest.warns(RuntimeWarning, match='locations of multiple')
    with ctx:
        add_reference_channels(raw, chs, copy=False)
    data = raw.get_data()
    assert_array_equal(data[-1], 0.)
    assert raw.ch_names[-n_ref:] == chs
    raw.reorder_channels(raw.ch_names[-1:] + raw.ch_names[:-1])
    assert raw.ch_names == ['EEG %03d' % ii for ii in range(60 + n_ref)]
    data_new = raw.get_data()
    data_new = np.concatenate([data_new[1:], data_new[:1]])
    assert_allclose(data, data_new)


def test_bipolar_combinations():
    """Test bipolar channel generation."""
    ch_names = ['CH' + str(ni + 1) for ni in range(10)]
    info = create_info(
        ch_names=ch_names, sfreq=1000., ch_types=['eeg'] * len(ch_names))
    raw_data = np.random.randn(len(ch_names), 1000)
    raw = RawArray(raw_data, info)

    def _check_bipolar(raw_test, ch_a, ch_b):
        picks = [raw_test.ch_names.index(ch_a + '-' + ch_b)]
        get_data_res = raw_test.get_data(picks=picks)[0, :]
        manual_a = raw_data[ch_names.index(ch_a), :]
        manual_b = raw_data[ch_names.index(ch_b), :]
        assert_array_equal(get_data_res, manual_a - manual_b)

    # test classic EOG/ECG bipolar reference (only two channels per pair).
    raw_test = set_bipolar_reference(raw, ['CH2'], ['CH1'], copy=True)
    _check_bipolar(raw_test, 'CH2', 'CH1')

    # test all combinations.
    a_channels, b_channels = zip(*itertools.combinations(ch_names, 2))
    a_channels, b_channels = list(a_channels), list(b_channels)
    raw_test = set_bipolar_reference(raw, a_channels, b_channels, copy=True)
    for ch_a, ch_b in zip(a_channels, b_channels):
        _check_bipolar(raw_test, ch_a, ch_b)
    # check if reference channels have been dropped.
    assert (len(raw_test.ch_names) == len(a_channels))

    raw_test = set_bipolar_reference(
        raw, a_channels, b_channels, drop_refs=False, copy=True)
    # check if reference channels have been kept correctly.
    assert (len(raw_test.ch_names) == len(a_channels) + len(ch_names))
    for idx, ch_label in enumerate(ch_names):
        manual_ch = raw_data[idx, :]
        assert_array_equal(
            raw_test._data[raw_test.ch_names.index(ch_label), :], manual_ch)

    # test bipolars with a channel in both list (anode & cathode).
    raw_test = set_bipolar_reference(
        raw, ['CH2', 'CH1'], ['CH1', 'CH2'], copy=True)
    _check_bipolar(raw_test, 'CH2', 'CH1')
    _check_bipolar(raw_test, 'CH1', 'CH2')


run_tests_if_main()
