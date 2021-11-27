# Author: Daniel G Wakeman <dwakeman@nmr.mgh.harvard.edu>
#         Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

import os.path as op

from copy import deepcopy
from functools import partial

import pytest
import numpy as np
from scipy.io import savemat
from numpy.testing import assert_array_equal, assert_equal, assert_allclose

from mne.channels import (rename_channels, read_ch_adjacency, combine_channels,
                          find_ch_adjacency, make_1020_channel_selections,
                          read_custom_montage, equalize_channels,
                          make_standard_montage)
from mne.channels.channels import (_ch_neighbor_adjacency,
                                   _compute_ch_adjacency)
from mne.io import (read_info, read_raw_fif, read_raw_ctf, read_raw_bti,
                    read_raw_eeglab, read_raw_kit, RawArray)
from mne.io.constants import FIFF
from mne import (pick_types, pick_channels, EpochsArray, EvokedArray,
                 make_ad_hoc_cov, create_info, read_events, Epochs)
from mne.datasets import testing

io_dir = op.join(op.dirname(__file__), '..', '..', 'io')
base_dir = op.join(io_dir, 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
eve_fname = op.join(base_dir, 'test-eve.fif')
fname_kit_157 = op.join(io_dir, 'kit', 'tests', 'data', 'test.sqd')


@pytest.mark.parametrize('preload', (True, False))
@pytest.mark.parametrize('proj', (True, False))
def test_reorder_channels(preload, proj):
    """Test reordering of channels."""
    raw = read_raw_fif(raw_fname).crop(0, 0.1).del_proj()
    if proj:  # a no-op but should test it
        raw._projector = np.eye(len(raw.ch_names))
    if preload:
        raw.load_data()
    # with .reorder_channels
    if proj and not preload:
        with pytest.raises(RuntimeError, match='load data'):
            raw.copy().reorder_channels(raw.ch_names[::-1])
        return
    raw_new = raw.copy().reorder_channels(raw.ch_names[::-1])
    assert raw_new.ch_names == raw.ch_names[::-1]
    if proj:
        assert_allclose(raw_new._projector, raw._projector, atol=1e-12)
    else:
        assert raw._projector is None
        assert raw_new._projector is None
    assert_array_equal(raw[:][0], raw_new[:][0][::-1])
    raw_new.reorder_channels(raw_new.ch_names[::-1][1:-1])
    raw.drop_channels(raw.ch_names[:1] + raw.ch_names[-1:])
    assert_array_equal(raw[:][0], raw_new[:][0])
    with pytest.raises(ValueError, match='repeated'):
        raw.reorder_channels(raw.ch_names[:1] + raw.ch_names[:1])
    # and with .pick
    reord = [1, 0] + list(range(2, len(raw.ch_names)))
    rev = np.argsort(reord)
    raw_new = raw.copy().pick(reord)
    assert_array_equal(raw[:][0], raw_new[rev][0])


def test_rename_channels():
    """Test rename channels."""
    info = read_info(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    mapping = {'EEG 160': 'EEG060'}
    pytest.raises(ValueError, rename_channels, info, mapping)
    # Test improper mapping configuration
    mapping = {'MEG 2641': 1.0}
    pytest.raises(TypeError, rename_channels, info, mapping)
    # Test non-unique mapping configuration
    mapping = {'MEG 2641': 'MEG 2642'}
    pytest.raises(ValueError, rename_channels, info, mapping)
    # Test bad input
    pytest.raises(ValueError, rename_channels, info, 1.)
    pytest.raises(ValueError, rename_channels, info, 1.)

    # Test successful changes
    # Test ch_name and ch_names are changed
    info2 = deepcopy(info)  # for consistency at the start of each test
    info2['bads'] = ['EEG 060', 'EOG 061']
    mapping = {'EEG 060': 'EEG060', 'EOG 061': 'EOG061'}
    rename_channels(info2, mapping)
    assert info2['chs'][374]['ch_name'] == 'EEG060'
    assert info2['ch_names'][374] == 'EEG060'
    assert info2['chs'][375]['ch_name'] == 'EOG061'
    assert info2['ch_names'][375] == 'EOG061'
    assert_array_equal(['EEG060', 'EOG061'], info2['bads'])
    info2 = deepcopy(info)
    rename_channels(info2, lambda x: x.replace(' ', ''))
    assert info2['chs'][373]['ch_name'] == 'EEG059'
    info2 = deepcopy(info)
    info2['bads'] = ['EEG 060', 'EEG 060']
    rename_channels(info2, mapping)
    assert_array_equal(['EEG060', 'EEG060'], info2['bads'])

    # test that keys in Raw._orig_units will be renamed, too
    raw = read_raw_fif(raw_fname).crop(0, 0.1)
    old, new = 'EEG 060', 'New'
    raw._orig_units = {old: 'V'}

    raw.rename_channels({old: new})
    assert old not in raw._orig_units
    assert new in raw._orig_units


def test_set_channel_types():
    """Test set_channel_types."""
    raw = read_raw_fif(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    mapping = {'EEG 160': 'EEG060'}
    with pytest.raises(ValueError, match=r"name \(EEG 160\) doesn't exist"):
        raw.set_channel_types(mapping)
    # Test change to illegal channel type
    mapping = {'EOG 061': 'xxx'}
    with pytest.raises(ValueError, match='cannot change to this channel type'):
        raw.set_channel_types(mapping)
    # Test changing type if in proj
    mapping = {'EEG 057': 'dbs', 'EEG 058': 'ecog', 'EEG 059': 'ecg',
               'EEG 060': 'eog', 'EOG 061': 'seeg', 'MEG 2441': 'eeg',
               'MEG 2443': 'eeg', 'MEG 2442': 'hbo', 'EEG 001': 'resp'}
    raw2 = read_raw_fif(raw_fname)
    raw2.info['bads'] = ['EEG 059', 'EEG 060', 'EOG 061']
    with pytest.raises(RuntimeError, match='type .* in projector "PCA-v1"'):
        raw2.set_channel_types(mapping)  # has prj
    raw2.add_proj([], remove_existing=True)
    with pytest.warns(RuntimeWarning, match='unit for channel.* has changed'):
        raw2 = raw2.set_channel_types(mapping)
    info = raw2.info
    assert info['chs'][371]['ch_name'] == 'EEG 057'
    assert info['chs'][371]['kind'] == FIFF.FIFFV_DBS_CH
    assert info['chs'][371]['unit'] == FIFF.FIFF_UNIT_V
    assert info['chs'][371]['coil_type'] == FIFF.FIFFV_COIL_EEG
    assert info['chs'][372]['ch_name'] == 'EEG 058'
    assert info['chs'][372]['kind'] == FIFF.FIFFV_ECOG_CH
    assert info['chs'][372]['unit'] == FIFF.FIFF_UNIT_V
    assert info['chs'][372]['coil_type'] == FIFF.FIFFV_COIL_EEG
    assert info['chs'][373]['ch_name'] == 'EEG 059'
    assert info['chs'][373]['kind'] == FIFF.FIFFV_ECG_CH
    assert info['chs'][373]['unit'] == FIFF.FIFF_UNIT_V
    assert info['chs'][373]['coil_type'] == FIFF.FIFFV_COIL_NONE
    assert info['chs'][374]['ch_name'] == 'EEG 060'
    assert info['chs'][374]['kind'] == FIFF.FIFFV_EOG_CH
    assert info['chs'][374]['unit'] == FIFF.FIFF_UNIT_V
    assert info['chs'][374]['coil_type'] == FIFF.FIFFV_COIL_NONE
    assert info['chs'][375]['ch_name'] == 'EOG 061'
    assert info['chs'][375]['kind'] == FIFF.FIFFV_SEEG_CH
    assert info['chs'][375]['unit'] == FIFF.FIFF_UNIT_V
    assert info['chs'][375]['coil_type'] == FIFF.FIFFV_COIL_EEG
    for idx in pick_channels(raw.ch_names, ['MEG 2441', 'MEG 2443']):
        assert info['chs'][idx]['kind'] == FIFF.FIFFV_EEG_CH
        assert info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_V
        assert info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_EEG
    idx = pick_channels(raw.ch_names, ['MEG 2442'])[0]
    assert info['chs'][idx]['kind'] == FIFF.FIFFV_FNIRS_CH
    assert info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_MOL
    assert info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_FNIRS_HBO

    # resp channel type
    idx = pick_channels(raw.ch_names, ['EEG 001'])[0]
    assert info['chs'][idx]['kind'] == FIFF.FIFFV_RESP_CH
    assert info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_V
    assert info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_NONE

    # Test meaningful error when setting channel type with unknown unit
    raw.info['chs'][0]['unit'] = 0.
    ch_types = {raw.ch_names[0]: 'misc'}
    pytest.raises(ValueError, raw.set_channel_types, ch_types)


def test_read_ch_adjacency(tmp_path):
    """Test reading channel adjacency templates."""
    tempdir = str(tmp_path)
    a = partial(np.array, dtype='<U7')
    # no pep8
    nbh = np.array([[(['MEG0111'], [[a(['MEG0131'])]]),
                     (['MEG0121'], [[a(['MEG0111'])],
                                    [a(['MEG0131'])]]),
                     (['MEG0131'], [[a(['MEG0111'])],
                                    [a(['MEG0121'])]])]],
                   dtype=[('label', 'O'), ('neighblabel', 'O')])
    mat = dict(neighbours=nbh)
    mat_fname = op.join(tempdir, 'test_mat.mat')
    savemat(mat_fname, mat, oned_as='row')

    ch_adjacency, ch_names = read_ch_adjacency(mat_fname)
    x = ch_adjacency
    assert_equal(x.shape[0], len(ch_names))
    assert_equal(x.shape, (3, 3))
    assert_equal(x[0, 1], False)
    assert_equal(x[0, 2], True)
    assert np.all(x.diagonal())
    pytest.raises(ValueError, read_ch_adjacency, mat_fname, [0, 3])
    ch_adjacency, ch_names = read_ch_adjacency(mat_fname, picks=[0, 2])
    assert_equal(ch_adjacency.shape[0], 2)
    assert_equal(len(ch_names), 2)

    ch_names = ['EEG01', 'EEG02', 'EEG03']
    neighbors = [['EEG02'], ['EEG04'], ['EEG02']]
    pytest.raises(ValueError, _ch_neighbor_adjacency, ch_names, neighbors)
    neighbors = [['EEG02'], ['EEG01', 'EEG03'], ['EEG 02']]
    pytest.raises(ValueError, _ch_neighbor_adjacency, ch_names[:2],
                  neighbors)
    neighbors = [['EEG02'], 'EEG01', ['EEG 02']]
    pytest.raises(ValueError, _ch_neighbor_adjacency, ch_names, neighbors)
    adjacency, ch_names = read_ch_adjacency('neuromag306mag')
    assert_equal(adjacency.shape, (102, 102))
    assert_equal(len(ch_names), 102)
    pytest.raises(ValueError, read_ch_adjacency, 'bananas!')

    # In EGI 256, E31 sensor has no neighbour
    a = partial(np.array)
    nbh = np.array([[(['E31'], []),
                     (['E1'], [[a(['E2'])],
                               [a(['E3'])]]),
                     (['E2'], [[a(['E1'])],
                               [a(['E3'])]]),
                     (['E3'], [[a(['E1'])],
                               [a(['E2'])]])]],
                   dtype=[('label', 'O'), ('neighblabel', 'O')])
    mat = dict(neighbours=nbh)
    mat_fname = op.join(tempdir, 'test_isolated_mat.mat')
    savemat(mat_fname, mat, oned_as='row')
    ch_adjacency, ch_names = read_ch_adjacency(mat_fname)
    x = ch_adjacency.todense()
    assert_equal(x.shape[0], len(ch_names))
    assert_equal(x.shape, (4, 4))
    assert np.all(x.diagonal())
    assert not np.any(x[0, 1:])
    assert not np.any(x[1:, 0])

    # Check for neighbours consistency. If a sensor is marked as a neighbour,
    # then it should also have its neighbours defined.
    a = partial(np.array)
    nbh = np.array([[(['E31'], []),
                     (['E1'], [[a(['E8'])],
                               [a(['E3'])]]),
                     (['E2'], [[a(['E1'])],
                               [a(['E3'])]]),
                     (['E3'], [[a(['E1'])],
                               [a(['E2'])]])]],
                   dtype=[('label', 'O'), ('neighblabel', 'O')])
    mat = dict(neighbours=nbh)
    mat_fname = op.join(tempdir, 'test_error_mat.mat')
    savemat(mat_fname, mat, oned_as='row')
    pytest.raises(ValueError, read_ch_adjacency, mat_fname)


def test_get_set_sensor_positions():
    """Test get/set functions for sensor positions."""
    raw1 = read_raw_fif(raw_fname)
    picks = pick_types(raw1.info, meg=False, eeg=True)
    pos = np.array([ch['loc'][:3] for ch in raw1.info['chs']])[picks]
    raw_pos = raw1._get_channel_positions(picks=picks)
    assert_array_equal(raw_pos, pos)

    ch_name = raw1.info['ch_names'][13]
    pytest.raises(ValueError, raw1._set_channel_positions, [1, 2], ['name'])
    raw2 = read_raw_fif(raw_fname)
    raw2.info['chs'][13]['loc'][:3] = np.array([1, 2, 3])
    raw1._set_channel_positions([[1, 2, 3]], [ch_name])
    assert_array_equal(raw1.info['chs'][13]['loc'],
                       raw2.info['chs'][13]['loc'])


@testing.requires_testing_data
def test_1020_selection():
    """Test making a 10/20 selection dict."""
    base_dir = op.join(testing.data_path(download=False), 'EEGLAB')
    raw_fname = op.join(base_dir, 'test_raw.set')
    loc_fname = op.join(base_dir, 'test_chans.locs')
    raw = read_raw_eeglab(raw_fname, preload=True)
    montage = read_custom_montage(loc_fname)
    raw = raw.rename_channels(dict(zip(raw.ch_names, montage.ch_names)))
    raw.set_montage(montage)

    for input in ("a_string", 100, raw, [1, 2]):
        pytest.raises(TypeError, make_1020_channel_selections, input)

    sels = make_1020_channel_selections(raw.info)
    # are all frontal channels placed before all occipital channels?
    for name, picks in sels.items():
        fs = min([ii for ii, pick in enumerate(picks)
                  if raw.ch_names[pick].startswith("F")])
        ps = max([ii for ii, pick in enumerate(picks)
                  if raw.ch_names[pick].startswith("O")])
        assert fs > ps

    # are channels in the correct selection?
    fz_c3_c4 = [raw.ch_names.index(ch) for ch in ("Fz", "C3", "C4")]
    for channel, roi in zip(fz_c3_c4, ("Midline", "Left", "Right")):
        assert channel in sels[roi]


@testing.requires_testing_data
def test_find_ch_adjacency():
    """Test computing the adjacency matrix."""
    data_path = testing.data_path()

    raw = read_raw_fif(raw_fname, preload=True)
    sizes = {'mag': 828, 'grad': 1700, 'eeg': 384}
    nchans = {'mag': 102, 'grad': 204, 'eeg': 60}
    for ch_type in ['mag', 'grad', 'eeg']:
        conn, ch_names = find_ch_adjacency(raw.info, ch_type)
        # Silly test for checking the number of neighbors.
        assert_equal(conn.getnnz(), sizes[ch_type])
        assert_equal(len(ch_names), nchans[ch_type])
    pytest.raises(ValueError, find_ch_adjacency, raw.info, None)

    # Test computing the conn matrix with gradiometers.
    conn, ch_names = _compute_ch_adjacency(raw.info, 'grad')
    assert_equal(conn.getnnz(), 2680)

    # Test ch_type=None.
    raw.pick_types(meg='mag')
    find_ch_adjacency(raw.info, None)

    bti_fname = op.join(data_path, 'BTi', 'erm_HFH', 'c,rfDC')
    bti_config_name = op.join(data_path, 'BTi', 'erm_HFH', 'config')
    raw = read_raw_bti(bti_fname, bti_config_name, None)
    _, ch_names = find_ch_adjacency(raw.info, 'mag')
    assert 'A1' in ch_names

    ctf_fname = op.join(data_path, 'CTF', 'testdata_ctf_short.ds')
    raw = read_raw_ctf(ctf_fname)
    _, ch_names = find_ch_adjacency(raw.info, 'mag')
    assert 'MLC11' in ch_names

    pytest.raises(ValueError, find_ch_adjacency, raw.info, 'eog')

    raw_kit = read_raw_kit(fname_kit_157)
    neighb, ch_names = find_ch_adjacency(raw_kit.info, 'mag')
    assert neighb.data.size == 1329
    assert ch_names[0] == 'MEG 001'


@testing.requires_testing_data
def test_neuromag122_adjacency():
    """Test computing the adjacency matrix of Neuromag122-Data."""
    nm122_fname = op.join(testing.data_path(), 'misc',
                          'neuromag122_test_file-raw.fif')
    raw = read_raw_fif(nm122_fname, preload=True)
    conn, ch_names = find_ch_adjacency(raw.info, 'grad')
    assert conn.getnnz() == 1564
    assert len(ch_names) == 122
    assert conn.shape == (122, 122)


def test_drop_channels():
    """Test if dropping channels works with various arguments."""
    raw = read_raw_fif(raw_fname, preload=True).crop(0, 0.1)
    raw.drop_channels(["MEG 0111"])  # list argument
    raw.drop_channels("MEG 0112")  # str argument
    raw.drop_channels({"MEG 0132", "MEG 0133"})  # set argument
    pytest.raises(ValueError, raw.drop_channels, ["MEG 0111", 5])
    pytest.raises(ValueError, raw.drop_channels, 5)  # must be list or str


def test_pick_channels():
    """Test if picking channels works with various arguments."""
    raw = read_raw_fif(raw_fname, preload=True).crop(0, 0.1)

    # selected correctly 3 channels
    raw.pick(['MEG 0113', 'MEG 0112', 'MEG 0111'])
    assert len(raw.ch_names) == 3

    # selected correctly 3 channels and ignored 'meg', and emit warning
    with pytest.warns(RuntimeWarning, match='not present in the info'):
        raw.pick(['MEG 0113', "meg", 'MEG 0112', 'MEG 0111'])
        assert len(raw.ch_names) == 3

    names_len = len(raw.ch_names)
    raw.pick(['all'])  # selected correctly all channels
    assert len(raw.ch_names) == names_len
    raw.pick('all')  # selected correctly all channels
    assert len(raw.ch_names) == names_len


def test_add_reference_channels():
    """Test if there is a new reference channel that consist of all zeros."""
    raw = read_raw_fif(raw_fname, preload=True)
    n_raw_original_channels = len(raw.ch_names)
    epochs = Epochs(raw, read_events(eve_fname))
    epochs.load_data()
    epochs_original_shape = epochs._data.shape[1]
    evoked = epochs.average()
    n_evoked_original_channels = len(evoked.ch_names)

    # Raw object
    raw.add_reference_channels(['REF 123'])
    assert len(raw.ch_names) == n_raw_original_channels + 1
    assert np.all(raw.get_data()[-1] == 0)

    # Epochs object
    epochs.add_reference_channels(['REF 123'])
    assert epochs._data.shape[1] == epochs_original_shape + 1

    # Evoked object
    evoked.add_reference_channels(['REF 123'])
    assert len(evoked.ch_names) == n_evoked_original_channels + 1
    assert np.all(evoked._data[-1] == 0)


def test_equalize_channels():
    """Test equalizing channels and their ordering."""
    # This function only tests the generic functionality of equalize_channels.
    # Additional tests for each instance type are included in the accompanying
    # test suite for each type.
    pytest.raises(TypeError, equalize_channels, ['foo', 'bar'],
                  match='Instances to be modified must be an instance of')

    raw = RawArray([[1.], [2.], [3.], [4.]],
                   create_info(['CH1', 'CH2', 'CH3', 'CH4'], sfreq=1.))
    epochs = EpochsArray([[[1.], [2.], [3.]]],
                         create_info(['CH5', 'CH2', 'CH1'], sfreq=1.))
    cov = make_ad_hoc_cov(create_info(['CH2', 'CH1', 'CH8'], sfreq=1.,
                                      ch_types='eeg'))
    cov['bads'] = ['CH1']
    ave = EvokedArray([[1.], [2.]], create_info(['CH1', 'CH2'], sfreq=1.))

    raw2, epochs2, cov2, ave2 = equalize_channels([raw, epochs, cov, ave],
                                                  copy=True)

    # The Raw object was the first in the list, so should have been used as
    # template for the ordering of the channels. No bad channels should have
    # been dropped.
    assert raw2.ch_names == ['CH1', 'CH2']
    assert_array_equal(raw2.get_data(), [[1.], [2.]])
    assert epochs2.ch_names == ['CH1', 'CH2']
    assert_array_equal(epochs2.get_data(), [[[3.], [2.]]])
    assert cov2.ch_names == ['CH1', 'CH2']
    assert cov2['bads'] == cov['bads']
    assert ave2.ch_names == ave.ch_names
    assert_array_equal(ave2.data, ave.data)

    # All objects should have been copied, except for the Evoked object which
    # did not have to be touched.
    assert raw is not raw2
    assert epochs is not epochs2
    assert cov is not cov2
    assert ave is ave2

    # Test in-place operation
    raw2, epochs2 = equalize_channels([raw, epochs], copy=False)
    assert raw is raw2
    assert epochs is epochs2


def test_combine_channels():
    """Test channel combination on Raw, Epochs, and Evoked."""
    raw = read_raw_fif(raw_fname, preload=True)
    raw_ch_bad = read_raw_fif(raw_fname, preload=True)
    raw_ch_bad.info['bads'] = ['MEG 0113', 'MEG 0112']
    epochs = Epochs(raw, read_events(eve_fname))
    evoked = epochs.average()
    good = dict(foo=[0, 1, 3, 4], bar=[5, 2])  # good grad and mag

    # Test good cases
    combine_channels(raw, good)
    combined_epochs = combine_channels(epochs, good)
    assert_array_equal(combined_epochs.events, epochs.events)
    combine_channels(evoked, good)
    combine_channels(raw, good, drop_bad=True)
    combine_channels(raw_ch_bad, good, drop_bad=True)

    # Test with stimulus channels
    combine_stim = combine_channels(raw, good, keep_stim=True)
    target_nchan = len(good) + len(pick_types(raw.info, meg=False, stim=True))
    assert combine_stim.info['nchan'] == target_nchan

    # Test results with one ROI
    good_single = dict(foo=[0, 1, 3, 4])  # good grad
    combined_mean = combine_channels(raw, good_single, method='mean')
    combined_median = combine_channels(raw, good_single, method='median')
    combined_std = combine_channels(raw, good_single, method='std')
    foo_mean = np.mean(raw.get_data()[good_single['foo']], axis=0)
    foo_median = np.median(raw.get_data()[good_single['foo']], axis=0)
    foo_std = np.std(raw.get_data()[good_single['foo']], axis=0)
    assert_array_equal(combined_mean.get_data(),
                       np.expand_dims(foo_mean, axis=0))
    assert_array_equal(combined_median.get_data(),
                       np.expand_dims(foo_median, axis=0))
    assert_array_equal(combined_std.get_data(),
                       np.expand_dims(foo_std, axis=0))

    # Test bad cases
    bad1 = dict(foo=[0, 376], bar=[5, 2])  # out of bounds
    bad2 = dict(foo=[0, 2], bar=[5, 2])  # type mix in same group
    with pytest.raises(ValueError, match='"method" must be a callable, or'):
        combine_channels(raw, good, method='bad_method')
    with pytest.raises(TypeError, match='"keep_stim" must be of type bool'):
        combine_channels(raw, good, keep_stim='bad_type')
    with pytest.raises(TypeError, match='"drop_bad" must be of type bool'):
        combine_channels(raw, good, drop_bad='bad_type')
    with pytest.raises(ValueError, match='Some channel indices are out of'):
        combine_channels(raw, bad1)
    with pytest.raises(ValueError, match='Cannot combine sensors of diff'):
        combine_channels(raw, bad2)

    # Test warnings
    raw_no_stim = read_raw_fif(raw_fname, preload=True)
    raw_no_stim.pick_types(meg=True, stim=False)
    warn1 = dict(foo=[375, 375], bar=[5, 2])  # same channel in same group
    warn2 = dict(foo=[375], bar=[5, 2])  # one channel (last channel)
    warn3 = dict(foo=[0, 4], bar=[5, 2])  # one good channel left
    with pytest.warns(RuntimeWarning, match='Could not find stimulus'):
        combine_channels(raw_no_stim, good, keep_stim=True)
    with pytest.warns(RuntimeWarning, match='Less than 2 channels') as record:
        combine_channels(raw, warn1)
        combine_channels(raw, warn2)
        combine_channels(raw_ch_bad, warn3, drop_bad=True)
    assert len(record) == 3


def test_get_montage():
    """Test ContainsMixin.get_montage()."""
    ch_names = make_standard_montage('standard_1020').ch_names
    sfreq = 512
    data = np.zeros((len(ch_names), sfreq * 2))
    raw = RawArray(data, create_info(ch_names, sfreq, 'eeg'))
    raw.set_montage('standard_1020')

    assert len(raw.get_montage().ch_names) == len(ch_names)
    raw.info['bads'] = [ch_names[0]]
    assert len(raw.get_montage().ch_names) == len(ch_names)
