# Author: Daniel G Wakeman <dwakeman@nmr.mgh.harvard.edu>
#         Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from copy import deepcopy

import numpy as np
import warnings
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_true, assert_equal

from mne.channels import rename_channels, read_ch_connectivity
from mne.channels.channels import _ch_neighbor_connectivity
from mne.io import read_info, Raw
from mne.io.constants import FIFF
from mne.fixes import partial, savemat
from mne.utils import _TempDir
from mne import pick_types

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

warnings.simplefilter('always')


def test_rename_channels():
    """Test rename channels
    """
    info = read_info(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    mapping = {'EEG 160': 'EEG060'}
    assert_raises(ValueError, rename_channels, info, mapping)
    # Test change to EEG channel
    mapping = {'EOG 061': ('EEG 061', 'eeg')}
    with warnings.catch_warnings(record=True):
        assert_raises(ValueError, rename_channels, info, mapping)
    # Test change to illegal channel type
    mapping = {'EOG 061': ('MEG 061', 'meg')}
    with warnings.catch_warnings(record=True):
        assert_raises(ValueError, rename_channels, info, mapping)
    # Test channel type which you are changing from e.g. MEG
    mapping = {'MEG 2641': ('MEG2641', 'eeg')}
    with warnings.catch_warnings(record=True):
        assert_raises(ValueError, rename_channels, info, mapping)
    # Test improper mapping configuration
    mapping = {'MEG 2641': 1.0}
    assert_raises(ValueError, rename_channels, info, mapping)
    # Test successful changes
    # Test ch_name and ch_names are changed
    info2 = deepcopy(info)  # for consistency at the start of each test
    info2['bads'] = ['EEG 060', 'EOG 061']
    mapping = {'EEG 060': 'EEG060', 'EOG 061': 'EOG061'}
    rename_channels(info2, mapping)
    assert_true(info2['chs'][374]['ch_name'] == 'EEG060')
    assert_true(info2['ch_names'][374] == 'EEG060')
    assert_true('EEG060' in info2['bads'])
    assert_true(info2['chs'][375]['ch_name'] == 'EOG061')
    assert_true(info2['ch_names'][375] == 'EOG061')
    assert_true('EOG061' in info2['bads'])


def test_set_channel_types():
    """Test set_channel_types
    """
    raw = Raw(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    mapping = {'EEG 160': 'EEG060'}
    assert_raises(ValueError, raw.set_channel_types, mapping)
    # Test change to illegal channel type
    mapping = {'EOG 061': 'xxx'}
    assert_raises(ValueError, raw.set_channel_types, mapping)
    # Test type change
    raw2 = Raw(raw_fname)
    raw2.info['bads'] = ['EEG 059', 'EEG 060', 'EOG 061']
    mapping = {'EEG 060': 'eog', 'EEG 059': 'ecg', 'EOG 061': 'seeg'}
    raw2.set_channel_types(mapping)
    info = raw2.info
    assert_true(info['chs'][374]['ch_name'] == 'EEG 060')
    assert_true(info['chs'][374]['kind'] == FIFF.FIFFV_EOG_CH)
    assert_true(info['chs'][374]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][374]['coil_type'] == FIFF.FIFFV_COIL_NONE)
    assert_true(info['chs'][373]['ch_name'] == 'EEG 059')
    assert_true(info['chs'][373]['kind'] == FIFF.FIFFV_ECG_CH)
    assert_true(info['chs'][373]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][373]['coil_type'] == FIFF.FIFFV_COIL_NONE)
    assert_true(info['chs'][375]['ch_name'] == 'EOG 061')
    assert_true(info['chs'][375]['kind'] == FIFF.FIFFV_SEEG_CH)
    assert_true(info['chs'][375]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][375]['coil_type'] == FIFF.FIFFV_COIL_EEG)


def test_read_ch_connectivity():
    "Test reading channel connectivity templates"
    tempdir = _TempDir()
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

    ch_connectivity, ch_names = read_ch_connectivity(mat_fname)
    x = ch_connectivity
    assert_equal(x.shape[0], len(ch_names))
    assert_equal(x.shape, (3, 3))
    assert_equal(x[0, 1], False)
    assert_equal(x[0, 2], True)
    assert_true(np.all(x.diagonal()))
    assert_raises(ValueError, read_ch_connectivity, mat_fname, [0, 3])
    ch_connectivity, ch_names = read_ch_connectivity(mat_fname, picks=[0, 2])
    assert_equal(ch_connectivity.shape[0], 2)
    assert_equal(len(ch_names), 2)

    ch_names = ['EEG01', 'EEG02', 'EEG03']
    neighbors = [['EEG02'], ['EEG04'], ['EEG02']]
    assert_raises(ValueError, _ch_neighbor_connectivity, ch_names, neighbors)
    neighbors = [['EEG02'], ['EEG01', 'EEG03'], ['EEG 02']]
    assert_raises(ValueError, _ch_neighbor_connectivity, ch_names[:2],
                  neighbors)
    neighbors = [['EEG02'], 'EEG01', ['EEG 02']]
    assert_raises(ValueError, _ch_neighbor_connectivity, ch_names, neighbors)
    connectivity, ch_names = read_ch_connectivity('neuromag306mag')
    assert_equal(connectivity.shape, (102, 102))
    assert_equal(len(ch_names), 102)
    assert_raises(ValueError, read_ch_connectivity, 'bananas!')


def test_get_set_sensor_positions():
    """Test get/set functions for sensor positions
    """
    raw1 = Raw(raw_fname)
    picks = pick_types(raw1.info, meg=False, eeg=True)
    pos = np.array([ch['loc'][:3] for ch in raw1.info['chs']])[picks]
    raw_pos = raw1._get_channel_positions(picks=picks)
    assert_array_equal(raw_pos, pos)

    ch_name = raw1.info['ch_names'][13]
    assert_raises(ValueError, raw1._set_channel_positions, [1, 2], ['name'])
    raw2 = Raw(raw_fname)
    raw2.info['chs'][13]['loc'][:3] = np.array([1, 2, 3])
    raw1._set_channel_positions([[1, 2, 3]], [ch_name])
    assert_array_equal(raw1.info['chs'][13]['loc'],
                       raw2.info['chs'][13]['loc'])
