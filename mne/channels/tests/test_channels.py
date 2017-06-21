# Author: Daniel G Wakeman <dwakeman@nmr.mgh.harvard.edu>
#         Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from copy import deepcopy
from functools import partial
import warnings

import numpy as np
from scipy.io import savemat
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_true, assert_equal

from mne.channels import rename_channels, read_ch_connectivity
from mne.channels.channels import _ch_neighbor_connectivity
from mne.io import read_info, read_raw_fif
from mne.io.constants import FIFF
from mne.utils import _TempDir, run_tests_if_main
from mne import pick_types, pick_channels

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

warnings.simplefilter('always')


def test_rename_channels():
    """Test rename channels"""
    info = read_info(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    mapping = {'EEG 160': 'EEG060'}
    assert_raises(ValueError, rename_channels, info, mapping)
    # Test improper mapping configuration
    mapping = {'MEG 2641': 1.0}
    assert_raises(ValueError, rename_channels, info, mapping)
    # Test non-unique mapping configuration
    mapping = {'MEG 2641': 'MEG 2642'}
    assert_raises(ValueError, rename_channels, info, mapping)
    # Test bad input
    assert_raises(ValueError, rename_channels, info, 1.)

    # Test successful changes
    # Test ch_name and ch_names are changed
    info2 = deepcopy(info)  # for consistency at the start of each test
    info2['bads'] = ['EEG 060', 'EOG 061']
    mapping = {'EEG 060': 'EEG060', 'EOG 061': 'EOG061'}
    rename_channels(info2, mapping)
    assert_true(info2['chs'][374]['ch_name'] == 'EEG060')
    assert_true(info2['ch_names'][374] == 'EEG060')
    assert_true(info2['chs'][375]['ch_name'] == 'EOG061')
    assert_true(info2['ch_names'][375] == 'EOG061')
    assert_array_equal(['EEG060', 'EOG061'], info2['bads'])
    info2 = deepcopy(info)
    rename_channels(info2, lambda x: x.replace(' ', ''))
    assert_true(info2['chs'][373]['ch_name'] == 'EEG059')
    info2 = deepcopy(info)
    info2['bads'] = ['EEG 060', 'EEG 060']
    rename_channels(info2, mapping)
    assert_array_equal(['EEG060', 'EEG060'], info2['bads'])


def test_set_channel_types():
    """Test set_channel_types"""
    raw = read_raw_fif(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    mapping = {'EEG 160': 'EEG060'}
    assert_raises(ValueError, raw.set_channel_types, mapping)
    # Test change to illegal channel type
    mapping = {'EOG 061': 'xxx'}
    assert_raises(ValueError, raw.set_channel_types, mapping)
    # Test changing type if in proj (avg eeg ref here)
    mapping = {'EEG 058': 'ecog', 'EEG 059': 'ecg', 'EEG 060': 'eog',
               'EOG 061': 'seeg', 'MEG 2441': 'eeg', 'MEG 2443': 'eeg',
               'MEG 2442': 'hbo'}
    assert_raises(RuntimeError, raw.set_channel_types, mapping)
    # Test type change
    raw2 = read_raw_fif(raw_fname)
    raw2.info['bads'] = ['EEG 059', 'EEG 060', 'EOG 061']
    with warnings.catch_warnings(record=True):  # MEG channel change
        assert_raises(RuntimeError, raw2.set_channel_types, mapping)  # has prj
    raw2.add_proj([], remove_existing=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw2.set_channel_types(mapping)
    assert_true(len(w) >= 1, msg=[str(ww.message) for ww in w])
    assert_true(all('The unit for channel' in str(ww.message) for ww in w))
    info = raw2.info
    assert_true(info['chs'][372]['ch_name'] == 'EEG 058')
    assert_true(info['chs'][372]['kind'] == FIFF.FIFFV_ECOG_CH)
    assert_true(info['chs'][372]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][372]['coil_type'] == FIFF.FIFFV_COIL_EEG)
    assert_true(info['chs'][373]['ch_name'] == 'EEG 059')
    assert_true(info['chs'][373]['kind'] == FIFF.FIFFV_ECG_CH)
    assert_true(info['chs'][373]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][373]['coil_type'] == FIFF.FIFFV_COIL_NONE)
    assert_true(info['chs'][374]['ch_name'] == 'EEG 060')
    assert_true(info['chs'][374]['kind'] == FIFF.FIFFV_EOG_CH)
    assert_true(info['chs'][374]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][374]['coil_type'] == FIFF.FIFFV_COIL_NONE)
    assert_true(info['chs'][375]['ch_name'] == 'EOG 061')
    assert_true(info['chs'][375]['kind'] == FIFF.FIFFV_SEEG_CH)
    assert_true(info['chs'][375]['unit'] == FIFF.FIFF_UNIT_V)
    assert_true(info['chs'][375]['coil_type'] == FIFF.FIFFV_COIL_EEG)
    for idx in pick_channels(raw.ch_names, ['MEG 2441', 'MEG 2443']):
        assert_true(info['chs'][idx]['kind'] == FIFF.FIFFV_EEG_CH)
        assert_true(info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_V)
        assert_true(info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_EEG)
    idx = pick_channels(raw.ch_names, ['MEG 2442'])[0]
    assert_true(info['chs'][idx]['kind'] == FIFF.FIFFV_FNIRS_CH)
    assert_true(info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_MOL)
    assert_true(info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_FNIRS_HBO)

    # Test meaningful error when setting channel type with unknown unit
    raw.info['chs'][0]['unit'] = 0.
    ch_types = {raw.ch_names[0]: 'misc'}
    assert_raises(ValueError, raw.set_channel_types, ch_types)


def test_read_ch_connectivity():
    """Test reading channel connectivity templates"""
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
    """Test get/set functions for sensor positions"""
    raw1 = read_raw_fif(raw_fname)
    picks = pick_types(raw1.info, meg=False, eeg=True)
    pos = np.array([ch['loc'][:3] for ch in raw1.info['chs']])[picks]
    raw_pos = raw1._get_channel_positions(picks=picks)
    assert_array_equal(raw_pos, pos)

    ch_name = raw1.info['ch_names'][13]
    assert_raises(ValueError, raw1._set_channel_positions, [1, 2], ['name'])
    raw2 = read_raw_fif(raw_fname)
    raw2.info['chs'][13]['loc'][:3] = np.array([1, 2, 3])
    raw1._set_channel_positions([[1, 2, 3]], [ch_name])
    assert_array_equal(raw1.info['chs'][13]['loc'],
                       raw2.info['chs'][13]['loc'])

run_tests_if_main()
