# Author: Daniel G Wakeman <dwakeman@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op

from copy import deepcopy

from nose.tools import assert_raises, assert_true

from mne import fiff
from mne.fiff.channels import rename_channels
from mne.fiff.constants import FIFF

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')


def test_rename_channels():
    """Test rename channels
    """
    info = fiff.read_info(raw_fname)
    # Error Tests
    # Test channel name exists in ch_names
    alias = {'EEG 160': 'EEG060'}
    assert_raises(RuntimeError, rename_channels, info, alias)
    # Test change to EEG channel
    alias = {'EOG 061': ('EEG 061', 'eeg')}
    assert_raises(RuntimeError, rename_channels, info, alias)
    # Test change to illegal channel type
    alias = {'EOG 061': ('MEG 061', 'meg')}
    assert_raises(RuntimeError, rename_channels, info, alias)
    # Test channel type which you are changing from e.g. MEG
    alias = {'MEG 2641': ('MEG2641', 'eeg')}
    assert_raises(RuntimeError, rename_channels, info, alias)
    # Test improper alias configuration
    alias = {'MEG 2641': 1.0}
    assert_raises(RuntimeError, rename_channels, info, alias)
    # Test duplicate named channels
    alias = {'EEG 060': 'EOG 061'}
    assert_raises(RuntimeError, rename_channels, info, alias)
    # Test successful changes
    # Test ch_name and ch_names are changed
    info2 = deepcopy(info)  # for consistency at the start of each test
    info2['bads'] = ['EEG 060', 'EOG 061']
    alias = {'EEG 060': 'EEG060', 'EOG 061': 'EOG061'}
    rename_channels(info2, alias)
    assert_true(info2['chs'][374]['ch_name'] == 'EEG060')
    assert_true(info2['ch_names'][374] == 'EEG060')
    assert_true('EEG060' in info2['bads'])
    assert_true(info2['chs'][375]['ch_name'] == 'EOG061')
    assert_true(info2['ch_names'][375] == 'EOG061')
    assert_true('EOG061' in info2['bads'])
    # Test type change
    info2 = deepcopy(info)
    info2['bads'] = ['EEG 060', 'EEG 059']
    alias = {'EEG 060': ('EOG 060', 'eog'), 'EEG 059': ('EOG 059', 'eog')}
    rename_channels(info2, alias)
    assert_true(info2['chs'][374]['ch_name'] == 'EOG 060')
    assert_true(info2['ch_names'][374] == 'EOG 060')
    assert_true('EOG 060' in info2['bads'])
    assert_true(info2['chs'][374]['kind'] is FIFF.FIFFV_EOG_CH)
    assert_true(info2['chs'][373]['ch_name'] == 'EOG 059')
    assert_true(info2['ch_names'][373] == 'EOG 059')
    assert_true('EOG 059' in info2['bads'])
    assert_true(info2['chs'][373]['kind'] is FIFF.FIFFV_EOG_CH)
