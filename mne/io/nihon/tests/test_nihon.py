# -*- coding: utf-8 -*-
# Authors: Federico Raimondo  <federaimondo@gmail.com>
#          simplified BSD-3 license
import os.path as op

from numpy.testing import assert_array_almost_equal

from mne.io import read_raw_nihon, read_raw_edf
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import run_tests_if_main
from mne.datasets.testing import data_path, requires_testing_data


@requires_testing_data
def test_nihon_eeg():
    """Test reading Nihon Kohden EEG files."""
    fname = op.join(data_path(), 'NihonKohden', 'MB0400FU.EEG')
    raw = read_raw_nihon(fname, preload=True)
    assert 'RawNihon' in repr(raw)
    _test_raw_reader(read_raw_nihon, fname=fname)
    fname_edf = op.join(data_path(), 'NihonKohden', 'MB0400FU.EDF')
    raw_edf = read_raw_edf(fname_edf, preload=True)

    assert raw._data.shape == raw_edf._data.shape
    assert raw.info['sfreq'] == raw.info['sfreq']
    edf_ch_names = {}
    # ch names and order are switched in the EDF
    for x in raw_edf.ch_names:
        to_replace =  x.split(' ')[1]
        to_replace = to_replace.replace('Fp', 'FP').replace('-Ref', '')
        to_replace = to_replace.replace('Fz', 'FZ')
        to_replace = to_replace.replace('Cz', 'CZ')
        to_replace = to_replace.replace('Pz', 'PZ')
        to_replace = to_replace.replace('$A1', 'Mark2').replace('$A2', 'Mark1')
        edf_ch_names[x] = to_replace
    raw_edf.rename_channels(edf_ch_names)
    assert set(raw.ch_names) == set(raw_edf.ch_names)
    raw_edf = raw_edf.pick_channels(raw.ch_names)
    # This does not work, the EDF says everything is EEG
    # types_dict = {2: 'eeg', 3: 'stim', 202: 'eog', 502: 'misc', 102: 'bio'}
    # ch_types = [types_dict[raw.info['chs'][x]['kind']]
    #             for x in range(len(raw.ch_names))]
    # edf_ch_types = [types_dict[raw_edf.info['chs'][x]['kind']]
    #                 for x in range(len(raw_edf.ch_names))]
    # assert ch_types == edf_ch_types

    assert_array_almost_equal(raw._data, raw_edf._data)


run_tests_if_main()
