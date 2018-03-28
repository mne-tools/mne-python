# -*- coding: utf-8 -*-
# Authors: Federico Raimondo  <federaimondo@gmail.com>
#          simplified BSD-3 license
import os.path as op

from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_true
from scipy import io as sio

from mne.io import read_raw_eximia
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import run_tests_if_main
from mne.datasets.testing import data_path, requires_testing_data


@requires_testing_data
def test_eximia_nxe():
    """Test reading Eximia NXE files"""
    fname = op.join(data_path(), 'eximia', 'test_eximia.nxe')
    raw = read_raw_eximia(fname, preload=True)
    assert_true('RawEximia' in repr(raw))
    _test_raw_reader(read_raw_eximia, fname=fname)
    fname_mat = op.join(data_path(), 'eximia', 'test_eximia.mat')
    mc = sio.loadmat(fname_mat)
    m_data = mc['data']
    m_header = mc['header']
    assert_equal(raw._data.shape, m_data.shape)
    assert_equal(m_header['Fs'][0, 0][0, 0], raw.info['sfreq'])
    m_names = [x[0][0] for x in m_header['label'][0, 0]]
    m_names = list(
        map(lambda x: x.replace('GATE', 'GateIn').replace('TRIG', 'Trig'),
            m_names))
    assert_equal(raw.ch_names, m_names)
    m_ch_types = [x[0][0] for x in m_header['chantype'][0, 0]]
    m_ch_types = list(
        map(lambda x: x.replace('unknown', 'stim').replace('trigger', 'stim'),
            m_ch_types))
    types_dict = {2: 'eeg', 3: 'stim', 202: 'eog'}
    ch_types = [types_dict[raw.info['chs'][x]['kind']]
                for x in range(len(raw.ch_names))]
    assert_equal(ch_types, m_ch_types)

    assert_array_equal(m_data, raw._data)

run_tests_if_main()
