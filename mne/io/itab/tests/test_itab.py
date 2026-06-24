# -*- coding: utf-8 -*-
# Authors: Roberto Guidotti  <rob.guidotti@gmail.com>
#          simplified BSD-3 license
import os.path as op

from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy import io as sio

from mne.io import read_raw_itab
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing

data_path = testing.data_path(download=False)
raw_itab_fname = op.join(data_path, 'itab', 'test_itab.raw')
mat_itab_fname = op.join(data_path, 'itab', 'test_itab.mat')

@testing.requires_testing_data
def test_itab_raw():
    """Test reading ITAB .raw files."""
    
    raw = read_raw_itab(raw_itab_fname, preload=True)
    assert 'RawITAB' in repr(raw)
    
    _test_raw_reader(read_raw_itab, 
                    fname=raw_itab_fname,
                    test_scaling=False, 
    )
    
    mc = sio.loadmat(mat_itab_fname)

    m_data = mc['dat']
    m_header = mc['hdr']
    
    assert raw._data.shape == m_data.shape
    assert m_header['Fs'][0, 0][0, 0] == raw.info['sfreq']
    
    m_names = [x[0][0] for x in m_header['label'][0, 0]]   
    assert raw.ch_names == m_names
    
    assert_array_almost_equal(m_data, raw._data)
