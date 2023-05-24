# Author: Proloy Das <pdas6@mgh.harvard.edu>
#
# License: BSD-3-Clause
import os
import numpy as np

from mne.io import read_raw_nsx
from mne.datasets.testing import data_path, requires_testing_data


testing_path = data_path(download=False)
data_dir = os.path.join(testing_path, 'nsx')
nsx_22_fname = os.path.join(data_dir, 'test_NEURALCD_raw.ns3')
nsx_31_fname = os.path.join(data_dir, 'test_BRSMPGRP_raw.ns3')


@requires_testing_data
def test_nsx_ver_31():
    """Primary tests for NSx reader"""
    raw = read_raw_nsx(nsx_31_fname)
    assert getattr(raw, '_data', False) is False
    assert raw.info['sfreq'] == 2000

    # Check info object
    assert raw.info['meas_date'].day == 31
    assert raw.info['meas_date'].year == 2023
    assert raw.info['meas_date'].month == 1
    assert raw.info['chs'][0]['cal'] == 0.6103515625
    assert raw.info['chs'][0]['range'] == 0.001

    # check raw_extras
    for r in raw._raw_extras:
        assert r['orig_format'] == raw.orig_format
        assert r['orig_nchan'] == 128
        assert len(r['timestamp']) == len(r['nb_data_points'])
        assert len(r['timestamp']) == len(r['offset_to_data_block'])

    # Check annotations
    assert raw.annotations[0]['onset'] * raw.info['sfreq'] == 101
    assert raw.annotations[0]['duration'] * raw.info['sfreq'] == 49

    raw.load_data()
    raw_data, times = raw[:]
    n_channels, n_times = raw_data.shape
    assert times.shape[0] == n_times
    orig_data = raw_data / (raw.info['chs'][0]['cal']
                            * raw.info['chs'][0]['range'])
    assert np.allclose(sum(orig_data[:, 50] - 10 - np.arange(n_channels)), 76.)

    assert np.allclose(orig_data[n_channels // 2, :100] - 100, np.arange(100))
    assert np.allclose(orig_data[n_channels // 2, 150:] - 100, np.arange(150))

    data, times = raw.get_data(start=10, stop=20, return_times=True)
    assert n_channels, 10 == data.shape

    data, times = raw.get_data(start=0, stop=300, return_times=True)


def test_nsx_ver_22():
    """Primary tests for NSx reader"""
    raw = read_raw_nsx(nsx_22_fname)
    assert getattr(raw, '_data', False) is False
    assert raw.info['sfreq'] == 2000

    # Check info object
    assert raw.info['meas_date'].day == 31
    assert raw.info['meas_date'].year == 2023
    assert raw.info['meas_date'].month == 1
    assert raw.info['chs'][0]['cal'] == 0.6103515625
    assert raw.info['chs'][0]['range'] == 0.001

    # check raw_extras
    for r in raw._raw_extras:
        assert r['orig_format'] == raw.orig_format
        assert r['orig_nchan'] == 128
        assert len(r['timestamp']) == len(r['nb_data_points'])
        assert len(r['timestamp']) == len(r['offset_to_data_block'])

    # Check annotations
    assert len(raw.annotations) == 0

    raw.load_data()
    raw_data, times = raw[:]
    n_channels, n_times = raw_data.shape
    assert times.shape[0] == n_times
    orig_data = raw_data / (raw.info['chs'][0]['cal']
                            * raw.info['chs'][0]['range'])
    assert np.allclose(sum(orig_data[:, 50] - 10 - np.arange(n_channels)), 76.)

    assert np.allclose(orig_data[n_channels // 2, :100] - 100, np.arange(100))

    data, times = raw.get_data(start=10, stop=20, return_times=True)
    assert n_channels, 10 == data.shape

    data, times = raw.get_data(start=0, stop=300, return_times=True)



## TODO
# check stim channels