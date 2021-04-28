# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
from numpy.testing import assert_allclose, assert_almost_equal
import shutil
import pytest

from mne.datasets.testing import data_path, requires_testing_data
from mne.utils import run_tests_if_main, requires_h5py
from mne.io import read_raw_snirf, read_raw_nirx
from mne.io.tests.test_raw import _test_raw_reader
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    short_channels)

# SfNIRS files
sfnirs_homer_103_wShort = op.join(data_path(download=False),
                                  'SNIRF', 'SfNIRS', 'snirf_homer3', '1.0.3',
                                  'snirf_1_3_nirx_15_2_'
                                  'recording_w_short.snirf')
sfnirs_homer_103_wShort_original = op.join(data_path(download=False),
                                           'NIRx', 'nirscout',
                                           'nirx_15_2_recording_w_short')

sfnirs_homer_103_153 = op.join(data_path(download=False),
                               'SNIRF', 'SfNIRS', 'snirf_homer3', '1.0.3',
                               'nirx_15_3_recording.snirf')

# NIRx files
nirx_nirsport2_103 = op.join(data_path(download=False),
                             'SNIRF', 'NIRx', 'NIRSport2', '1.0.3',
                             '2021-04-23_005.snirf')


@requires_h5py
@requires_testing_data
@pytest.mark.filterwarnings('ignore:.*contains 2D location.*:')
@pytest.mark.parametrize('fname', ([sfnirs_homer_103_wShort,
                                    nirx_nirsport2_103,
                                    sfnirs_homer_103_153]))
def test_basic_reading_and_min_process(fname):
    """Test reading SNIRF files and minimum typical processing."""
    raw = read_raw_snirf(fname, preload=True)
    # SNIRF data can contain several types, so only apply appropriate functions
    if 'fnirs_cw_amplitude' in raw:
        raw = optical_density(raw)
    if 'fnirs_od' in raw:
        raw = beer_lambert_law(raw)
    assert 'hbo' in raw


@requires_testing_data
@requires_h5py
def test_snirf_basic():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(sfnirs_homer_103_wShort, preload=True)

    # Test data import
    assert raw._data.shape == (26, 145)
    assert raw.info['sfreq'] == 12.5

    # Test channel naming
    assert raw.info['ch_names'][:4] == ["S1_D1 760", "S1_D1 850",
                                        "S1_D9 760", "S1_D9 850"]
    assert raw.info['ch_names'][24:26] == ["S5_D13 760", "S5_D13 850"]

    # Test frequency encoding
    assert raw.info['chs'][0]['loc'][9] == 760
    assert raw.info['chs'][1]['loc'][9] == 850

    # Test source locations
    assert_allclose([-8.6765 * 1e-2, 0.0049 * 1e-2, -2.6167 * 1e-2],
                    raw.info['chs'][0]['loc'][3:6], rtol=0.02)
    assert_allclose([7.9579 * 1e-2, -2.7571 * 1e-2, -2.2631 * 1e-2],
                    raw.info['chs'][4]['loc'][3:6], rtol=0.02)
    assert_allclose([-2.1387 * 1e-2, -8.8874 * 1e-2, 3.8393 * 1e-2],
                    raw.info['chs'][8]['loc'][3:6], rtol=0.02)
    assert_allclose([1.8602 * 1e-2, 9.7164 * 1e-2, 1.7539 * 1e-2],
                    raw.info['chs'][12]['loc'][3:6], rtol=0.02)
    assert_allclose([-0.1108 * 1e-2, 0.7066 * 1e-2, 8.9883 * 1e-2],
                    raw.info['chs'][16]['loc'][3:6], rtol=0.02)

    # Test detector locations
    assert_allclose([-8.0409 * 1e-2, -2.9677 * 1e-2, -2.5415 * 1e-2],
                    raw.info['chs'][0]['loc'][6:9], rtol=0.02)
    assert_allclose([-8.7329 * 1e-2, 0.7577 * 1e-2, -2.7980 * 1e-2],
                    raw.info['chs'][3]['loc'][6:9], rtol=0.02)
    assert_allclose([9.2027 * 1e-2, 0.0161 * 1e-2, -2.8909 * 1e-2],
                    raw.info['chs'][5]['loc'][6:9], rtol=0.02)
    assert_allclose([7.7548 * 1e-2, -3.5901 * 1e-2, -2.3179 * 1e-2],
                    raw.info['chs'][7]['loc'][6:9], rtol=0.02)

    assert 'fnirs_cw_amplitude' in raw


@requires_testing_data
@requires_h5py
def test_snirf_against_nirx():
    """Test against file snirf was created from."""
    raw = read_raw_snirf(sfnirs_homer_103_wShort, preload=True)
    raw_orig = read_raw_nirx(sfnirs_homer_103_wShort_original, preload=True)

    # Check annotations are the same
    assert_allclose(raw.annotations.onset, raw_orig.annotations.onset)
    assert_allclose([float(d) for d in raw.annotations.description],
                    [float(d) for d in raw_orig.annotations.description])
    assert_allclose(raw.annotations.duration, raw_orig.annotations.duration)

    # Check names are the same
    assert raw.info['ch_names'] == raw_orig.info['ch_names']

    # Check frequencies are the same
    num_chans = len(raw.ch_names)
    new_chs = raw.info['chs']
    ori_chs = raw_orig.info['chs']
    assert_allclose([new_chs[idx]['loc'][9] for idx in range(num_chans)],
                    [ori_chs[idx]['loc'][9] for idx in range(num_chans)])

    # Check data is the same
    assert_allclose(raw.get_data(), raw_orig.get_data())


@requires_h5py
@requires_testing_data
def test_snirf_nonstandard(tmpdir):
    """Test custom tags."""
    from mne.externals.pymatreader.utils import _import_h5py
    h5py = _import_h5py()
    shutil.copy(sfnirs_homer_103_wShort, str(tmpdir) + "/mod.snirf")
    fname = str(tmpdir) + "/mod.snirf"
    # Manually mark up the file to match MNE-NIRS custom tags
    with h5py.File(fname, "r+") as f:
        f.create_dataset("nirs/metaDataTags/middleName",
                         data=['X'.encode('UTF-8')])
        f.create_dataset("nirs/metaDataTags/lastName",
                         data=['Y'.encode('UTF-8')])
        f.create_dataset("nirs/metaDataTags/sex",
                         data=['1'.encode('UTF-8')])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["middle_name"] == 'X'
    assert raw.info["subject_info"]["last_name"] == 'Y'
    assert raw.info["subject_info"]["sex"] == 1
    with h5py.File(fname, "r+") as f:
        del f['nirs/metaDataTags/sex']
        f.create_dataset("nirs/metaDataTags/sex",
                         data=['2'.encode('UTF-8')])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["sex"] == 2
    with h5py.File(fname, "r+") as f:
        del f['nirs/metaDataTags/sex']
        f.create_dataset("nirs/metaDataTags/sex",
                         data=['0'.encode('UTF-8')])
    raw = read_raw_snirf(fname, preload=True)
    assert raw.info["subject_info"]["sex"] == 0

    with h5py.File(fname, "r+") as f:
        f.create_dataset("nirs/metaDataTags/MNE_coordFrame", data=[1])


@requires_testing_data
@requires_h5py
def test_snirf_nirsport2():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(nirx_nirsport2_103, preload=True)

    # Test data import
    assert raw._data.shape == (92, 84)
    assert_almost_equal(raw.info['sfreq'], 7.6, decimal=1)

    # Test channel naming
    assert raw.info['ch_names'][:4] == ['S1_D1 760', 'S1_D1 850',
                                        'S1_D3 760', 'S1_D3 850']
    assert raw.info['ch_names'][24:26] == ['S6_D4 760', 'S6_D4 850']

    # Test frequency encoding
    assert raw.info['chs'][0]['loc'][9] == 760
    assert raw.info['chs'][1]['loc'][9] == 850

    assert sum(short_channels(raw.info)) == 16


@requires_testing_data
@requires_h5py
def test_snirf_standard():
    """Test standard operations."""
    _test_raw_reader(read_raw_snirf, fname=sfnirs_homer_103_wShort,
                     boundary_decimal=0)  # low fs


run_tests_if_main()
