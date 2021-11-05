# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import shutil
import pytest

from mne.datasets.testing import data_path, requires_testing_data
from mne.utils import requires_h5py
from mne.io import read_raw_snirf, read_raw_nirx
from mne.io.tests.test_raw import _test_raw_reader
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    short_channels, source_detector_distances)
from mne.transforms import apply_trans, _get_trans
from mne.io.constants import FIFF

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

# NIRSport2 files
nirx_nirsport2_103 = op.join(data_path(download=False),
                             'SNIRF', 'NIRx', 'NIRSport2', '1.0.3',
                             '2021-04-23_005.snirf')
nirx_nirsport2_103_2 = op.join(data_path(download=False),
                               'SNIRF', 'NIRx', 'NIRSport2', '1.0.3',
                               '2021-05-05_001.snirf')
snirf_nirsport2_20219 = op.join(data_path(download=False),
                                'SNIRF', 'NIRx', 'NIRSport2', '2021.9',
                                '2021-10-01_002.snirf')
nirx_nirsport2_20219 = op.join(data_path(download=False), 'NIRx',
                               'nirsport_v2', 'aurora_2021_9')


@requires_h5py
@requires_testing_data
@pytest.mark.filterwarnings('ignore:.*contains 2D location.*:')
@pytest.mark.parametrize('fname', ([sfnirs_homer_103_wShort,
                                    nirx_nirsport2_103,
                                    sfnirs_homer_103_153,
                                    nirx_nirsport2_103,
                                    nirx_nirsport2_103_2,
                                    snirf_nirsport2_20219]))
def test_basic_reading_and_min_process(fname):
    """Test reading SNIRF files and minimum typical processing."""
    raw = read_raw_snirf(fname, preload=True)
    # SNIRF data can contain several types, so only apply appropriate functions
    if 'fnirs_cw_amplitude' in raw:
        raw = optical_density(raw)
    if 'fnirs_od' in raw:
        raw = beer_lambert_law(raw, ppf=6)
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
def test_snirf_nonstandard(tmp_path):
    """Test custom tags."""
    from mne.externals.pymatreader.utils import _import_h5py
    h5py = _import_h5py()
    shutil.copy(sfnirs_homer_103_wShort, str(tmp_path) + "/mod.snirf")
    fname = str(tmp_path) + "/mod.snirf"
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
def test_snirf_coordframe():
    """Test reading SNIRF files."""
    raw = read_raw_snirf(nirx_nirsport2_103, optode_frame="head").\
        info['chs'][3]['coord_frame']
    assert raw == FIFF.FIFFV_COORD_HEAD

    raw = read_raw_snirf(nirx_nirsport2_103, optode_frame="mri").\
        info['chs'][3]['coord_frame']
    assert raw == FIFF.FIFFV_COORD_HEAD

    raw = read_raw_snirf(nirx_nirsport2_103, optode_frame="unknown").\
        info['chs'][3]['coord_frame']
    assert raw == FIFF.FIFFV_COORD_UNKNOWN


@requires_testing_data
@requires_h5py
def test_snirf_nirsport2_w_positions():
    """Test reading SNIRF files with known positions."""
    raw = read_raw_snirf(nirx_nirsport2_103_2, preload=True,
                         optode_frame="mri")

    # Test data import
    assert raw._data.shape == (40, 128)
    assert_almost_equal(raw.info['sfreq'], 10.2, decimal=1)

    # Test channel naming
    assert raw.info['ch_names'][:4] == ['S1_D1 760', 'S1_D1 850',
                                        'S1_D6 760', 'S1_D6 850']
    assert raw.info['ch_names'][24:26] == ['S6_D4 760', 'S6_D4 850']

    # Test frequency encoding
    assert raw.info['chs'][0]['loc'][9] == 760
    assert raw.info['chs'][1]['loc'][9] == 850

    assert sum(short_channels(raw.info)) == 16

    # Test distance between optodes matches values from
    # nirsite https://github.com/mne-tools/mne-testing-data/pull/86
    # figure 3
    allowed_distance_error = 0.005
    distances = source_detector_distances(raw.info)
    assert_allclose(distances[::2][:14],
                    [0.0304, 0.0411, 0.008, 0.0400, 0.008, 0.0310, 0.0411,
                     0.008, 0.0299, 0.008, 0.0370, 0.008, 0.0404, 0.008],
                    atol=allowed_distance_error)

    # Test location of detectors
    # The locations of detectors can be seen in the first
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/86
    allowed_dist_error = 0.0002
    locs = [ch['loc'][6:9] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][3:5] == 'D1'
    assert_allclose(
        mni_locs[0], [-0.0841, -0.0464, -0.0129], atol=allowed_dist_error)

    assert raw.info['ch_names'][2][3:5] == 'D6'
    assert_allclose(
        mni_locs[2], [-0.0841, -0.0138, 0.0248], atol=allowed_dist_error)

    assert raw.info['ch_names'][34][3:5] == 'D5'
    assert_allclose(
        mni_locs[34], [0.0845, -0.0451, -0.0123], atol=allowed_dist_error)

    # Test location of sensors
    # The locations of sensors can be seen in the second
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/86
    allowed_dist_error = 0.0002
    locs = [ch['loc'][3:6] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][:2] == 'S1'
    assert_allclose(
        mni_locs[0], [-0.0848, -0.0162, -0.0163], atol=allowed_dist_error)

    assert raw.info['ch_names'][9][:2] == 'S2'
    assert_allclose(
        mni_locs[9], [-0.0, -0.1195, 0.0142], atol=allowed_dist_error)

    assert raw.info['ch_names'][34][:2] == 'S8'
    assert_allclose(
        mni_locs[34], [0.0828, -0.046, 0.0285], atol=allowed_dist_error)

    mon = raw.get_montage()
    assert len(mon.dig) == 43


@requires_testing_data
@requires_h5py
@pytest.mark.parametrize('fname, boundary_decimal, test_scaling, test_rank', (
    [sfnirs_homer_103_wShort, 0, True, True],
    [nirx_nirsport2_103, 0, True, False],  # strange rank behavior
    [nirx_nirsport2_103_2, 0, False, True],  # weirdly small values
    [snirf_nirsport2_20219, 0, True, True],
))
def test_snirf_standard(fname, boundary_decimal, test_scaling, test_rank):
    """Test standard operations."""
    _test_raw_reader(read_raw_snirf, fname=fname,
                     boundary_decimal=boundary_decimal,
                     test_scaling=test_scaling,
                     test_rank=test_rank)  # low fs


@requires_testing_data
@requires_h5py
def test_annotation_description_from_stim_groups():
    """Test annotation descriptions parsed from stim group names."""
    raw = read_raw_snirf(nirx_nirsport2_103_2, preload=True)
    expected_descriptions = ['1', '2', '6']
    assert_equal(expected_descriptions, raw.annotations.description)
