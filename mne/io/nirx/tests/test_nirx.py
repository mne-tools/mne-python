# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          simplified BSD-3 license

import os.path as op
import shutil
import os
import datetime as dt
import numpy as np

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import pick_types
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_nirx
from mne.io.tests.test_raw import _test_raw_reader
from mne.preprocessing import annotate_nan
from mne.transforms import apply_trans, _get_trans
from mne.preprocessing.nirs import source_detector_distances,\
    short_channels, _reorder_nirx
from mne.io.constants import FIFF

testing_path = data_path(download=False)
fname_nirx_15_0 = op.join(
    testing_path, 'NIRx', 'nirscout', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(
    testing_path, 'NIRx', 'nirscout', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(
    testing_path, 'NIRx', 'nirscout', 'nirx_15_2_recording_w_short')
fname_nirx_15_3_short = op.join(
    testing_path, 'NIRx', 'nirscout', 'nirx_15_3_recording')


# This file has no saturated sections
nirsport1_wo_sat = op.join(testing_path, 'NIRx', 'nirsport_v1',
                           'nirx_15_3_recording_wo_saturation')
# This file has saturation, but not on the optode pairing in montage
nirsport1_w_sat = op.join(testing_path, 'NIRx', 'nirsport_v1',
                          'nirx_15_3_recording_w_saturation_'
                          'not_on_montage_channels')
# This file has saturation in channels of interest
nirsport1_w_fullsat = op.join(
    testing_path, 'NIRx', 'nirsport_v1', 'nirx_15_3_recording_w_'
    'saturation_on_montage_channels')

# NIRSport2 device using Aurora software
nirsport2 = op.join(
    testing_path, 'NIRx', 'nirsport_v2', 'aurora_recording _w_short_and_acc')
nirsport2_2021_9 = op.join(
    testing_path, 'NIRx', 'nirsport_v2', 'aurora_2021_9')
nirsport2_2021_9_6 = op.join(
    testing_path, 'NIRx', 'nirsport_v2', 'aurora_2021_9_6')


def test_nirsport_v2_matches_snirf(nirx_snirf):
    """Test NIRSport2 raw files return same data as snirf."""
    raw, raw_snirf = nirx_snirf
    _reorder_nirx(raw_snirf)
    assert raw.ch_names == raw_snirf.ch_names

    assert_allclose(raw._data, raw_snirf._data)

    # Check the timing of annotations match (naming is different)
    assert_allclose(raw.annotations.onset, raw_snirf.annotations.onset)

    assert_array_equal(raw.ch_names, raw_snirf.ch_names)

    # This test fails as snirf encodes name incorrectly.
    # assert raw.info["subject_info"]["first_name"] ==
    # raw_snirf.info["subject_info"]["first_name"]


@requires_testing_data
@pytest.mark.filterwarnings('ignore:.*Extraction of measurement.*:')
def test_nirsport_v2():
    """Test NIRSport2 file."""
    raw = read_raw_nirx(nirsport2, preload=True)
    assert raw._data.shape == (40, 128)

    # Test distance between optodes matches values from
    # nirsite https://github.com/mne-tools/mne-testing-data/pull/86
    # figure 3
    allowed_distance_error = 0.005
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S1_D1 760").info),
                    [0.0304], atol=allowed_distance_error)
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S2_D2 760").info),
                    [0.0400], atol=allowed_distance_error)

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
    locs = [ch['loc'][3:6] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][:2] == 'S1'
    assert_allclose(
        mni_locs[0], [-0.0848, -0.0162, -0.0163], atol=allowed_dist_error)

    assert raw.info['ch_names'][9][:2] == 'S2'
    assert_allclose(
        mni_locs[9], [-0.0, -0.1195, 0.0142], atol=allowed_dist_error)

    assert raw.info['ch_names'][39][:2] == 'S8'
    assert_allclose(
        mni_locs[34], [0.0828, -0.046, 0.0285], atol=allowed_dist_error)

    assert len(raw.annotations) == 3
    assert raw.annotations.description[0] == '1.0'
    assert raw.annotations.description[2] == '6.0'
    # Lose tolerance as I am eyeballing the time differences on screen
    assert_allclose(
        np.diff(raw.annotations.onset), [2.3, 3.1], atol=0.1)

    mon = raw.get_montage()
    assert len(mon.dig) == 27


@requires_testing_data
@pytest.mark.filterwarnings('ignore:.*Extraction of measurement.*:')
def test_nirsport_v1_wo_sat():
    """Test NIRSport1 file with no saturation."""
    raw = read_raw_nirx(nirsport1_wo_sat, preload=True)

    # Test data import
    assert raw._data.shape == (26, 164)
    assert raw.info['sfreq'] == 10.416667

    # By default real data is returned
    assert np.sum(np.isnan(raw.get_data())) == 0

    raw = read_raw_nirx(nirsport1_wo_sat, preload=True, saturated='nan')
    data = raw.get_data()
    assert data.shape == (26, 164)
    assert np.sum(np.isnan(data)) == 0

    raw = read_raw_nirx(nirsport1_wo_sat, saturated='annotate')
    data = raw.get_data()
    assert data.shape == (26, 164)
    assert np.sum(np.isnan(data)) == 0


@pytest.mark.filterwarnings('ignore:.*Extraction of measurement.*:')
@requires_testing_data
def test_nirsport_v1_w_sat():
    """Test NIRSport1 file with NaNs but not in channel of interest."""
    raw = read_raw_nirx(nirsport1_w_sat)

    # Test data import
    data = raw.get_data()
    assert data.shape == (26, 176)
    assert raw.info['sfreq'] == 10.416667
    assert np.sum(np.isnan(data)) == 0

    raw = read_raw_nirx(nirsport1_w_sat, saturated='nan')
    data = raw.get_data()
    assert data.shape == (26, 176)
    assert np.sum(np.isnan(data)) == 0

    raw = read_raw_nirx(nirsport1_w_sat, saturated='annotate')
    data = raw.get_data()
    assert data.shape == (26, 176)
    assert np.sum(np.isnan(data)) == 0


@pytest.mark.filterwarnings('ignore:.*Extraction of measurement.*:')
@requires_testing_data
@pytest.mark.parametrize('preload', (True, False))
@pytest.mark.parametrize('meas_date', (None, "orig"))
def test_nirsport_v1_w_bad_sat(preload, meas_date):
    """Test NIRSport1 file with NaNs."""
    fname = nirsport1_w_fullsat
    raw = read_raw_nirx(fname, preload=preload)
    data = raw.get_data()
    assert not np.isnan(data).any()
    assert len(raw.annotations) == 5
    # annotated version and ignore should have same data but different annot
    raw_ignore = read_raw_nirx(fname, saturated='ignore', preload=preload)
    assert_allclose(raw_ignore.get_data(), data)
    assert len(raw_ignore.annotations) == 2
    assert not any('NAN' in d for d in raw_ignore.annotations.description)
    # nan version should not have same data, but we can give it the same annot
    raw_nan = read_raw_nirx(fname, saturated='nan', preload=preload)
    data_nan = raw_nan.get_data()
    assert np.isnan(data_nan).any()
    assert not np.allclose(raw_nan.get_data(), data)
    raw_nan_annot = raw_ignore.copy()
    if meas_date is None:
        raw.set_meas_date(None)
        raw_nan.set_meas_date(None)
        raw_nan_annot.set_meas_date(None)
    nan_annots = annotate_nan(raw_nan)
    assert nan_annots.orig_time == raw_nan.info["meas_date"]
    raw_nan_annot.set_annotations(nan_annots)
    use_mask = np.where(raw.annotations.description == 'BAD_SATURATED')
    for key in ('onset', 'duration'):
        a = getattr(raw_nan_annot.annotations, key)[::2]  # one ch in each
        b = getattr(raw.annotations, key)[use_mask]  # two chs in each
        assert_allclose(a, b)


@requires_testing_data
def test_nirx_hdr_load():
    """Test reading NIRX files using path to header file."""
    fname = fname_nirx_15_2_short + "/NIRS-2019-08-23_001.hdr"
    raw = read_raw_nirx(fname, preload=True)

    # Test data import
    assert raw._data.shape == (26, 145)
    assert raw.info['sfreq'] == 12.5


@requires_testing_data
def test_nirx_missing_warn():
    """Test reading NIRX files when missing data."""
    with pytest.raises(FileNotFoundError, match='does not exist'):
        read_raw_nirx(fname_nirx_15_2_short + "1", preload=True)


@requires_testing_data
def test_nirx_missing_evt(tmp_path):
    """Test reading NIRX files when missing data."""
    shutil.copytree(fname_nirx_15_2_short, str(tmp_path) + "/data/")
    os.rename(tmp_path / "data" / "NIRS-2019-08-23_001.evt",
              tmp_path / "data" / "NIRS-2019-08-23_001.xxx")
    fname = tmp_path / "data" / "NIRS-2019-08-23_001.hdr"
    raw = read_raw_nirx(fname, preload=True)
    assert raw.annotations.onset.shape == (0, )


@requires_testing_data
def test_nirx_dat_warn(tmp_path):
    """Test reading NIRX files when missing data."""
    shutil.copytree(fname_nirx_15_2_short, str(tmp_path) + "/data/")
    os.rename(tmp_path / "data" / "NIRS-2019-08-23_001.dat",
              tmp_path / "data" / "NIRS-2019-08-23_001.tmp")
    fname = tmp_path / "data" / "NIRS-2019-08-23_001.hdr"
    with pytest.warns(RuntimeWarning, match='A single dat'):
        read_raw_nirx(fname, preload=True)


@requires_testing_data
def test_nirx_15_2_short():
    """Test reading NIRX files."""
    raw = read_raw_nirx(fname_nirx_15_2_short, preload=True)

    # Test data import
    assert raw._data.shape == (26, 145)
    assert raw.info['sfreq'] == 12.5
    assert raw.info['meas_date'] == dt.datetime(2019, 8, 23, 7, 37, 4, 540000,
                                                tzinfo=dt.timezone.utc)

    # Test channel naming
    assert raw.info['ch_names'][:4] == ["S1_D1 760", "S1_D1 850",
                                        "S1_D9 760", "S1_D9 850"]
    assert raw.info['ch_names'][24:26] == ["S5_D13 760", "S5_D13 850"]

    # Test frequency encoding
    assert raw.info['chs'][0]['loc'][9] == 760
    assert raw.info['chs'][1]['loc'][9] == 850

    # Test info import
    assert raw.info['subject_info'] == dict(sex=1, first_name="MNE",
                                            middle_name="Test",
                                            last_name="Recording",
                                            birthday=(2014, 8, 23),
                                            his_id="MNE_Test_Recording")

    # Test distance between optodes matches values from
    # nirsite https://github.com/mne-tools/mne-testing-data/pull/51
    # step 4 figure 2
    allowed_distance_error = 0.0002
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S1_D1 760").info),
                    [0.0304], atol=allowed_distance_error)
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S2_D10 760").info),
                    [0.0086], atol=allowed_distance_error)

    # Test which channels are short
    # These are the ones marked as red at
    # https://github.com/mne-tools/mne-testing-data/pull/51 step 4 figure 2
    is_short = short_channels(raw.info)
    assert_array_equal(is_short[:9:2], [False, True, False, True, False])
    is_short = short_channels(raw.info, threshold=0.003)
    assert_array_equal(is_short[:3:2], [False, False])
    is_short = short_channels(raw.info, threshold=50)
    assert_array_equal(is_short[:3:2], [True, True])

    # Test trigger events
    assert_array_equal(raw.annotations.description, ['3.0', '2.0', '1.0'])

    # Test location of detectors
    # The locations of detectors can be seen in the first
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/51
    # And have been manually copied below
    # These values were reported in mm, but according to this page...
    # https://mne.tools/stable/auto_tutorials/intro/plot_40_sensor_locations.html
    # 3d locations should be specified in meters, so that's what's tested below
    # Detector locations are stored in the third three loc values
    allowed_dist_error = 0.0002
    locs = [ch['loc'][6:9] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][3:5] == 'D1'
    assert_allclose(
        mni_locs[0], [-0.0841, -0.0464, -0.0129], atol=allowed_dist_error)

    assert raw.info['ch_names'][4][3:5] == 'D3'
    assert_allclose(
        mni_locs[4], [0.0846, -0.0142, -0.0156], atol=allowed_dist_error)

    assert raw.info['ch_names'][8][3:5] == 'D2'
    assert_allclose(
        mni_locs[8], [0.0207, -0.1062, 0.0484], atol=allowed_dist_error)

    assert raw.info['ch_names'][12][3:5] == 'D4'
    assert_allclose(
        mni_locs[12], [-0.0196, 0.0821, 0.0275], atol=allowed_dist_error)

    assert raw.info['ch_names'][16][3:5] == 'D5'
    assert_allclose(
        mni_locs[16], [-0.0360, 0.0276, 0.0778], atol=allowed_dist_error)

    assert raw.info['ch_names'][19][3:5] == 'D6'
    assert_allclose(
        mni_locs[19], [0.0352, 0.0283, 0.0780], atol=allowed_dist_error)

    assert raw.info['ch_names'][21][3:5] == 'D7'
    assert_allclose(
        mni_locs[21], [0.0388, -0.0477, 0.0932], atol=allowed_dist_error)


@requires_testing_data
def test_nirx_15_3_short():
    """Test reading NIRX files."""
    raw = read_raw_nirx(fname_nirx_15_3_short, preload=True)

    # Test data import
    assert raw._data.shape == (26, 220)
    assert raw.info['sfreq'] == 12.5

    # Test channel naming
    assert raw.info['ch_names'][:4] == ["S1_D2 760", "S1_D2 850",
                                        "S1_D9 760", "S1_D9 850"]
    assert raw.info['ch_names'][24:26] == ["S5_D13 760", "S5_D13 850"]

    # Test frequency encoding
    assert raw.info['chs'][0]['loc'][9] == 760
    assert raw.info['chs'][1]['loc'][9] == 850

    # Test info import
    assert raw.info['subject_info'] == dict(birthday=(2020, 8, 18),
                                            sex=0,
                                            first_name="testMontage\\0A"
                                                       "TestMontage",
                                            his_id="testMontage\\0A"
                                                   "TestMontage")

    # Test distance between optodes matches values from
    # https://github.com/mne-tools/mne-testing-data/pull/72
    allowed_distance_error = 0.001
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S1_D2 760").info),
                    [0.0304], atol=allowed_distance_error)
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S5_D13 760").info),
                    [0.0076], atol=allowed_distance_error)

    # Test which channels are short
    # These are the ones marked as red at
    # https://github.com/mne-tools/mne-testing-data/pull/72
    is_short = short_channels(raw.info)
    assert_array_equal(is_short[:9:2], [False, True, False, True, False])
    is_short = short_channels(raw.info, threshold=0.003)
    assert_array_equal(is_short[:3:2], [False, False])
    is_short = short_channels(raw.info, threshold=50)
    assert_array_equal(is_short[:3:2], [True, True])

    # Test trigger events
    assert_array_equal(raw.annotations.description, ['4.0', '2.0', '1.0'])

    # Test location of detectors
    # The locations of detectors can be seen in the first
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/72
    # And have been manually copied below
    allowed_dist_error = 0.0002
    locs = [ch['loc'][6:9] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][3:5] == 'D2'
    assert_allclose(
        mni_locs[0], [-0.0841, -0.0464, -0.0129], atol=allowed_dist_error)

    assert raw.info['ch_names'][4][3:5] == 'D1'
    assert_allclose(
        mni_locs[4], [0.0846, -0.0142, -0.0156], atol=allowed_dist_error)

    assert raw.info['ch_names'][8][3:5] == 'D3'
    assert_allclose(
        mni_locs[8], [0.0207, -0.1062, 0.0484], atol=allowed_dist_error)

    assert raw.info['ch_names'][12][3:5] == 'D4'
    assert_allclose(
        mni_locs[12], [-0.0196, 0.0821, 0.0275], atol=allowed_dist_error)

    assert raw.info['ch_names'][16][3:5] == 'D5'
    assert_allclose(
        mni_locs[16], [-0.0360, 0.0276, 0.0778], atol=allowed_dist_error)

    assert raw.info['ch_names'][19][3:5] == 'D6'
    assert_allclose(
        mni_locs[19], [0.0388, -0.0477, 0.0932], atol=allowed_dist_error)

    assert raw.info['ch_names'][21][3:5] == 'D7'
    assert_allclose(
        mni_locs[21], [-0.0394, -0.0483, 0.0928], atol=allowed_dist_error)


@requires_testing_data
def test_locale_encoding(tmp_path):
    """Test NIRx encoding."""
    fname = tmp_path / 'latin'
    shutil.copytree(fname_nirx_15_2, fname)
    hdr_fname = op.join(fname, 'NIRS-2019-10-02_003.hdr')
    hdr = list()
    with open(hdr_fname, 'rb') as fid:
        hdr.extend(line for line in fid)
    # French
    hdr[2] = b'Date="jeu. 13 f\xe9vr. 2020"\r\n'
    with open(hdr_fname, 'wb') as fid:
        for line in hdr:
            fid.write(line)
    read_raw_nirx(fname, verbose='debug')
    # German
    hdr[2] = b'Date="mi 13 dez 2020"\r\n'
    with open(hdr_fname, 'wb') as fid:
        for line in hdr:
            fid.write(line)
    read_raw_nirx(fname, verbose='debug')
    # Italian
    hdr[2] = b'Date="ven 24 gen 2020"\r\n'
    hdr[3] = b'Time="10:57:41.454"\r\n'
    with open(hdr_fname, 'wb') as fid:
        for line in hdr:
            fid.write(line)
    raw = read_raw_nirx(fname, verbose='debug')
    want_dt = dt.datetime(
        2020, 1, 24, 10, 57, 41, 454000, tzinfo=dt.timezone.utc)
    assert raw.info['meas_date'] == want_dt


@requires_testing_data
def test_nirx_15_2():
    """Test reading NIRX files."""
    raw = read_raw_nirx(fname_nirx_15_2, preload=True)

    # Test data import
    assert raw._data.shape == (64, 67)
    assert raw.info['sfreq'] == 3.90625
    assert raw.info['meas_date'] == dt.datetime(2019, 10, 2, 9, 8, 47, 511000,
                                                tzinfo=dt.timezone.utc)

    # Test channel naming
    assert raw.info['ch_names'][:4] == ["S1_D1 760", "S1_D1 850",
                                        "S1_D10 760", "S1_D10 850"]

    # Test info import
    assert raw.info['subject_info'] == dict(sex=1, first_name="TestRecording",
                                            birthday=(1989, 10, 2),
                                            his_id="TestRecording")

    # Test trigger events
    assert_array_equal(raw.annotations.description, ['4.0', '6.0', '2.0'])
    print(raw.annotations.onset)

    # Test location of detectors
    allowed_dist_error = 0.0002
    locs = [ch['loc'][6:9] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][3:5] == 'D1'
    assert_allclose(
        mni_locs[0], [-0.0292, 0.0852, -0.0142], atol=allowed_dist_error)

    assert raw.info['ch_names'][15][3:5] == 'D4'
    assert_allclose(
        mni_locs[15], [-0.0739, -0.0756, -0.0075], atol=allowed_dist_error)

    # Old name aliases for backward compat
    assert 'fnirs_cw_amplitude' in raw
    with pytest.raises(ValueError, match='Invalid value'):
        'fnirs_raw' in raw
    assert 'fnirs_od' not in raw
    picks = pick_types(raw.info, fnirs='fnirs_cw_amplitude')
    assert len(picks) > 0


@requires_testing_data
def test_nirx_aurora_2021_9_6():
    """Test reading NIRX files."""
    raw = read_raw_nirx(nirsport2_2021_9_6, preload=True)
    assert len(raw.annotations) == 3
    assert raw.annotations.description[0] == "1.0"
    assert raw.annotations.description[2] == "3.0"


@requires_testing_data
def test_nirx_15_0():
    """Test reading NIRX files."""
    raw = read_raw_nirx(fname_nirx_15_0, preload=True)

    # Test data import
    assert raw._data.shape == (20, 92)
    assert raw.info['sfreq'] == 6.25
    assert raw.info['meas_date'] == dt.datetime(2019, 10, 27, 13, 53, 34,
                                                209000,
                                                tzinfo=dt.timezone.utc)

    # Test channel naming
    assert raw.info['ch_names'][:12] == ["S1_D1 760", "S1_D1 850",
                                         "S2_D2 760", "S2_D2 850",
                                         "S3_D3 760", "S3_D3 850",
                                         "S4_D4 760", "S4_D4 850",
                                         "S5_D5 760", "S5_D5 850",
                                         "S6_D6 760", "S6_D6 850"]

    # Test info import
    assert raw.info['subject_info'] == {'birthday': (2004, 10, 27),
                                        'first_name': 'NIRX',
                                        'last_name': 'Test',
                                        'sex': FIFF.FIFFV_SUBJ_SEX_UNKNOWN,
                                        'his_id': "NIRX_Test"}

    # Test trigger events
    assert_array_equal(raw.annotations.description, ['1.0', '2.0', '2.0'])

    # Test location of detectors
    allowed_dist_error = 0.0002
    locs = [ch['loc'][6:9] for ch in raw.info['chs']]
    head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
    mni_locs = apply_trans(head_mri_t, locs)

    assert raw.info['ch_names'][0][3:5] == 'D1'
    assert_allclose(
        mni_locs[0], [0.0287, -0.1143, -0.0332], atol=allowed_dist_error)

    assert raw.info['ch_names'][15][3:5] == 'D8'
    assert_allclose(
        mni_locs[15], [-0.0693, -0.0480, 0.0657], atol=allowed_dist_error)

    # Test distance between optodes matches values from
    allowed_distance_error = 0.0002

    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S1_D1 760").info),
                    [0.0300], atol=allowed_distance_error)
    assert_allclose(source_detector_distances(raw.copy().
                                              pick("S7_D7 760").info),
                    [0.0392], atol=allowed_distance_error)


@requires_testing_data
@pytest.mark.parametrize('fname, boundary_decimal', (
    [fname_nirx_15_2_short, 1],
    [fname_nirx_15_2, 0],
    [fname_nirx_15_2, 0],
    [nirsport2_2021_9, 0],
))
def test_nirx_standard(fname, boundary_decimal):
    """Test standard operations."""
    _test_raw_reader(read_raw_nirx, fname=fname,
                     boundary_decimal=boundary_decimal)  # low fs


# Below are the native (on-disk) orders, which should be preserved
@requires_testing_data
@pytest.mark.parametrize('fname, want_order', [
    (fname_nirx_15_0, ['S1_D1', 'S2_D2', 'S3_D3', 'S4_D4', 'S5_D5', 'S6_D6', 'S7_D7', 'S8_D8', 'S9_D9', 'S10_D10']),  # noqa: E501
    (fname_nirx_15_2, ['S1_D1', 'S1_D10', 'S2_D1', 'S2_D2', 'S3_D2', 'S3_D3', 'S4_D3', 'S4_D4', 'S5_D4', 'S5_D5', 'S6_D5', 'S6_D6', 'S7_D6', 'S7_D7', 'S8_D7', 'S8_D8', 'S9_D8', 'S9_D9', 'S10_D9', 'S10_D10', 'S11_D11', 'S11_D12', 'S12_D12', 'S12_D13', 'S13_D13', 'S13_D14', 'S14_D14', 'S14_D15', 'S15_D15', 'S15_D16', 'S16_D11', 'S16_D16']),  # noqa: E501
    (fname_nirx_15_2_short, ['S1_D1', 'S1_D9', 'S2_D3', 'S2_D10', 'S3_D2', 'S3_D11', 'S4_D4', 'S4_D12', 'S5_D5', 'S5_D6', 'S5_D7', 'S5_D8', 'S5_D13']),  # noqa: E501
    (fname_nirx_15_3_short, ['S1_D2', 'S1_D9', 'S2_D1', 'S2_D10', 'S3_D3', 'S3_D11', 'S4_D4', 'S4_D12', 'S5_D5', 'S5_D6', 'S5_D7', 'S5_D8', 'S5_D13']),  # noqa: E501
    (nirsport1_wo_sat, ['S1_D4', 'S1_D5', 'S1_D6', 'S2_D5', 'S2_D6', 'S3_D5', 'S4_D1', 'S4_D3', 'S4_D4', 'S5_D1', 'S5_D2', 'S6_D1', 'S6_D3']),  # noqa: E501
    (nirsport2, ['S1_D1', 'S1_D6', 'S1_D9', 'S2_D2', 'S2_D10', 'S3_D5', 'S3_D7', 'S3_D11', 'S4_D8', 'S4_D12', 'S5_D3', 'S5_D13', 'S6_D4', 'S6_D14', 'S7_D1', 'S7_D6', 'S7_D15', 'S8_D5', 'S8_D7', 'S8_D16']),  # noqa: E501
    (nirsport2_2021_9, ['S1_D1', 'S1_D3', 'S2_D1', 'S2_D2', 'S2_D4', 'S3_D2', 'S3_D5', 'S4_D1', 'S4_D3', 'S4_D4', 'S4_D6', 'S5_D2', 'S5_D4', 'S5_D5', 'S5_D7', 'S6_D3', 'S6_D6', 'S7_D4', 'S7_D6', 'S7_D7', 'S8_D5', 'S8_D7']),  # noqa: E501
])
def test_channel_order(fname, want_order):
    """Test that logical channel order is preserved."""
    raw = read_raw_nirx(fname)
    ch_names = raw.ch_names
    prefixes = [ch_name.split()[0] for ch_name in ch_names]
    assert prefixes[::2] == prefixes[1::2]
    prefixes = prefixes[::2]
    assert prefixes == want_order
