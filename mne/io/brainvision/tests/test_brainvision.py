# -*- coding: utf-8 -*-
"""Test reading of BrainVision format."""
# Author: Teon Brooks <teon.brooks@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
from os import unlink
import shutil

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest
from tempfile import NamedTemporaryFile

from mne.utils import _TempDir, run_tests_if_main
from mne import pick_types, read_annotations, concatenate_raws
from mne.io.constants import FIFF
from mne.io import read_raw_fif, read_raw_brainvision
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing
from mne.annotations import events_from_annotations

data_dir = op.join(op.dirname(__file__), 'data')
vhdr_path = op.join(data_dir, 'test.vhdr')
vmrk_path = op.join(data_dir, 'test.vmrk')
eeg_path = op.join(data_dir, 'test.eeg')

vhdr_partially_disabled_hw_filter_path = op.join(data_dir,
                                                 'test_partially_disabled'
                                                 '_hw_filter.vhdr')

vhdr_old_path = op.join(data_dir,
                        'test_old_layout_latin1_software_filter.vhdr')

vhdr_v2_path = op.join(data_dir, 'testv2.vhdr')

vhdr_highpass_path = op.join(data_dir, 'test_highpass.vhdr')
vhdr_mixed_highpass_path = op.join(data_dir, 'test_mixed_highpass.vhdr')
vhdr_highpass_hz_path = op.join(data_dir, 'test_highpass_hz.vhdr')
vhdr_mixed_highpass_hz_path = op.join(data_dir, 'test_mixed_highpass_hz.vhdr')

# Not a typo: we can reuse the highpass file for the lowpass (Hz) test
vhdr_lowpass_path = op.join(data_dir, 'test_highpass.vhdr')
vhdr_mixed_lowpass_path = op.join(data_dir, 'test_mixed_lowpass.vhdr')
vhdr_lowpass_s_path = op.join(data_dir, 'test_lowpass_s.vhdr')
vhdr_mixed_lowpass_s_path = op.join(data_dir, 'test_mixed_lowpass_s.vhdr')

# VHDR exported with neuroone
data_path = testing.data_path(download=False)
neuroone_vhdr = op.join(data_path, 'Brainvision', 'test_NO.vhdr')

# Test for nanovolts as unit
vhdr_nV_path = op.join(data_dir, 'test_nV.vhdr')

# Test bad date
vhdr_bad_date = op.join(data_dir, 'test_bad_date.vhdr')

montage = op.join(data_dir, 'test.hpts')
eeg_bin = op.join(data_dir, 'test_bin_raw.fif')
eog = ['HL', 'HR', 'Vb']


def test_orig_units(recwarn):
    """Test exposure of original channel units."""
    raw = read_raw_brainvision(vhdr_path)
    orig_units = raw._orig_units
    assert len(orig_units) == 32
    assert orig_units['FP1'] == u'µV'
    assert orig_units['CP5'] == 'n/a'  # originally BS, not a valid unit
    assert orig_units['CP6'] == u'µS'
    assert orig_units['HL'] == 'n/a'  # originally ARU, not a valid unit
    assert orig_units['HR'] == 'n/a'  # originally uS ...
    assert orig_units['Vb'] == 'S'
    assert orig_units['ReRef'] == 'C'


def test_vmrk_meas_date():
    """Test successful extraction of measurement date."""
    # Test file that does have a specific date
    raw = read_raw_brainvision(vhdr_path)
    assert_allclose(raw.info['meas_date'], [1384359243, 794231])
    assert '2013-11-13 16:14:03 GMT' in repr(raw.info)

    # Test file with multiple dates ... we should only take the first
    with pytest.warns(RuntimeWarning, match='software filter'):
        raw = read_raw_brainvision(vhdr_old_path)
    assert_allclose(raw.info['meas_date'], [1184588560, 937453])
    assert '2007-07-16 12:22:40 GMT' in repr(raw.info)

    # Test files with no date, we should get DATE_NONE from mne.io.write
    with pytest.warns(RuntimeWarning, match='coordinate information'):
        raw = read_raw_brainvision(vhdr_v2_path)
    assert raw.info['meas_date'] is None
    assert 'unspecified' in repr(raw.info)

    # Test files with faulty dates introduced by segmenting a file without
    # date information. Should not raise a strptime ValueError
    raw = read_raw_brainvision(vhdr_bad_date)
    assert raw.info['meas_date'] is None
    assert 'unspecified' in repr(raw.info)


def test_vhdr_codepage_ansi():
    """Test BV reading with ANSI codepage."""
    raw_init = read_raw_brainvision(vhdr_path)
    data_expected, times_expected = raw_init[:]
    tempdir = _TempDir()
    ansi_vhdr_path = op.join(tempdir, op.split(vhdr_path)[-1])
    ansi_vmrk_path = op.join(tempdir, op.split(vmrk_path)[-1])
    ansi_eeg_path = op.join(tempdir, op.split(eeg_path)[-1])
    # copy data file
    shutil.copy(eeg_path, ansi_eeg_path)
    # modify header file
    with open(ansi_vhdr_path, 'wb') as fout:
        with open(vhdr_path, 'rb') as fin:
            for line in fin:
                # Common Infos section
                if line.startswith(b'Codepage'):
                    line = b'Codepage=ANSI\n'
                fout.write(line)
    # modify marker file
    with open(ansi_vmrk_path, 'wb') as fout:
        with open(vmrk_path, 'rb') as fin:
            for line in fin:
                # Common Infos section
                if line.startswith(b'Codepage'):
                    line = b'Codepage=ANSI\n'
                fout.write(line)

    raw = read_raw_brainvision(ansi_vhdr_path)
    data_new, times_new = raw[:]

    assert_equal(raw_init.ch_names, raw.ch_names)
    assert_allclose(data_new, data_expected, atol=1e-15)
    assert_allclose(times_new, times_expected, atol=1e-15)


def test_ascii():
    """Test ASCII BV reading."""
    raw = read_raw_brainvision(vhdr_path)
    tempdir = _TempDir()
    ascii_vhdr_path = op.join(tempdir, op.split(vhdr_path)[-1])
    # copy marker file
    shutil.copy(vhdr_path.replace('.vhdr', '.vmrk'),
                ascii_vhdr_path.replace('.vhdr', '.vmrk'))
    # modify header file
    skipping = False
    with open(ascii_vhdr_path, 'wb') as fout:
        with open(vhdr_path, 'rb') as fin:
            for line in fin:
                # Common Infos section
                if line.startswith(b'DataFormat'):
                    line = b'DataFormat=ASCII\n'
                elif line.startswith(b'DataFile='):
                    line = b'DataFile=test.dat\n'
                # Replace the "'Binary Infos'" section
                elif line.startswith(b'[Binary Infos]'):
                    skipping = True
                    fout.write(b'[ASCII Infos]\nDecimalSymbol=.\nSkipLines=1\n'
                               b'SkipColumns=0\n\n')
                elif skipping and line.startswith(b'['):
                    skipping = False
                if not skipping:
                    fout.write(line)
    # create the .dat file
    data, times = raw[:]
    with open(ascii_vhdr_path.replace('.vhdr', '.dat'), 'wb') as fid:
        fid.write(b' '.join(ch_name.encode('ASCII')
                            for ch_name in raw.ch_names) + b'\n')
        fid.write(b'\n'.join(b' '.join(b'%.3f' % dd for dd in d)
                             for d in data.T / raw._cals))
    raw = read_raw_brainvision(ascii_vhdr_path)
    data_new, times_new = raw[:]
    assert_allclose(data_new, data, atol=1e-15)
    assert_allclose(times_new, times)


def test_brainvision_data_highpass_filters():
    """Test reading raw Brain Vision files with amplifier filter settings."""
    # Homogeneous highpass in seconds (default measurement unit)
    raw = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_highpass_path,
        montage=montage, eog=eog)

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 250.)

    # Heterogeneous highpass in seconds (default measurement unit)
    with pytest.warns(RuntimeWarning, match='different .*pass filters') as w:
        raw = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_mixed_highpass_path,
            montage=montage, eog=eog)

    lowpass_warning = ['different lowpass filters' in str(ww.message)
                       for ww in w]
    highpass_warning = ['different highpass filters' in str(ww.message)
                        for ww in w]

    expected_warnings = zip(lowpass_warning, highpass_warning)

    assert (all(any([lp, hp]) for lp, hp in expected_warnings))

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 250.)

    # Homogeneous highpass in Hertz
    raw = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_highpass_hz_path,
        montage=montage, eog=eog)

    assert_equal(raw.info['highpass'], 10.)
    assert_equal(raw.info['lowpass'], 250.)

    # Heterogeneous highpass in Hertz
    with pytest.warns(RuntimeWarning, match='different .*pass filters') as w:
        raw = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_mixed_highpass_hz_path,
            montage=montage, eog=eog)

    trigger_warning = ['will be dropped' in str(ww.message)
                       for ww in w]
    lowpass_warning = ['different lowpass filters' in str(ww.message)
                       for ww in w]
    highpass_warning = ['different highpass filters' in str(ww.message)
                        for ww in w]

    expected_warnings = zip(trigger_warning, lowpass_warning, highpass_warning)

    assert (all(any([trg, lp, hp]) for trg, lp, hp in expected_warnings))

    assert_equal(raw.info['highpass'], 5.)
    assert_equal(raw.info['lowpass'], 250.)


def test_brainvision_data_lowpass_filters():
    """Test files with amplifier LP filter settings."""
    # Homogeneous lowpass in Hertz (default measurement unit)
    raw = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_lowpass_path,
        montage=montage, eog=eog)

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 250.)

    # Heterogeneous lowpass in Hertz (default measurement unit)
    with pytest.warns(RuntimeWarning) as w:  # event parsing
        raw = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_mixed_lowpass_path,
            montage=montage, eog=eog)

    lowpass_warning = ['different lowpass filters' in str(ww.message)
                       for ww in w]
    highpass_warning = ['different highpass filters' in str(ww.message)
                        for ww in w]

    expected_warnings = zip(lowpass_warning, highpass_warning)

    assert (all(any([lp, hp]) for lp, hp in expected_warnings))

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 250.)

    # Homogeneous lowpass in seconds
    raw = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_lowpass_s_path,
        montage=montage, eog=eog)

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 1. / (2 * np.pi * 0.004))

    # Heterogeneous lowpass in seconds
    with pytest.warns(RuntimeWarning) as w:  # filter settings
        raw = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_mixed_lowpass_s_path,
            montage=montage, eog=eog)

    lowpass_warning = ['different lowpass filters' in str(ww.message)
                       for ww in w]
    highpass_warning = ['different highpass filters' in str(ww.message)
                        for ww in w]

    expected_warnings = zip(lowpass_warning, highpass_warning)

    assert (all(any([lp, hp]) for lp, hp in expected_warnings))

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 1. / (2 * np.pi * 0.004))


def test_brainvision_data_partially_disabled_hw_filters():
    """Test heterogeneous filter settings including non-numeric values."""
    with pytest.warns(RuntimeWarning) as w:  # event parsing
        raw = _test_raw_reader(
            read_raw_brainvision,
            vhdr_fname=vhdr_partially_disabled_hw_filter_path,
            montage=montage, eog=eog)

    trigger_warning = ['will be dropped' in str(ww.message)
                       for ww in w]
    lowpass_warning = ['different lowpass filters' in str(ww.message)
                       for ww in w]
    highpass_warning = ['different highpass filters' in str(ww.message)
                        for ww in w]

    expected_warnings = zip(trigger_warning, lowpass_warning, highpass_warning)

    assert (all(any([trg, lp, hp]) for trg, lp, hp in expected_warnings))

    assert_equal(raw.info['highpass'], 0.)
    assert_equal(raw.info['lowpass'], 500.)


def test_brainvision_data_software_filters_latin1_global_units():
    """Test reading raw Brain Vision files."""
    with pytest.warns(RuntimeWarning, match='software filter'):
        raw = _test_raw_reader(
            read_raw_brainvision, vhdr_fname=vhdr_old_path,
            eog=("VEOGo", "VEOGu", "HEOGli", "HEOGre"), misc=("A2",))

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 0.9))
    assert_equal(raw.info['lowpass'], 50.)


def test_brainvision_data():
    """Test reading raw Brain Vision files."""
    pytest.raises(IOError, read_raw_brainvision, vmrk_path)
    pytest.raises(ValueError, read_raw_brainvision, vhdr_path, montage,
                  preload=True, scale="foo")

    raw_py = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_path, montage=montage,
        eog=eog, misc='auto')

    assert ('RawBrainVision' in repr(raw_py))

    assert_equal(raw_py.info['highpass'], 0.)
    assert_equal(raw_py.info['lowpass'], 250.)

    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, times_py = raw_py[picks]

    # compare with a file that was generated using MNE-C
    raw_bin = read_raw_fif(eeg_bin, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_bin, times_bin = raw_bin[picks]

    assert_array_almost_equal(data_py, data_bin)
    assert_array_almost_equal(times_py, times_bin)

    # Make sure EOG channels are marked correctly
    for ch in raw_py.info['chs']:
        if ch['ch_name'] in eog:
            assert_equal(ch['kind'], FIFF.FIFFV_EOG_CH)
        elif ch['ch_name'] == 'STI 014':
            assert_equal(ch['kind'], FIFF.FIFFV_STIM_CH)
        elif ch['ch_name'] in ('CP5', 'CP6'):
            assert_equal(ch['kind'], FIFF.FIFFV_MISC_CH)
            assert_equal(ch['unit'], FIFF.FIFF_UNIT_NONE)
        elif ch['ch_name'] == 'ReRef':
            assert_equal(ch['kind'], FIFF.FIFFV_MISC_CH)
            assert_equal(ch['unit'], FIFF.FIFF_UNIT_CEL)
        elif ch['ch_name'] in raw_py.info['ch_names']:
            assert_equal(ch['kind'], FIFF.FIFFV_EEG_CH)
            assert_equal(ch['unit'], FIFF.FIFF_UNIT_V)
        else:
            raise RuntimeError("Unknown Channel: %s" % ch['ch_name'])

    # test loading v2
    read_raw_brainvision(vhdr_v2_path, eog=eog, preload=True,
                         verbose='error')
    # For the nanovolt unit test we use the same data file with a different
    # header file.
    raw_nV = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_nV_path, montage=montage,
        eog=eog, misc='auto')
    assert_equal(raw_nV.info['chs'][0]['ch_name'], 'FP1')
    assert_equal(raw_nV.info['chs'][0]['kind'], FIFF.FIFFV_EEG_CH)
    data_nanovolt, _ = raw_nV[0]
    assert_array_almost_equal(data_py[0, :], data_nanovolt[0, :])


def test_brainvision_vectorized_data():
    """Test reading BrainVision data files with vectorized data."""
    with pytest.warns(RuntimeWarning, match='software filter'):
        raw = read_raw_brainvision(vhdr_old_path, preload=True)

    assert_array_equal(raw._data.shape, (29, 251))

    first_two_samples_all_chs = np.array([[+5.22000008e-06, +5.10000000e-06],
                                          [+2.10000000e-06, +2.27000008e-06],
                                          [+1.15000000e-06, +1.33000002e-06],
                                          [+4.00000000e-07, +4.00000000e-07],
                                          [-3.02999992e-06, -2.82000008e-06],
                                          [+2.71000004e-06, +2.45000000e-06],
                                          [+2.41000004e-06, +2.36000004e-06],
                                          [+1.01999998e-06, +1.18000002e-06],
                                          [-1.33999996e-06, -1.25000000e-06],
                                          [-2.60000000e-06, -2.46000004e-06],
                                          [+6.80000019e-07, +8.00000000e-07],
                                          [+1.48000002e-06, +1.48999996e-06],
                                          [+1.61000004e-06, +1.51000004e-06],
                                          [+7.19999981e-07, +8.60000038e-07],
                                          [-3.00000000e-07, -4.00000006e-08],
                                          [-1.20000005e-07, +6.00000024e-08],
                                          [+8.19999981e-07, +9.89999962e-07],
                                          [+1.13000002e-06, +1.28000002e-06],
                                          [+1.08000002e-06, +1.33999996e-06],
                                          [+2.20000005e-07, +5.69999981e-07],
                                          [-4.09999990e-07, +4.00000006e-08],
                                          [+5.19999981e-07, +9.39999962e-07],
                                          [+1.01000004e-06, +1.51999998e-06],
                                          [+1.01000004e-06, +1.55000000e-06],
                                          [-1.43000002e-06, -1.13999996e-06],
                                          [+3.65000000e-06, +3.65999985e-06],
                                          [+4.15999985e-06, +3.79000015e-06],
                                          [+9.26999969e-06, +8.95999985e-06],
                                          [-7.35999985e-06, -7.18000031e-06],
                                          ])

    assert_array_almost_equal(raw._data[:, :2], first_two_samples_all_chs)


def test_coodinates_extraction():
    """Test reading of [Coordinates] section if present."""
    # vhdr 2 has a Coordinates section
    with pytest.warns(RuntimeWarning, match='coordinate information'):
        raw = read_raw_brainvision(vhdr_v2_path)

    # Basic check of extracted coordinates
    assert raw.info['dig'] is not None
    diglist = raw.info['dig']
    coords = np.array([dig['r'] for dig in diglist])
    assert coords.shape[0] == len(raw.ch_names)
    assert coords.shape[1] == 3

    # Make sure the scaling seems right
    # a coordinate more than 20cm away from origin is implausible
    assert coords.max() < 0.2

    # vhdr 1 does not have a Coordinates section
    raw2 = read_raw_brainvision(vhdr_path)
    assert raw2.info['dig'] is None


@testing.requires_testing_data
def test_brainvision_neuroone_export():
    """Test Brainvision file exported with neuroone system."""
    raw = read_raw_brainvision(neuroone_vhdr, verbose='error')
    assert raw.info['meas_date'] is None
    assert len(raw.info['chs']) == 65
    assert raw.info['sfreq'] == 5000.


@testing.requires_testing_data
def test_read_vmrk_annotations():
    """Test load brainvision annotations."""
    sfreq = 1000.0

    # Test vmrk file without annotations
    # delete=False is for Windows compatibility
    with open(vmrk_path) as myfile:
        head = [next(myfile) for x in range(6)]
    with NamedTemporaryFile(mode='w+', suffix='.vmrk', delete=False) as temp:
        for item in head:
            temp.write(item)
        temp.seek(0)
        read_annotations(temp.name, sfreq=sfreq)
    try:
        temp.close()
        unlink(temp.name)
    except IOError:
        pass


@testing.requires_testing_data
def test_read_vhdr_annotations_and_events():
    """Test load brainvision annotations and parse them to events."""
    sfreq = 1000.0
    expected_orig_time = 1384359243.794231
    expected_onset_latency = np.array(
        [0, 486., 496., 1769., 1779., 3252., 3262., 4935., 4945., 5999., 6619.,
         6629., 7629., 7699.]
    )
    expected_annot_description = [
        'New Segment/', 'Stimulus/S253', 'Stimulus/S255', 'Event/254',
        'Stimulus/S255', 'Event/254', 'Stimulus/S255', 'Stimulus/S253',
        'Stimulus/S255', 'Response/R255', 'Event/254', 'Stimulus/S255',
        'SyncStatus/Sync On', 'Optic/O  1'
    ]
    expected_events = np.stack([
        expected_onset_latency,
        np.zeros_like(expected_onset_latency),
        [99999, 253, 255, 254, 255, 254, 255, 253, 255, 1255, 254, 255, 99998,
         2001],
    ]).astype('int64').T
    expected_event_id = {'New Segment/': 99999, 'Stimulus/S253': 253,
                         'Stimulus/S255': 255, 'Event/254': 254,
                         'Response/R255': 1255, 'SyncStatus/Sync On': 99998,
                         'Optic/O  1': 2001}

    raw = read_raw_brainvision(vhdr_path, eog=eog)

    # validate annotations
    assert raw.annotations.orig_time == expected_orig_time
    assert_allclose(raw.annotations.onset, expected_onset_latency / sfreq)
    assert_array_equal(raw.annotations.description, expected_annot_description)

    # validate event extraction
    events, event_id = events_from_annotations(raw)
    assert_array_equal(events, expected_events)
    assert event_id == expected_event_id

    # validate that None gives us a sorted list
    expected_none_event_id = {desc: idx + 1 for idx, desc in enumerate(sorted(
        event_id.keys()))}
    events, event_id = events_from_annotations(raw, event_id=None)
    assert event_id == expected_none_event_id

    # Add some custom ones, plus a 2-digit one
    s_10 = 'Stimulus/S 10'
    raw.annotations.append([1, 2, 3], 10, ['ZZZ', s_10, 'YYY'])
    expected_event_id.update(YYY=10001, ZZZ=10002)  # others starting at 10001
    expected_event_id[s_10] = 10
    _, event_id = events_from_annotations(raw)
    assert event_id == expected_event_id

    # Concatenating two shouldn't change the resulting event_id
    # (BAD and EDGE should be ignored)
    with pytest.warns(RuntimeWarning, match='expanding outside'):
        raw_concat = concatenate_raws([raw.copy(), raw.copy()])
    _, event_id = events_from_annotations(raw_concat)
    assert event_id == expected_event_id


@testing.requires_testing_data
def test_automatic_vmrk_sfreq_recovery():
    """Test proper sfreq inference by checking the onsets."""
    assert_array_equal(read_annotations(vmrk_path, sfreq='auto'),
                       read_annotations(vmrk_path, sfreq=1000.0))


@testing.requires_testing_data
def test_event_id_stability_when_save_and_fif_reload(tmpdir):
    """Test load events from brainvision annotations when read_raw_fif."""
    fname = op.join(str(tmpdir), 'bv-raw.fif')
    raw = read_raw_brainvision(vhdr_path, eog=eog)
    original_events, original_event_id = events_from_annotations(raw)

    raw.save(fname)
    raw = read_raw_fif(fname)
    events, event_id = events_from_annotations(raw)

    assert event_id == original_event_id
    assert_array_equal(events, original_events)

run_tests_if_main()
