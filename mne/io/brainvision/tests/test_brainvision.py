# -*- coding: utf-8 -*-
# Author: Teon Brooks <teon.brooks@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

from functools import partial
import inspect
import os.path as op
import shutil

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest

from mne.utils import _TempDir, run_tests_if_main
from mne import pick_types, find_events, read_annotations
from mne.io.constants import FIFF
from mne.io import read_raw_fif, read_raw_brainvision
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing

# Helper for deprecation
read_raw_brainvision_orig = read_raw_brainvision
read_raw_brainvision = partial(read_raw_brainvision_orig, stim_channel=False)
read_raw_brainvision_old = partial(read_raw_brainvision_orig,
                                   stim_channel=True)

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
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

montage = op.join(data_dir, 'test.hpts')
eeg_bin = op.join(data_dir, 'test_bin_raw.fif')
eog = ['HL', 'HR', 'Vb']
event_id = {'Sync On': 5}


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


def test_vhdr_codepage_ansi():
    """Test BV reading with ANSI codepage."""
    raw_init = read_raw_brainvision(vhdr_path, event_id=event_id)
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

    raw = read_raw_brainvision(ansi_vhdr_path, event_id=event_id)
    data_new, times_new = raw[:]

    assert_equal(raw_init.ch_names, raw.ch_names)
    assert_allclose(data_new, data_expected, atol=1e-15)
    assert_allclose(times_new, times_expected, atol=1e-15)


def test_ascii():
    """Test ASCII BV reading."""
    raw = read_raw_brainvision(vhdr_path, event_id=event_id)
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
    raw = read_raw_brainvision(ascii_vhdr_path, event_id=event_id)
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
            read_raw_brainvision_old, vhdr_fname=vhdr_mixed_highpass_path,
            montage=montage, eog=eog, event_id=event_id)

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
        montage=montage, eog=eog, event_id=event_id)

    assert_equal(raw.info['highpass'], 10.)
    assert_equal(raw.info['lowpass'], 250.)

    # Heterogeneous highpass in Hertz
    with pytest.warns(RuntimeWarning, match='different .*pass filters') as w:
        raw = _test_raw_reader(
            read_raw_brainvision_old, vhdr_fname=vhdr_mixed_highpass_hz_path,
            montage=montage, eog=eog, event_id=event_id)

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
        montage=montage, eog=eog, event_id=event_id)

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 250.)

    # Heterogeneous lowpass in Hertz (default measurement unit)
    with pytest.warns(RuntimeWarning) as w:  # event parsing
        raw = _test_raw_reader(
            read_raw_brainvision_old, vhdr_fname=vhdr_mixed_lowpass_path,
            montage=montage, eog=eog, event_id=event_id)

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
        montage=montage, eog=eog, event_id=event_id)

    assert_equal(raw.info['highpass'], 1. / (2 * np.pi * 10))
    assert_equal(raw.info['lowpass'], 1. / (2 * np.pi * 0.004))

    # Heterogeneous lowpass in seconds
    with pytest.warns(RuntimeWarning) as w:  # filter settings
        raw = _test_raw_reader(
            read_raw_brainvision_old, vhdr_fname=vhdr_mixed_lowpass_s_path,
            montage=montage, eog=eog, event_id=event_id)

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
        eog=eog, misc='auto', event_id=event_id)

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
                         event_id=event_id,
                         trig_shift_by_type={'response': 1000},
                         verbose='error')
    # For the nanovolt unit test we use the same data file with a different
    # header file.
    raw_nV = _test_raw_reader(
        read_raw_brainvision, vhdr_fname=vhdr_nV_path, montage=montage,
        eog=eog, misc='auto', event_id=event_id)
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


def test_events():
    """Test reading and modifying events."""
    tempdir = _TempDir()
    # Note: BrainVision event offsets are 1-based, mne offsets are 0-based.
    # So in all tests below, the "onset" is 1 less than what's in the file

    # check that events are read and stim channel is synthesized correctly
    raw = read_raw_brainvision_old(vhdr_path, eog=eog, event_id=event_id)
    events = raw._get_brainvision_events()
    events = events[events[:, 2] != event_id['Sync On']]
    assert_array_equal(events, [[486, 0, 253],
                                [496, 0, 255],
                                [1769, 0, 254],
                                [1779, 0, 255],
                                [3252, 0, 254],
                                [3262, 0, 255],
                                [4935, 0, 253],
                                [4945, 0, 255],
                                [5999, 0, 255],
                                [6619, 0, 254],
                                [6629, 0, 255],
                                [7699, 0, 1]])

    # check that events are read and stim channel is synthesized correctly and
    # response triggers are shifted using the deprecated response_trig_shift.
    with pytest.deprecated_call():
        raw = read_raw_brainvision_old(vhdr_path, eog=eog,
                                       response_trig_shift=1000,
                                       event_id=event_id)
    events = raw._get_brainvision_events()
    assert_array_equal(events, [[486, 0, 253],
                                [496, 0, 255],
                                [1769, 0, 254],
                                [1779, 0, 255],
                                [3252, 0, 254],
                                [3262, 0, 255],
                                [4935, 0, 253],
                                [4945, 0, 255],
                                [5999, 0, 1255],
                                [6619, 0, 254],
                                [6629, 0, 255],
                                [7629, 0, 5],
                                [7699, 0, 1]])

    # check that trig_shift_by_type works as well
    raw = read_raw_brainvision_old(vhdr_path, eog=eog,
                                   trig_shift_by_type={'response': 1000,
                                                       'Optic': 2000},
                                   event_id=event_id)
    events = raw._get_brainvision_events()
    assert_array_equal(events, [[486, 0, 253],
                                [496, 0, 255],
                                [1769, 0, 254],
                                [1779, 0, 255],
                                [3252, 0, 254],
                                [3262, 0, 255],
                                [4935, 0, 253],
                                [4945, 0, 255],
                                [5999, 0, 1255],
                                [6619, 0, 254],
                                [6629, 0, 255],
                                [7629, 0, 5],
                                [7699, 0, 2001]])

    # Check that we warn if a trigger is dropped
    with pytest.warns(RuntimeWarning, match='will be dropped'):
        raw = read_raw_brainvision_old(vhdr_path)
    # check that events are read and stim channel is synthesized correctly and
    # response triggers are ignored.
    raw = read_raw_brainvision_old(vhdr_path, eog=eog, event_id=event_id,
                                   trig_shift_by_type={'response': None})
    events = raw._get_brainvision_events()
    events = events[events[:, 2] != event_id['Sync On']]
    assert_array_equal(events, [[486, 0, 253],
                                [496, 0, 255],
                                [1769, 0, 254],
                                [1779, 0, 255],
                                [3252, 0, 254],
                                [3262, 0, 255],
                                [4935, 0, 253],
                                [4945, 0, 255],
                                [6619, 0, 254],
                                [6629, 0, 255],
                                [7699, 0, 1]])

    # Error handling of trig_shift_by_type
    pytest.raises(TypeError, read_raw_brainvision_old, vhdr_path, eog=eog,
                  preload=True, trig_shift_by_type=1)
    pytest.raises(TypeError, read_raw_brainvision_old, vhdr_path, eog=eog,
                  preload=True, trig_shift_by_type={'response': 0.1})
    pytest.raises(TypeError, read_raw_brainvision_old, vhdr_path, eog=eog,
                  preload=True, trig_shift_by_type={'response': np.nan})
    pytest.raises(ValueError, read_raw_brainvision_old, vhdr_path, eog=eog,
                  preload=True, trig_shift_by_type={'response': 1000,
                                                    'Response': 1001})
    with pytest.warns(DeprecationWarning):
        pytest.raises(ValueError, read_raw_brainvision_old, vhdr_path, eog=eog,
                      preload=True, trig_shift_by_type={'response': 1000},
                      response_trig_shift=1001)

    # Check that events of type "Comment" are read if they contain square
    # brackets (which usually signify a new section within a BrainVision file)
    # If no event_id specified, skip the marker and continue as planned
    with pytest.warns(RuntimeWarning, match='channel types to misc'):
        raw = read_raw_brainvision_old(vhdr_v2_path)
    events = raw._get_brainvision_events()
    assert events.shape == (11, 3)  # shape of events without comment

    # with event_id specified, get that comment and assert it's there
    tmp_event_id = {'comment using [square] brackets': 999}
    with pytest.warns(RuntimeWarning, match='channel types to misc'):
        raw = read_raw_brainvision_old(vhdr_v2_path, event_id=tmp_event_id)
    events = raw._get_brainvision_events()
    assert 999 in events[:, -1]
    assert events.shape == (12, 3)  # shape of events with comment

    # check that events are read properly when event_id is specified for
    # auxiliary events
    raw = read_raw_brainvision_old(vhdr_path, eog=eog, preload=True,
                                   trig_shift_by_type={'response': None},
                                   event_id=event_id)
    events = raw._get_brainvision_events()

    expected_events = np.array([[486, 0, 253],
                                [496, 0, 255],
                                [1769, 0, 254],
                                [1779, 0, 255],
                                [3252, 0, 254],
                                [3262, 0, 255],
                                [4935, 0, 253],
                                [4945, 0, 255],
                                [6619, 0, 254],
                                [6629, 0, 255],
                                [7629, 0, 5],
                                [7699, 0, 1]])

    assert_array_equal(events, expected_events)

    # Test that both trig_shift_by_type and event_id can be set
    read_raw_brainvision(vhdr_path, eog=eog, preload=False,
                         trig_shift_by_type={'response': 100},
                         event_id=event_id)
    mne_events = find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # modify events and check that stim channel is updated
    index = events[:, 2] == 255
    events = events[index]
    raw._set_brainvision_events(events)
    mne_events = find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # remove events
    nchan = raw.info['nchan']
    ch_name = raw.info['chs'][-2]['ch_name']
    events = np.empty((0, 3))
    raw._set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-2]['ch_name'], ch_name)
    assert_equal(len(find_events(raw, 'STI 014')), 0)
    assert_allclose(raw[-1][0], 0.)
    fname = op.join(tempdir, 'evt_raw.fif')
    raw.save(fname)

    # add events back in
    events = [[10, 1, 2]]
    raw._set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-1]['ch_name'], 'STI 014')

    # degenerate
    with pytest.raises(TypeError, match='stim_channel must be an instance of'):
        read_raw_brainvision(vhdr_path, stim_channel='foo')


def test_brainvision_with_montage():
    """Test reading embedded montage information."""
    with pytest.warns(RuntimeWarning, match='Omitted 3 annot'):
        raw = read_raw_brainvision(vhdr_v2_path, eog=eog, misc=['ReRef'])
    for i, d in enumerate(raw.info['dig'], 1):
        assert_equal(d['coord_frame'], FIFF.FIFFV_COORD_HEAD)
        assert_equal(d['ident'], i)
        assert_equal(d['kind'], FIFF.FIFFV_POINT_EEG)
        assert_equal(len(d['r']), 3)

    raw_none = read_raw_brainvision(vhdr_v2_path, verbose='error')
    for r, n in zip(raw.info['chs'], raw_none.info['chs']):
        if r['kind'] != n['kind']:
            assert_array_equal(r['loc'], n['loc'])


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
    annotations = read_annotations(vmrk_path, sfreq=sfreq)
    assert annotations.orig_time == 1384359243.794231
    expected = np.array([0, 486., 496., 1769., 1779., 3252., 3262., 4935.,
                         4945., 5999., 6619., 6629., 7629., 7699.]) / sfreq
    description = ['New Segment/',
                   'Stimulus/S253', 'Stimulus/S255', 'Stimulus/S254',
                   'Stimulus/S255', 'Stimulus/S254', 'Stimulus/S255',
                   'Stimulus/S253', 'Stimulus/S255', 'Response/R255',
                   'Stimulus/S254', 'Stimulus/S255',
                   'SyncStatus/Sync On', 'Optic/O  1']
    assert_array_almost_equal(annotations.onset,
                              expected, decimal=7)
    assert_array_equal(annotations.description, description)

    # Test automatic detection of sfreq from header file
    annotations_auto = read_annotations(vmrk_path)
    assert_array_equal(annotations.onset, annotations_auto.onset)


def test_read_raw_brainvision_warn_on_marker_drop():
    """Test that loading test.vhdr warns a summary of each dropped type."""
    EXPECTED_WARN_MSG = (r"1 event\(s\) will be dropped, such as "
                         r"\[\'SyncStatus\/Sync On\'\]. Consider using the "
                         r"event\_id parameter to parse events that do not "
                         r"follow the BrainVision format. For more "
                         r"information, see the docstring of "
                         r"read_raw_brainvision.")
    with pytest.warns(RuntimeWarning, match=EXPECTED_WARN_MSG) as recwarn:
        read_raw_brainvision_old(vhdr_path, verbose=True)
    assert len(recwarn) == 1


run_tests_if_main()
