# -*- coding: utf-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from shutil import copyfile

import pytest

from numpy.testing import assert_allclose, assert_array_equal
from mne.annotations import events_from_annotations
from mne.datasets import testing
from mne.event import find_events
from mne.io.constants import FIFF
from mne.io.edf import read_raw_bdf
from mne.io.bti import read_raw_bti
from mne.io.curry import read_raw_curry
from mne.utils import check_version, run_tests_if_main
from mne.annotations import read_annotations
from mne.io.curry.curry import (
    _check_missing_files, _read_events_curry, _get_curry_version,
    INFO_FILE_EXTENSION, EVENT_FILE_EXTENSION
)


data_dir = testing.data_path(download=False)
curry_dir = op.join(data_dir, "curry")

bdf_file = op.join(data_dir, 'BDF', 'test_bdf_stim_channel.bdf')

bti_rfDC_file = op.join(data_dir, 'BTi', 'erm_HFH', 'c,rfDC')

curry7_rfDC_file = op.join(curry_dir, "c,rfDC Curry 7.dat")
curry8_rfDC_file = op.join(curry_dir, "c,rfDC Curry 8.cdt")

curry7_bdf_file = op.join(curry_dir, "test_bdf_stim_channel Curry 7.dat")
curry7_bdf_ascii_file = op.join(curry_dir,
                                "test_bdf_stim_channel Curry 7 ASCII.dat")

curry8_bdf_file = op.join(curry_dir, "test_bdf_stim_channel Curry 8.cdt")
curry8_bdf_ascii_file = op.join(curry_dir,
                                "test_bdf_stim_channel Curry 8 ASCII.cdt")

test_sfreq_0 = op.join(curry_dir, "test_sfreq_0.dat")

if not check_version("numpy", "1.16.0"):
    do_warn = 'ignore:.*take longer for ASCII.*:'
else:
    do_warn = ''


@pytest.fixture(scope='session')
def bdf_curry_ref():
    """Return a view of the reference bdf used to create the curry files."""
    raw = read_raw_bdf(bdf_file, preload=True).drop_channels(['Status'])
    return raw


@pytest.mark.filterwarnings(do_warn)
@testing.requires_testing_data
@pytest.mark.parametrize('fname,tol', [
    pytest.param(curry7_bdf_file, 1e-7, id='curry 7'),
    pytest.param(curry8_bdf_file, 1e-7, id='curry 8'),
    pytest.param(curry7_bdf_ascii_file, 1e-4, id='curry 7 ascii'),
    pytest.param(curry8_bdf_ascii_file, 1e-4, id='curry 8 ascii'),
])
@pytest.mark.parametrize('preload', [True, False])
def test_read_raw_curry(fname, tol, preload, bdf_curry_ref):
    """Test reading CURRY files."""
    raw = read_raw_curry(fname, preload=preload)

    if preload:
        isinstance(getattr(raw, '_data'), np.ndarray)
    else:
        with pytest.raises(AttributeError, match="no attribute '_data'"):
            getattr(raw, '_data')

    assert raw.n_times == bdf_curry_ref.n_times
    assert raw.info['sfreq'] == bdf_curry_ref.info['sfreq']

    for field in ['kind', 'ch_name']:
        assert_array_equal([ch[field] for ch in raw.info['chs']],
                           [ch[field] for ch in bdf_curry_ref.info['chs']])

    assert_allclose(raw.get_data(), bdf_curry_ref.get_data(), atol=tol)

    picks, start, stop = ["C3", "C4"], 200, 800
    assert_allclose(
        raw.get_data(picks=picks, start=start, stop=stop),
        bdf_curry_ref.get_data(picks=picks, start=start, stop=stop),
        rtol=tol)


@pytest.mark.filterwarnings(do_warn)
@testing.requires_testing_data
@pytest.mark.parametrize('fname,tol', [
    pytest.param(curry7_rfDC_file, 1e-6, id='curry 7'),
    pytest.param(curry8_rfDC_file, 1e-3, id='curry 8'),
])
def test_read_raw_curry_rfDC(fname, tol):
    """Test reading CURRY files."""
    # check data
    bti_rfDC = read_raw_bti(pdf_fname=bti_rfDC_file, head_shape_fname=None)
    raw = read_raw_curry(fname)

    # test on the eeg chans, since these were not renamed by curry
    eeg_names = [ch["ch_name"] for ch in raw.info["chs"]
                 if ch["kind"] == FIFF.FIFFV_EEG_CH]

    assert_allclose(raw.get_data(eeg_names),
                    bti_rfDC.get_data(eeg_names), rtol=tol)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', [
    pytest.param(curry7_bdf_file, id='curry 7'),
    pytest.param(curry8_bdf_file, id='curry 8'),
])
def test_read_events_curry_are_same_as_bdf(fname):
    """Test events from curry annotations recovers the right events."""
    EVENT_ID = {str(ii): ii for ii in range(5)}
    REF_EVENTS = find_events(read_raw_bdf(bdf_file, preload=True))

    raw = read_raw_curry(fname)
    events, _ = events_from_annotations(raw, event_id=EVENT_ID)
    assert_allclose(events, REF_EVENTS)


def test_check_missing_files():
    """Test checking for missing curry files (smoke test)."""
    invalid_fname = "/invalid/path/name"

    with pytest.raises(IOError, match="file type .*? must end with"):
        _read_events_curry(invalid_fname)

    with pytest.raises(FileNotFoundError, match="files cannot be found"):
        _check_missing_files(invalid_fname, 7)
        _check_missing_files(invalid_fname, 8)


def _mock_info_file(src, dst, sfreq, time_step):
    with open(src, 'r') as in_file, open(dst, 'w') as out_file:
        for line in in_file:
            if 'SampleFreqHz' in line:
                out_file.write(line.replace('500', str(sfreq)))
            elif 'SampleTimeUsec' in line:
                out_file.write(line.replace('2000', str(time_step)))
            else:
                out_file.write(line)


@pytest.fixture(params=[
    pytest.param(dict(sfreq=500, time_step=0), id='correct sfreq'),
    pytest.param(dict(sfreq=0, time_step=2000), id='correct time_step'),
    pytest.param(dict(sfreq=500, time_step=2000), id='both correct'),
    pytest.param(dict(sfreq=0, time_step=0), id='both 0',
                 marks=pytest.mark.raises),
    pytest.param(dict(sfreq=500, time_step=42), id='mismatch',
                 marks=pytest.mark.raises),
])
def sfreq_testing_data(tmpdir, request):
    """Generate different sfreq, time_step scenarios to be tested."""
    sfreq, time_step = request.param['sfreq'], request.param['time_step']

    in_base_name = curry7_bdf_file.strip('dat')
    out_base_name = str(tmpdir.join('curry.'))

    # create dummy empty files for 'dat' and 'rs3'
    for fname in [out_base_name + ext for ext in ['dat', 'rs3']]:
        open(fname, 'a').close()

    _mock_info_file(src=in_base_name + 'dap', dst=out_base_name + 'dap',
                    sfreq=sfreq, time_step=time_step)

    return out_base_name + 'dat'


@testing.requires_testing_data
def test_sfreq(sfreq_testing_data):
    """Test sfreq and time_step."""
    raw = read_raw_curry(sfreq_testing_data, preload=False)
    assert raw.info['sfreq'] == 500


@testing.requires_testing_data
@pytest.mark.parametrize('fname', [
    pytest.param(curry_dir + '/test_bdf_stim_channel Curry 7.cef', id='7'),
    pytest.param(curry_dir + '/test_bdf_stim_channel Curry 8.cdt.cef', id='8'),
    pytest.param(curry_dir + '/test_bdf_stim_channel Curry 7 ASCII.cef',
                 id='7 ascii'),
    pytest.param(curry_dir + '/test_bdf_stim_channel Curry 8 ASCII.cdt.cef',
                 id='8 ascii'),
])
def test_read_curry_annotations(fname):
    """Test reading for Curry events file."""
    EXPECTED_ONSET = [0.484, 0.486, 0.62, 0.622, 1.904, 1.906, 3.212, 3.214,
                      4.498, 4.5, 5.8, 5.802, 7.074, 7.076, 8.324, 8.326, 9.58,
                      9.582]
    EXPECTED_DURATION = np.zeros_like(EXPECTED_ONSET)
    EXPECTED_DESCRIPTION = ['4', '50000', '2', '50000', '1', '50000', '1',
                            '50000', '1', '50000', '1', '50000', '1', '50000',
                            '1', '50000', '1', '50000']

    annot = read_annotations(fname, sfreq='auto')

    assert annot.orig_time is None
    assert_array_equal(annot.onset, EXPECTED_ONSET)
    assert_array_equal(annot.duration, EXPECTED_DURATION)
    assert_array_equal(annot.description, EXPECTED_DESCRIPTION)


@pytest.fixture(params=[
    pytest.param('7.cef', id='time_stamp 7'),
    pytest.param('8.cdt.cef', id='time_stamp 8'),
    pytest.param('7 ASCII.cef', id='time_stamp 7 (ascii)'),
    pytest.param('8 ASCII.cdt.cef', id='time_stamp 8 (ascii)'),
])
def time_step_annot_data(tmpdir, request):
    """Generate info files with time_stamp to test read_annotations."""
    in_fname = (curry_dir + '/test_bdf_stim_channel ' + request.param)
    in_base_name, ext = in_fname.split(".", maxsplit=1)
    curry_vers = _get_curry_version(ext)
    out_base_name = str(tmpdir.join('curry.'))
    out_fname = out_base_name + EVENT_FILE_EXTENSION[curry_vers]

    copyfile(src=in_fname, dst=out_fname)
    _mock_info_file(src=in_base_name + INFO_FILE_EXTENSION[curry_vers],
                    dst=out_base_name + INFO_FILE_EXTENSION[curry_vers],
                    sfreq=0, time_step=2000)

    return out_base_name


def test_read_curry_annotations_using_mocked_info(time_step_annot_data):
    """Test reading for Curry events file."""
    EXPECTED_ONSET = [0.484, 0.486, 0.62, 0.622, 1.904, 1.906, 3.212, 3.214,
                      4.498, 4.5, 5.8, 5.802, 7.074, 7.076, 8.324, 8.326, 9.58,
                      9.582]
    EXPECTED_DURATION = np.zeros_like(EXPECTED_ONSET)
    EXPECTED_DESCRIPTION = ['4', '50000', '2', '50000', '1', '50000', '1',
                            '50000', '1', '50000', '1', '50000', '1', '50000',
                            '1', '50000', '1', '50000']

    annot = read_annotations(time_step_annot_data, sfreq='auto')

    assert annot.orig_time is None
    assert_array_equal(annot.onset, EXPECTED_ONSET)
    assert_array_equal(annot.duration, EXPECTED_DURATION)
    assert_array_equal(annot.description, EXPECTED_DESCRIPTION)


@pytest.fixture(params=[
    pytest.param('7.cef', id='time_stamp 7'),
    pytest.param('8.cdt.cef', id='time_stamp 8'),
    # maybe we don't need to test the entire matrix
    # (you should remove these comment)
])
def no_info_annot_data(tmpdir, request):
    """Generate info files with time_stamp to test read_annotations."""
    in_fname = (curry_dir + '/test_bdf_stim_channel ' + request.param)
    in_base_name, ext = in_fname.split(".", maxsplit=1)
    curry_vers = _get_curry_version(ext)
    out_base_name = str(tmpdir.join('curry.'))
    out_fname = out_base_name + EVENT_FILE_EXTENSION[curry_vers]

    copyfile(src=in_fname, dst=out_fname)

    return out_base_name


def test_read_curry_annotations_without_info(no_info_annot_data):
    """Test reading for Curry events file."""
    _msg = 'meaningful message stating that annotations cannot infer sfreq foo.xx not found'  # noqa
    with pytest.raises(RuntimeError, match=_msg):
        read_annotations(time_step_annot_data, sfreq='auto')

run_tests_if_main()
