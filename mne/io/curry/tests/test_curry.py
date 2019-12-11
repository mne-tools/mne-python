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
from mne.io.curry.curry import (_get_curry_version, _get_curry_file_structure,
                                _read_events_curry, FILE_EXTENSIONS)


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

missing_event_file = op.join(curry_dir, "test_sfreq_0.dat")


@pytest.fixture(scope='session')
def bdf_curry_ref():
    """Return a view of the reference bdf used to create the curry files."""
    raw = read_raw_bdf(bdf_file, preload=True).drop_channels(['Status'])
    return raw


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
    with pytest.warns(None) as wrn:
        raw = read_raw_curry(fname, preload=preload)

    if not check_version('numpy', '1.16') and preload and fname.endswith(
            'ASCII.dat'):
        assert len(wrn) > 0
    else:
        assert len(wrn) == 0

    assert hasattr(raw, '_data') == preload
    assert raw.n_times == bdf_curry_ref.n_times
    assert raw.info['sfreq'] == bdf_curry_ref.info['sfreq']

    for field in ['kind', 'ch_name']:
        assert_array_equal([ch[field] for ch in raw.info['chs']],
                           [ch[field] for ch in bdf_curry_ref.info['chs']])

    raw.verbose = 'error'  # don't emit warnings about slow reading
    assert_allclose(raw.get_data(), bdf_curry_ref.get_data(), atol=tol)

    picks, start, stop = ["C3", "C4"], 200, 800
    assert_allclose(
        raw.get_data(picks=picks, start=start, stop=stop),
        bdf_curry_ref.get_data(picks=picks, start=start, stop=stop),
        rtol=tol)


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
    invalid_fname = "/invalid/path/name.xy"

    with pytest.raises(IOError, match="file type .*? must end with"):
        _read_events_curry(invalid_fname)

    with pytest.raises(FileNotFoundError, match="files cannot be found"):
        _get_curry_file_structure(invalid_fname)
        _get_curry_file_structure(missing_event_file,
                                  required=["info", "events"])


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
                 marks=pytest.mark.xfail(raises=ValueError)),
    pytest.param(dict(sfreq=500, time_step=42), id='mismatch',
                 marks=pytest.mark.xfail(raises=ValueError)),
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


def _get_read_annotations_mock_info(name_part, mock_dir):
    original, modified = dict(), dict()

    original['event'] = curry_dir + '/test_bdf_stim_channel ' + name_part
    original['base'], ext = original['event'].split(".", maxsplit=1)
    version = _get_curry_version(ext)
    original['info'] = original['base'] + FILE_EXTENSIONS[version]["info"]

    modified['base'] = str(mock_dir.join('curry'))
    modified['event'] = modified['base'] + FILE_EXTENSIONS[version]["events"]
    modified['info'] = modified['base'] + FILE_EXTENSIONS[version]["info"]

    return original, modified


@testing.requires_testing_data
@pytest.mark.parametrize('name_part', [
    pytest.param('7.cef', id='7'),
    pytest.param('8.cdt.cef', id='8'),
    pytest.param('7 ASCII.cef', id='7 (ascii)'),
    pytest.param('8 ASCII.cdt.cef', id='8 (ascii)'),
])
def test_read_curry_annotations_using_mocked_info(tmpdir, name_part):
    """Test reading for Curry events file."""
    EXPECTED_ONSET = [0.484, 0.486, 0.62, 0.622, 1.904, 1.906, 3.212, 3.214,
                      4.498, 4.5, 5.8, 5.802, 7.074, 7.076, 8.324, 8.326, 9.58,
                      9.582]
    EXPECTED_DURATION = np.zeros_like(EXPECTED_ONSET)
    EXPECTED_DESCRIPTION = ['4', '50000', '2', '50000', '1', '50000', '1',
                            '50000', '1', '50000', '1', '50000', '1', '50000',
                            '1', '50000', '1', '50000']

    original, fname = _get_read_annotations_mock_info("Curry " + name_part,
                                                      tmpdir)
    copyfile(src=original['event'], dst=fname['event'])

    _msg = 'required files cannot be found'
    with pytest.raises(FileNotFoundError, match=_msg):
        read_annotations(fname['event'], sfreq='auto')

    _mock_info_file(src=original['info'], dst=fname['info'],
                    sfreq=0, time_step=2000)

    annot = read_annotations(fname['event'], sfreq='auto')

    assert annot.orig_time is None
    assert_array_equal(annot.onset, EXPECTED_ONSET)
    assert_array_equal(annot.duration, EXPECTED_DURATION)
    assert_array_equal(annot.description, EXPECTED_DESCRIPTION)


run_tests_if_main()
