# -*- coding: utf-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

import pytest

from numpy.testing import assert_allclose, assert_array_equal
from mne.annotations import events_from_annotations
from mne.datasets import testing
from mne.event import find_events
from mne.io.constants import FIFF
from mne.io.edf import read_raw_bdf
from mne.io.bti import read_raw_bti
from mne.io.curry import read_raw_curry
from mne.io.curry.curry import _check_missing_files, _read_events_curry
from mne.utils import check_version


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
    pytest.param(test_sfreq_0, 1e-7, id='test sfreq = 0'),
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
