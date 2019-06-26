# -*- coding: utf-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD (3-clause)

import os
import os.path as op
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

bdf_file = op.join(data_dir, 'BDF/test_bdf_stim_channel.bdf')

bti_rfDC_file = op.join(data_dir, 'BTi/erm_HFH/c,rfDC')

curry7_rfDC_file = op.join(curry_dir, "c,rfDC Curry 7.dat")
curry8_rfDC_file = op.join(curry_dir, "c,rfDC Curry 8.cdt")

curry7_bdf_file = op.join(curry_dir, "test_bdf_stim_channel Curry 7.dat")
curry7_bdf_ascii_file = op.join(curry_dir,
                                "test_bdf_stim_channel Curry 7 ASCII.dat")

curry8_bdf_file = op.join(curry_dir, "test_bdf_stim_channel Curry 8.cdt")
curry8_bdf_ascii_file = op.join(curry_dir,
                                "test_bdf_stim_channel Curry 8 ASCII.cdt")

event_file = op.join(curry_dir, "test_bdf_stim_channel Curry 7.cef")

if not check_version("numpy", "1.16.0"):
    do_warn = 'ignore:.*take longer for ASCII.*:'
else:
    do_warn = ''


@pytest.mark.filterwarnings(do_warn)
@testing.requires_testing_data
def test_read_raw_curry():
    """Test reading CURRY files."""
    # make sure all types of files can be passed to read_raw_curry
    for filename in os.listdir(curry_dir):
        if "test_bdf_stim_channel Curry" in filename:
            read_raw_curry(op.join(curry_dir, filename))

    # check no preloading for preload=False
    with pytest.raises(AttributeError, match="no attribute '_data'"):
        read_raw_curry(curry7_bdf_file, preload=False)._data
        read_raw_curry(curry7_bdf_ascii_file, preload=False)._data

    # check data
    bti_rfDC = read_raw_bti(pdf_fname=bti_rfDC_file, head_shape_fname=None)
    curry7_rfDC = read_raw_curry(curry7_rfDC_file)
    curry8_rfDC = read_raw_curry(curry8_rfDC_file)

    # test on the eeg chans, since these were not renamed by curry
    eeg_names = [ch["ch_name"] for ch in curry7_rfDC.info["chs"]
                 if ch["kind"] == FIFF.FIFFV_EEG_CH]

    assert_allclose(curry7_rfDC.get_data(eeg_names),
                    bti_rfDC.get_data(eeg_names), rtol=1e-6)
    assert_allclose(curry8_rfDC.get_data(eeg_names),
                    bti_rfDC.get_data(eeg_names), rtol=1e-3)

    bdf = read_raw_bdf(bdf_file)
    curry7_bdf = read_raw_curry(curry7_bdf_file, preload=True)
    curry8_bdf = read_raw_curry(curry8_bdf_file, preload=False)
    curry7_bdf_ascii = read_raw_curry(curry7_bdf_ascii_file,
                                      preload=False)
    curry8_bdf_ascii = read_raw_curry(curry7_bdf_ascii_file,
                                      preload=True)

    assert_allclose([curry7_bdf.n_times, curry7_bdf_ascii.n_times,
                     curry8_bdf.n_times, curry8_bdf_ascii.n_times],
                    bdf.n_times)
    assert_allclose([curry7_bdf.info["sfreq"], curry7_bdf_ascii.info["sfreq"],
                     curry8_bdf.info["sfreq"], curry8_bdf_ascii.info["sfreq"]],
                    bdf.info["sfreq"])

    # test all types of data
    picks = ["C3", "C4"]
    start, stop = 200, 800
    for curry_file in [curry7_bdf, curry7_bdf_ascii,
                       curry8_bdf, curry8_bdf_ascii]:
        # use chans [:3] here since curry stim chan was already extracted
        assert_allclose([ch["kind"] for ch in curry_file.info["chs"]],
                        [ch["kind"] for ch in bdf.info["chs"][:3]])
        assert_array_equal([ch["ch_name"] for ch in curry_file.info["chs"]],
                           [ch["ch_name"] for ch in bdf.info["chs"][:3]])
        assert_allclose(curry_file.get_data(), bdf.get_data()[:3], atol=1e-6)
        # can't use bdf.get_data(picks) here, since it seems bugged
        assert_allclose(curry_file.get_data(picks, start, stop),
                        bdf.get_data(start=start, stop=stop)[:2], atol=1e-6)


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
    """Test checking for missing curry files."""
    invalid_fname = "/invalid/path/name"

    with pytest.raises(IOError, match="file type .*? must end with"):
        _read_events_curry(invalid_fname)

    with pytest.raises(FileNotFoundError, match="files cannot be found"):
        _check_missing_files(invalid_fname, 7)
        _check_missing_files(invalid_fname, 8)
