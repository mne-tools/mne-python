# Author: Mathieu Scheltienne <mathieu.scheltienne@fcbg.ch>
#
# License: BSD-3-Clause

import datetime
import itertools
from pathlib import Path
import re

import numpy as np
import pytest

from mne import create_info
from mne.annotations import Annotations
from mne.datasets import testing
from mne.io import RawArray, read_raw_fif
from mne.preprocessing import annotate_amplitude


date = datetime.datetime(2021, 12, 10, 7, 52, 24, 405305,
                         tzinfo=datetime.timezone.utc)
data_path = Path(testing.data_path(download=False))
skip_fname = data_path / 'misc' / 'intervalrecording_raw.fif'


@pytest.mark.parametrize('meas_date', (None, date))
@pytest.mark.parametrize('first_samp', (0, 10000))
def test_annotate_amplitude(meas_date, first_samp):
    """Test automatic annotation for segments based on peak-to-peak value."""
    n_ch, n_times = 11, 1000
    data = np.random.RandomState(0).randn(n_ch, n_times)
    assert not (np.diff(data, axis=-1) == 0).any()  # nothing flat at first
    info = create_info(n_ch, 1000., 'eeg')
    # from annotate_flat: test first_samp != for gh-6295
    raw = RawArray(data, info, first_samp=first_samp)
    raw.info['bads'] = [raw.ch_names[-1]]
    raw.set_meas_date(meas_date)

    # -- test bad channels spatial marking --
    for perc, dur in itertools.product((5, 99.9, 100.), (0.005, 0.95, 0.99)):
        kwargs = dict(bad_percent=perc, min_duration=dur)

        # test entire channel flat
        raw_ = raw.copy()
        raw_._data[0] = 0.
        annots, bads = annotate_amplitude(raw_, peak=None, flat=0., **kwargs)
        assert len(annots) == 0
        assert bads == ['0']

        # test multiple channels flat
        raw_ = raw.copy()
        raw_._data[0] = 0.
        raw_._data[2] = 0.
        annots, bads = annotate_amplitude(raw_, peak=None, flat=0., **kwargs)
        assert len(annots) == 0
        assert bads == ['0', '2']

        # test entire channel drifting
        raw_ = raw.copy()
        raw_._data[0] = np.arange(0, raw.times.size * 10, 10)
        annots, bads = annotate_amplitude(raw_, peak=5, flat=None, **kwargs)
        assert len(annots) == 0
        assert bads == ['0']

        # test multiple channels drifting
        raw_ = raw.copy()
        raw_._data[0] = np.arange(0, raw.times.size * 10, 10)
        raw_._data[2] = np.arange(0, raw.times.size * 10, 10)
        annots, bads = annotate_amplitude(raw_, peak=5, flat=None, **kwargs)
        assert len(annots) == 0
        assert bads == ['0', '2']

    # -- test bad channels temporal marking --
    # flat channel for the 20% last points
    n_good_times = int(round(0.8 * n_times))
    raw_ = raw.copy()
    raw_._data[0, n_good_times:] = 0.
    for perc in (5, 20):
        annots, bads = annotate_amplitude(raw_, peak=None, flat=0.,
                                          bad_percent=perc)
        assert len(annots) == 0
        assert bads == ['0']
    annots, bads = annotate_amplitude(raw_, peak=None, flat=0.,
                                      bad_percent=20.1)
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    assert annots[0]['description'] == 'BAD_flat'
    _check_annotation(raw_, annots[0], meas_date, first_samp, n_good_times, -1)

    # test multiple channels flat and multiple channels drift
    raw_ = raw.copy()
    raw_._data[0, 800:] = 0.
    raw_._data[1, 850:950] = 0.
    raw_._data[2, :200] = np.arange(0, 200 * 10, 10)
    raw_._data[2, 200:] += raw_._data[2, 199]  # add offset for next samples
    raw_._data[3, 50:150] = np.arange(0, 100 * 10, 10)
    raw_._data[3, 150:] += raw_._data[3, 149]  # add offset for next samples
    for perc in (5, 10):
        annots, bads = annotate_amplitude(raw_, peak=5, flat=0.,
                                          bad_percent=perc)
        assert len(annots) == 0
        assert bads == ['0', '1', '2', '3']
    for perc in (10.1, 20):
        annots, bads = annotate_amplitude(raw_, peak=5, flat=0.,
                                          bad_percent=perc)
        assert len(annots) == 2
        assert bads == ['0', '2']
        # check annotation instance
        assert all(annot['description'] in ('BAD_flat', 'BAD_peak')
                   for annot in annots)
        for annot in annots:
            start_idx = 50 if annot['description'] == 'BAD_peak' else 850
            stop_idx = 149 if annot['description'] == 'BAD_peak' else 949
            _check_annotation(raw_, annot, meas_date, first_samp, start_idx,
                              stop_idx)
    annots, bads = annotate_amplitude(raw_, peak=5, flat=0., bad_percent=20.1)
    assert len(annots) == 2
    assert len(bads) == 0
    # check annotation instance
    assert all(annot['description'] in ('BAD_flat', 'BAD_peak')
               for annot in annots)
    for annot in annots:
        start_idx = 0 if annot['description'] == 'BAD_peak' else 800
        stop_idx = 199 if annot['description'] == 'BAD_peak' else -1
        _check_annotation(raw_, annot, meas_date, first_samp, start_idx,
                          stop_idx)

    # test flat on already marked bad channel
    raw_ = raw.copy()
    raw_._data[-1, :] = 0.  # this channel is already in info['bads']
    annots, bads = annotate_amplitude(raw_, peak=None, flat=0., bad_percent=5)
    assert len(annots) == 0
    assert len(bads) == 0

    # test drift on already marked bad channel
    raw_ = raw.copy()
    raw_._data[-1, :] = np.arange(0, raw.times.size * 10, 10)
    annots, bads = annotate_amplitude(raw_, peak=5, flat=None, bad_percent=5)
    assert len(annots) == 0
    assert len(bads) == 0


@pytest.mark.parametrize('meas_date', (None, date))
@pytest.mark.parametrize('first_samp', (0, 10000))
def test_annotate_amplitude_with_overlap(meas_date, first_samp):
    """Test cases with overlap between annotations."""
    n_ch, n_times = 11, 1000
    data = np.random.RandomState(0).randn(n_ch, n_times)
    assert not (np.diff(data, axis=-1) == 0).any()  # nothing flat at first
    info = create_info(n_ch, 1000., 'eeg')
    # from annotate_flat: test first_samp != for gh-6295
    raw = RawArray(data, info, first_samp=first_samp)
    raw.info['bads'] = [raw.ch_names[-1]]
    raw.set_meas_date(meas_date)

    # -- overlap between peak and flat --
    raw_ = raw.copy()
    raw_._data[0, 800:] = 0.
    raw_._data[1, 700:900] = np.arange(0, 200 * 10, 10)
    raw_._data[1, 900:] += raw_._data[1, 899]  # add offset for next samples
    annots, bads = annotate_amplitude(raw_, peak=5, flat=0, bad_percent=25)
    assert len(annots) == 2
    assert len(bads) == 0
    # check annotation instance
    assert all(annot['description'] in ('BAD_flat', 'BAD_peak')
               for annot in annots)
    for annot in annots:
        start_idx = 700 if annot['description'] == 'BAD_peak' else 800
        stop_idx = 899 if annot['description'] == 'BAD_peak' else -1
        _check_annotation(raw_, annot, meas_date, first_samp, start_idx,
                          stop_idx)

    # -- overlap between peak and peak on same channel --
    raw_ = raw.copy()
    raw_._data[0, 700:900] = np.arange(0, 200 * 10, 10)
    raw_._data[0, 800:] = np.arange(1000, 300 * 10, 10)
    annots, bads = annotate_amplitude(raw_, peak=5, flat=None, bad_percent=50)
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    assert annots[0]['description'] == 'BAD_peak'
    _check_annotation(raw_, annots[0], meas_date, first_samp, 700, -1)

    # -- overlap between flat and flat on different channel --
    raw_ = raw.copy()
    raw_._data[0, 700:900] = 0.
    raw_._data[1, 800:] = 0.
    annots, bads = annotate_amplitude(raw_, peak=None, flat=0.01,
                                      bad_percent=50)
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    assert annots[0]['description'] == 'BAD_flat'
    _check_annotation(raw_, annots[0], meas_date, first_samp, 700, -1)


@pytest.mark.parametrize('meas_date', (None, date))
@pytest.mark.parametrize('first_samp', (0, 10000))
def test_annotate_amplitude_multiple_ch_types(meas_date, first_samp):
    """Test cases with several channel types."""
    n_ch, n_times = 11, 1000
    data = np.random.RandomState(0).randn(n_ch, n_times)
    assert not (np.diff(data, axis=-1) == 0).any()  # nothing flat at first
    info = create_info(n_ch, 1000.,
                       ['eeg'] * 3 + ['mag'] * 2 + ['grad'] * 4 + ['eeg'] * 2)
    # from annotate_flat: test first_samp != for gh-6295
    raw = RawArray(data, info, first_samp=first_samp)
    raw.info['bads'] = [raw.ch_names[-1]]
    raw.set_meas_date(meas_date)

    # -- 2 channel types both to annotate --
    raw_ = raw.copy()
    raw_._data[1, 800:] = 0.
    raw_._data[5, :200] = np.arange(0, 200 * 10, 10)
    raw_._data[5, 200:] += raw_._data[5, 199]  # add offset for next samples
    annots, bads = annotate_amplitude(raw_, peak=5, flat=0, bad_percent=50)
    assert len(annots) == 2
    assert len(bads) == 0
    # check annotation instance
    assert all(annot['description'] in ('BAD_flat', 'BAD_peak')
               for annot in annots)
    for annot in annots:
        start_idx = 0 if annot['description'] == 'BAD_peak' else 800
        stop_idx = 199 if annot['description'] == 'BAD_peak' else -1
        _check_annotation(raw_, annot, meas_date, first_samp, start_idx,
                          stop_idx)

    # -- 2 channel types, one flat picked, one not picked --
    raw_ = raw.copy()
    raw_._data[1, 800:] = 0.
    raw_._data[5, :200] = np.arange(0, 200 * 10, 10)
    raw_._data[5, 200:] += raw_._data[5, 199]  # add offset for next samples
    annots, bads = annotate_amplitude(raw_, peak=5, flat=0, bad_percent=50,
                                      picks='eeg')
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    _check_annotation(raw_, annots[0], meas_date, first_samp, 800, -1)
    assert annots[0]['description'] == 'BAD_flat'

    # -- 2 channel types, one flat, one not picked, reverse --
    raw_ = raw.copy()
    raw_._data[1, 800:] = 0.
    raw_._data[5, :200] = np.arange(0, 200 * 10, 10)
    raw_._data[5, 200:] += raw_._data[5, 199]  # add offset for next samples
    annots, bads = annotate_amplitude(raw_, peak=5, flat=0, bad_percent=50,
                                      picks='grad')
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    _check_annotation(raw_, annots[0], meas_date, first_samp, 0, 199)
    assert annots[0]['description'] == 'BAD_peak'


@testing.requires_testing_data
def test_flat_bad_acq_skip():
    """Test that acquisition skips are handled properly."""
    # -- file with a couple of skip and flat channels --
    raw = read_raw_fif(skip_fname, preload=True)
    annots, bads = annotate_amplitude(raw, flat=0)
    assert len(annots) == 0
    assert bads == [  # MaxFilter finds the same 21 channels
        'MEG%04d' % (int(num),) for num in
        '141 331 421 431 611 641 1011 1021 1031 1241 1421 '
        '1741 1841 2011 2131 2141 2241 2531 2541 2611 2621'.split()]

    # -- overlap of flat segment with bad_acq_skip --
    n_ch, n_times = 11, 1000
    data = np.random.RandomState(0).randn(n_ch, n_times)
    assert not (np.diff(data, axis=-1) == 0).any()  # nothing flat at first
    info = create_info(n_ch, 1000., 'eeg')
    raw = RawArray(data, info, first_samp=0)
    raw.info['bads'] = [raw.ch_names[-1]]
    bad_acq_skip = Annotations([0.5], [0.2], ['bad_acq_skip'], orig_time=None)
    raw.set_annotations(bad_acq_skip)
    # add flat channel overlapping with the left edge of bad_acq_skip
    raw_ = raw.copy()
    raw_._data[0, 400:600] = 0.
    annots, bads = annotate_amplitude(raw_, peak=None, flat=0, bad_percent=25)
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    assert annots[0]['description'] == 'BAD_flat'
    _check_annotation(raw_, annots[0], None, 0, 400, 499)

    # add flat channel overlapping with the right edge of bad_acq_skip
    raw_ = raw.copy()
    raw_._data[0, 600:800] = 0.
    annots, bads = annotate_amplitude(raw_, peak=None, flat=0, bad_percent=25)
    assert len(annots) == 1
    assert len(bads) == 0
    # check annotation instance
    assert annots[0]['description'] == 'BAD_flat'
    _check_annotation(raw_, annots[0], None, 0, 700, 799)

    # add flat channel overlapping entirely with bad_acq_skip
    raw_ = raw.copy()
    raw_._data[0, 200:800] = 0.
    annots, bads = annotate_amplitude(raw_, peak=None, flat=0, bad_percent=41)
    assert len(annots) == 2
    assert len(bads) == 0
    # check annotation instance
    annots = sorted(annots, key=lambda x: x['onset'])
    assert all(annot['description'] == 'BAD_flat' for annot in annots)
    _check_annotation(raw_, annots[0], None, 0, 200, 500)
    _check_annotation(raw_, annots[1], None, 0, 700, 799)


def _check_annotation(raw, annot, meas_date, first_samp, start_idx, stop_idx):
    """Util function to check an annotation."""
    assert meas_date == annot['orig_time']
    if meas_date is None:
        assert np.isclose(raw.times[start_idx], annot['onset'], atol=1e-4)
        assert np.isclose(
            raw.times[stop_idx], annot['onset'] + annot['duration'], atol=1e-4)
    else:
        first_time = first_samp / raw.info['sfreq']  # because of meas_date
        assert np.isclose(
            raw.times[start_idx], annot['onset'] - first_time, atol=1e-4)
        assert np.isclose(
            raw.times[stop_idx],
            annot['onset'] + annot['duration'] - first_time,
            atol=1e-4)


def test_invalid_arguments():
    """Test error messages raised by invalid arguments."""
    n_ch, n_times = 2, 100
    data = np.random.RandomState(0).randn(n_ch, n_times)
    info = create_info(n_ch, 100., 'eeg')
    raw = RawArray(data, info, first_samp=0)

    # negative floats PTP
    with pytest.raises(ValueError,
                       match="Argument 'flat' should define a positive "
                             "threshold. Provided: '-1'."):
        annotate_amplitude(raw, peak=None, flat=-1)
    with pytest.raises(ValueError,
                       match="Argument 'peak' should define a positive "
                             "threshold. Provided: '-1'."):
        annotate_amplitude(raw, peak=-1, flat=None)

    # negative PTP threshold for one channel type
    with pytest.raises(ValueError,
                       match="Argument 'flat' should define positive "
                             "thresholds. Provided for channel type "
                             "'eog': '-1'."):
        annotate_amplitude(raw, peak=None, flat=dict(eeg=1, eog=-1))
    with pytest.raises(ValueError,
                       match="Argument 'peak' should define positive "
                             "thresholds. Provided for channel type "
                             "'eog': '-1'."):
        annotate_amplitude(raw, peak=dict(eeg=1, eog=-1), flat=None)

    # test both PTP set to None
    with pytest.raises(ValueError,
                       match="At least one of the arguments 'peak' or 'flat' "
                             "must not be None."):
        annotate_amplitude(raw, peak=None, flat=None)

    # bad_percent outside [0, 100]
    with pytest.raises(ValueError,
                       match="Argument 'bad_percent' should define a "
                             "percentage between 0% and 100%. Provided: "
                             "-1.0%."):
        annotate_amplitude(raw, peak=dict(eeg=1), flat=None, bad_percent=-1)

    # min_duration negative
    with pytest.raises(ValueError,
                       match="Argument 'min_duration' should define a "
                             "positive duration in seconds. Provided: "
                             "'-1.0' seconds."):
        annotate_amplitude(raw, peak=dict(eeg=1), flat=None, min_duration=-1)

    # min_duration equal to the raw duration
    with pytest.raises(
            ValueError,
            match=re.escape("Argument 'min_duration' should define a "
                            "positive duration in seconds shorter than the "
                            "raw duration (1.0 seconds). Provided: "
                            "'1.0' seconds.")
    ):
        annotate_amplitude(raw, peak=dict(eeg=1), flat=None, min_duration=1.)

    # min_duration longer than the raw duration
    with pytest.raises(
            ValueError,
            match=re.escape("Argument 'min_duration' should define a "
                            "positive duration in seconds shorter than the "
                            "raw duration (1.0 seconds). Provided: "
                            "'10.0' seconds.")
    ):
        annotate_amplitude(raw, peak=dict(eeg=1), flat=None, min_duration=10)
