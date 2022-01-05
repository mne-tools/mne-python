# Author: Mathieu Scheltienne <mathieu.scheltienne@fcbg.ch>
#
# License: BSD-3-Clause

import datetime
import itertools
import numpy as np
import pytest
import re

from mne import create_info
from mne.io import RawArray
from mne.preprocessing import annotate_amplitude


date = datetime.datetime(2021, 12, 10, 7, 52, 24, 405305,
                         tzinfo=datetime.timezone.utc)


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
