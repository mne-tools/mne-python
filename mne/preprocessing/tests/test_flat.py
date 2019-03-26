# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray
from mne.preprocessing import mark_flat


def test_mark_flat():
    """Test marking flat segments."""
    # Test if ECG analysis will work on data that is not preloaded
    n_ch, n_times = 11, 1000
    data = np.random.RandomState(0).randn(n_ch, n_times)
    assert not (np.diff(data, axis=-1) == 0).any()  # nothing flat at first
    info = create_info(n_ch, 1000., 'eeg')
    raw = RawArray(data, info)
    raw.info['bads'] = [raw.ch_names[-1]]
    n_good = n_ch - 1

    #
    # First make a channel flat the whole time
    #
    raw_0 = raw.copy()
    raw_0._data[0] = 0.
    for kwargs, bads, want_times in [
            # These will all mark spatially
            (dict(ratio='inf'), [raw.ch_names[0]], n_times),
            (dict(ratio=n_good), [raw.ch_names[0]], n_times),
            (dict(), [raw.ch_names[0]], n_times),  # default (1)
            # right at this limit is the same
            (dict(ratio=1. / (n_good - 0.1)), [raw.ch_names[0]], n_times),
            # then it switches
            (dict(ratio=1. / n_good), [], 0),  # same
            (dict(ratio=0), [], 0)]:  # ratio=0 will mark all times
        raw_time = mark_flat(raw_0.copy(), verbose='debug', **kwargs)
        want_bads = raw.info['bads'] + bads
        assert raw_time.info['bads'] == want_bads
        n_good_times = raw_time.get_data(reject_by_annotation='omit').shape[1]
        assert n_good_times == want_times

    #
    # Now make a channel flat for 20% of the time points
    #
    raw_0 = raw.copy()
    bad_t_div = 5
    above_thresh = float(bad_t_div + 0.1) / n_good
    below_thresh = float(bad_t_div) / n_good
    assert n_times % bad_t_div == 0
    raw_0._data[0, :n_times // bad_t_div] = 0.
    n_good_times = n_times - n_times // bad_t_div
    for kwargs, bads, want_times in [
            # These will all mark spatially
            (dict(ratio='inf'), [raw.ch_names[0]], n_times),
            (dict(ratio=n_good), [raw.ch_names[0]], n_times),
            (dict(), [raw.ch_names[0]], n_times),  # default (1)
            # right at this limit is the same
            (dict(ratio=above_thresh), [raw.ch_names[0]], n_times),
            # then it switches
            (dict(ratio=below_thresh), [], n_good_times),  # same
            (dict(ratio=0), [], n_good_times)]:  # ratio=0 will mark all times
        raw_time = mark_flat(raw_0.copy(), verbose='debug', **kwargs)
        want_bads = raw.info['bads'] + bads
        assert raw_time.info['bads'] == want_bads
        n_good_times = raw_time.get_data(reject_by_annotation='omit').shape[1]
        assert n_good_times == want_times

    with pytest.raises(TypeError, match='must be an instance of BaseRaw'):
        mark_flat(0.)
    with pytest.raises(ValueError, match='not convert string to float'):
        mark_flat(raw, 'x')
