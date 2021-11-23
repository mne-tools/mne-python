# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
import numpy as np
import pytest

from mne import create_info
from mne.datasets import testing
from mne.io import RawArray, read_raw_fif
from mne.preprocessing import annotate_flat

data_path = testing.data_path(download=False)
skip_fname = op.join(data_path, 'misc', 'intervalrecording_raw.fif')


@pytest.mark.parametrize('first_samp', (0, 10000))
def test_annotate_flat(first_samp):
    """Test marking flat segments."""
    # Test if ECG analysis will work on data that is not preloaded
    n_ch, n_times = 11, 1000
    data = np.random.RandomState(0).randn(n_ch, n_times)
    assert not (np.diff(data, axis=-1) == 0).any()  # nothing flat at first
    info = create_info(n_ch, 1000., 'eeg')
    with info._unlock():
        info['meas_date'] = (1, 2)
    # test first_samp != for gh-6295
    raw = RawArray(data, info, first_samp=first_samp)
    raw.info['bads'] = [raw.ch_names[-1]]

    #
    # First make a channel flat the whole time
    #
    raw_0 = raw.copy()
    raw_0._data[0] = 0.
    for kwargs, bads, want_times in [
            # Anything < 1 will mark spatially
            (dict(bad_percent=100.), [], 0),
            (dict(bad_percent=99.9), [raw.ch_names[0]], n_times),
            (dict(), [raw.ch_names[0]], n_times)]:  # default (1)
        raw_time = raw_0.copy()
        annot, got_bads = annotate_flat(raw_0, verbose='debug', **kwargs)
        assert got_bads == bads
        raw_time.set_annotations(raw_time.annotations + annot)
        raw_time.info['bads'] += got_bads
        n_good_times = raw_time.get_data(reject_by_annotation='omit').shape[1]
        assert n_good_times == want_times
    #
    # Now make a channel flat for 20% of the time points
    #
    raw_0 = raw.copy()
    n_good_times = int(round(0.8 * n_times))
    raw_0._data[0, n_good_times:] = 0.
    threshold = 100 * (n_times - n_good_times) / n_times
    for kwargs, bads, want_times in [
            # Should change behavior at bad_percent=20
            (dict(bad_percent=100), [], n_good_times),
            (dict(bad_percent=threshold), [], n_good_times),
            (dict(bad_percent=threshold - 1e-5), [raw.ch_names[0]], n_times),
            (dict(), [raw.ch_names[0]], n_times)]:
        annot, got_bads = annotate_flat(raw_0, verbose='debug', **kwargs)
        assert got_bads == bads
        raw_time = raw_0.copy()
        raw_time.set_annotations(raw_time.annotations + annot)
        raw_time.info['bads'] += got_bads
        n_good_times = raw_time.get_data(reject_by_annotation='omit').shape[1]
        assert n_good_times == want_times

    with pytest.raises(TypeError, match='must be an instance of BaseRaw'):
        annotate_flat(0.)
    with pytest.raises(ValueError, match='not convert string to float'):
        annotate_flat(raw, 'x')


@testing.requires_testing_data
def test_flat_acq_skip():
    """Test that acquisition skips are handled properly."""
    raw = read_raw_fif(skip_fname).load_data()
    annot, bads = annotate_flat(raw)
    assert len(annot) == 0
    assert bads == [  # MaxFilter finds the same 21 channels
        'MEG%04d' % (int(num),) for num in
        '141 331 421 431 611 641 1011 1021 1031 1241 1421 '
        '1741 1841 2011 2131 2141 2241 2531 2541 2611 2621'.split()]
