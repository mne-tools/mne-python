
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#         Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from mne import pick_types
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.cnt import read_raw_cnt
from mne.annotations import read_annotations

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'CNT', 'scan41_short.cnt')


@testing.requires_testing_data
def test_data():
    """Test reading raw cnt files."""
    with pytest.warns(RuntimeWarning, match='number of bytes'):
        raw = _test_raw_reader(read_raw_cnt, input_fname=fname,
                               eog='auto', misc=['NA1', 'LEFT_EAR'])

    # make sure we use annotations event if we synthesized stim
    assert len(raw.annotations) == 6

    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert len(eog_chs) == 2  # test eog='auto'
    assert raw.info['bads'] == ['LEFT_EAR', 'VEOGR']  # test bads

    # XXX: the data has "05/10/200 17:35:31" so it is set to None
    assert raw.info['meas_date'] is None


@testing.requires_testing_data
def test_compare_events_and_annotations():
    """Test comparing annotations and events."""
    with pytest.warns(RuntimeWarning, match='Could not parse meas date'):
        raw = read_raw_cnt(fname)
    events = np.array([[333, 0, 7],
                       [1010, 0, 7],
                       [1664, 0, 109],
                       [2324, 0, 7],
                       [2984, 0, 109]])

    annot = read_annotations(fname)
    assert len(annot) == 6
    assert_array_equal(annot.onset[:-1], events[:, 0] / raw.info['sfreq'])
    assert 'STI 014' not in raw.info['ch_names']


run_tests_if_main()
