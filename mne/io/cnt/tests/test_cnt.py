
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import pytest

from mne import pick_types
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.cnt import read_raw_cnt

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'CNT', 'scan41_short.cnt')


@testing.requires_testing_data
def test_data():
    """Test reading raw cnt files."""
    with pytest.warns(RuntimeWarning, match='number of bytes'):
        raw = _test_raw_reader(read_raw_cnt, montage=None, input_fname=fname,
                               eog='auto', misc=['NA1', 'LEFT_EAR'])
    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert len(eog_chs) == 2  # test eog='auto'
    assert raw.info['bads'] == ['LEFT_EAR', 'VEOGR']  # test bads

    # XXX: the data has "05/10/200 17:35:31" so it is set to None
    assert raw.info['meas_date'] is None


run_tests_if_main()
