
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import warnings

from nose.tools import assert_equal, assert_true

from mne import pick_types
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.cnt import read_raw_cnt

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'CNT', 'scan41_short.cnt')


@testing.requires_testing_data
def test_data():
    """Test reading raw cnt files."""
    with warnings.catch_warnings(record=True) as w:
        raw = _test_raw_reader(read_raw_cnt, montage=None, input_fname=fname,
                               eog='auto', misc=['NA1', 'LEFT_EAR'])
    assert_true(all('meas date' in str(ww.message) or
                    'number of bytes' in str(ww.message) for ww in w))
    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert_equal(len(eog_chs), 2)  # test eog='auto'
    assert_equal(raw.info['bads'], ['LEFT_EAR', 'VEOGR'])  # test bads

run_tests_if_main()
