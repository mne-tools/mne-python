
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#         Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from mne import pick_types
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.cnt import read_raw_cnt
from mne.io.cnt.cnt import _read_annotations_cnt
from mne.annotations import read_annotations

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'CNT', 'scan41_short.cnt')


@testing.requires_testing_data
def test_data(recwarn):
    """Test reading raw cnt files."""
    raw = _test_raw_reader(read_raw_cnt, montage=None, input_fname=fname,
                           eog='auto', misc=['NA1', 'LEFT_EAR'])

    # inspect the warnings
    ACCEPTED_WARNING_MSG_ENDINGS = ('Setting to None.', 'Defaulting to 2.')
    assert recwarn.pop(DeprecationWarning)
    assert all([issubclass(_.category, RuntimeWarning) for _ in recwarn])
    assert all([_.message.args[0].endswith(ACCEPTED_WARNING_MSG_ENDINGS)
                for _ in recwarn])
    recwarn.clear()

    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert len(eog_chs) == 2  # test eog='auto'
    assert raw.info['bads'] == ['LEFT_EAR', 'VEOGR']  # test bads

    # XXX: the data has "05/10/200 17:35:31" so it is set to None
    assert raw.info['meas_date'] is None


@testing.requires_testing_data
def test_compare_events_and_annotations(recwarn):
    from mne import find_events
    from numpy.testing import assert_array_equal

    raw = read_raw_cnt(input_fname=fname, montage=None, preload=False)
    events = find_events(raw)

    annot = _read_annotations_cnt(fname)
    assert_array_equal(annot.onset[:-1], events[:, 0] / raw.info['sfreq'])


@testing.requires_testing_data
def test_read_annotations():
    """Test reading for annotations from a .CNT file."""
    annot = read_annotations(fname)
    assert len(annot) == 6


run_tests_if_main()
