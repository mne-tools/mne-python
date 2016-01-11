
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect
from nose.tools import assert_equal

from mne.io import read_raw_nicolet
from mne.io.nicolet import read_nicolet_annotations
from mne.io.tests.test_raw import _test_raw_reader

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    _test_raw_reader(read_raw_nicolet, input_fname=fname, ch_type='eeg',
                     ecg='auto', eog='auto', emg='auto', misc=['PHO'])


def test_annotations():
    """Test reading of annotations from file."""
    fname = op.join(base_dir, 'test.json')
    annot = read_nicolet_annotations(fname, '223')
    assert_equal(len(annot.segments), 2)
