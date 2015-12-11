# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

from mne.io import read_raw_eeglab
from mne.io.tests.test_raw import _test_raw_reader

base_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
fname = op.join(base_dir, 'eeglab_data.set')
montage = op.join(base_dir, 'eeglab_chan32.locs')


def test_io_set():
    """Test importing EEGLAB .set files"""
    _test_raw_reader(read_raw_eeglab, input_fname=fname, montage=montage)
