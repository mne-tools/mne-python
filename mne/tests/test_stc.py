import os.path as op

from numpy.testing import assert_array_almost_equal

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')


def test_io_stc():
    """Test IO for STC files
    """
    stc = mne.read_stc(fname)

    mne.write_stc("tmp.stc", stc['tmin'], stc['tstep'],
                             stc['vertices'], stc['data'])
    stc2 = mne.read_stc("tmp.stc")

    assert_array_almost_equal(stc['data'], stc2['data'])
    assert_array_almost_equal(stc['tmin'], stc2['tmin'])
    assert_array_almost_equal(stc['vertices'], stc2['vertices'])
    assert_array_almost_equal(stc['tstep'], stc2['tstep'])
