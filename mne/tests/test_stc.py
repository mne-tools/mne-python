import os
import os.path as op

from numpy.testing import assert_array_almost_equal, assert_equal

import mne

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis-ave-7-meg-lh.stc')

def test_io_stc():
    """Test IO for STC files
    """
    stc = mne.read_stc(fname)

    mne.write_stc("tmp.stc", stc)
    stc2 = mne.read_stc("tmp.stc")

    assert_array_almost_equal(stc['data'], stc2['data'])
    assert_array_almost_equal(stc['tmin'], stc2['tmin'])
    assert_array_almost_equal(stc['vertices'], stc2['vertices'])
    assert_array_almost_equal(stc['tstep'], stc2['tstep'])
