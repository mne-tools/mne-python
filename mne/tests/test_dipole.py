import os.path as op
from nose.tools import assert_true

from mne import read_dip
from mne.datasets import sample

data_path = sample.data_path(download=False)
dip_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_set1.dip')


@sample.requires_sample_data
def test_io_dip():
    """Test IO for .dip files
    """
    time, pos, amplitude, ori, gof = read_dip(dip_fname)

    assert_true(pos.shape[1] == 3)
    assert_true(ori.shape[1] == 3)
    assert_true(len(time) == len(pos))
    assert_true(len(time) == gof.size)
    assert_true(len(time) == amplitude.size)
