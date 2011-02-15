import os
import os.path as op

from numpy.testing import assert_array_almost_equal, assert_equal

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')

def test_io_forward():
    """Test IO for forward solutions
    """
    fwd = mne.read_forward_solution(fname)
    fwd = mne.read_forward_solution(fname, force_fixed=True)
    leadfield = fwd['sol']['data']
