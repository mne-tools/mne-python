# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from mne.datasets import sample
from mne.label import read_label
from mne import fiff, read_cov, read_forward_solution
from mne.mixed_norm.inverse import mixed_norm


examples_folder = op.join(op.dirname(__file__), '..', '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                            'sample_audvis-meg-oct-6-fwd.fif')
label = 'Aud-rh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)

evoked = fiff.Evoked(fname_data, setno=1, baseline=(None, 0))

# Read noise covariance matrix
cov = read_cov(fname_cov)

# Handling average file
setno = 0
loose = None

evoked = fiff.read_evoked(fname_data, setno=setno, baseline=(None, 0))
evoked.crop(tmin=0.08, tmax=0.12)

# Handling forward solution
forward = read_forward_solution(fname_fwd, force_fixed=True)
label = read_label(fname_label)


def test_MxNE_inverse():
    """Test MxNE inverse computation"""
    alpha = 60  # spatial regularization parameter
    stc = mixed_norm(evoked, forward, cov, alpha, loose=None, depth=0.9,
                     maxit=500, tol=1e-4, active_set_size=10)

    assert_array_almost_equal(stc.times, evoked.times, 5)
    assert_true(stc.vertno[1][0] in label['vertices'])
