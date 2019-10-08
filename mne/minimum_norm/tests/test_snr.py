# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
from os import path as op
import numpy as np
from numpy.testing import assert_allclose

from mne import read_evokeds
from mne.datasets import testing
from mne.minimum_norm import read_inverse_operator, estimate_snr

from mne.utils import _TempDir, requires_mne, run_subprocess

s_path = op.join(testing.data_path(download=False), 'MEG', 'sample')
fname_inv = op.join(s_path, 'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_evoked = op.join(s_path, 'sample_audvis-ave.fif')


@testing.requires_testing_data
@requires_mne
def test_snr():
    """Test SNR calculation."""
    tempdir = _TempDir()
    inv = read_inverse_operator(fname_inv)
    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0]
    snr = estimate_snr(evoked, inv)[0]
    orig_dir = os.getcwd()
    os.chdir(tempdir)
    try:
        cmd = ['mne_compute_mne', '--inv', fname_inv, '--meas', fname_evoked,
               '--snronly', '--bmin', '-200', '--bmax', '0']
        run_subprocess(cmd)
    except Exception:
        pass  # this returns 1 for some reason
    finally:
        os.chdir(orig_dir)
    times, snr_c, _ = np.loadtxt(op.join(tempdir, 'SNR')).T
    assert_allclose(times / 1000., evoked.times, atol=1e-2)
    assert_allclose(snr, snr_c, atol=1e-2, rtol=1e-2)
