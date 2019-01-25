import os.path as op
import itertools as itt

from numpy.testing import assert_equal, assert_array_equal
import numpy as np

import pytest

from mne.datasets import testing
# from mne.rank import _estimate_rank_meeg_cov
# from mne import read_evokeds, read_cov #, prepare_noise_cov
# from mne import compute_raw_covariance
# from mne import pick_types, pick_info
# from mne.io.proj import _has_eeg_average_ref_proj
from mne.io.pick import channel_type, _picks_by_type
from mne.io import read_raw_fif
from mne.proj import compute_proj_raw
# from mne.io.proc_history import _get_sss_rank, _get_rank_sss
from mne.rank import estimate_rank
from mne.io.proc_history import _get_rank_sss


base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
cov_fname = op.join(base_dir, 'test-cov.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
ave_fname = op.join(base_dir, 'test-ave.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')

testing_path = testing.data_path(download=False)
data_dir = op.join(testing_path, 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


def test_estimate_rank():
    """Test rank estimation."""
    data = np.eye(10)
    assert_array_equal(estimate_rank(data, return_singular=True)[1],
                       np.ones(10))
    data[0, 0] = 0
    assert_equal(estimate_rank(data), 9)
    pytest.raises(ValueError, estimate_rank, data, 'foo')


@pytest.mark.slowtest
@testing.requires_testing_data
def test_rank_estimation():
    """Test raw rank estimation."""
    iter_tests = itt.product(
        [fif_fname, hp_fif_fname],  # sss
        ['norm', dict(mag=1e11, grad=1e9, eeg=1e5)]
    )
    for fname, scalings in iter_tests:
        raw = read_raw_fif(fname).crop(0, 4.).load_data()
        (_, picks_meg), (_, picks_eeg) = _picks_by_type(raw.info,
                                                        meg_combined=True)
        n_meg = len(picks_meg)
        n_eeg = len(picks_eeg)

        if len(raw.info['proc_history']) == 0:
            expected_rank = n_meg + n_eeg
        else:
            expected_rank = _get_rank_sss(raw.info) + n_eeg
        assert_array_equal(raw.estimate_rank(scalings=scalings), expected_rank)
        assert_array_equal(raw.estimate_rank(picks=picks_eeg,
                                             scalings=scalings), n_eeg)
        if 'sss' in fname:
            raw.add_proj(compute_proj_raw(raw))
        raw.apply_proj()
        n_proj = len(raw.info['projs'])
        assert_array_equal(raw.estimate_rank(tstart=0, tstop=3.,
                                             scalings=scalings),
                           expected_rank - (0 if 'sss' in fname else n_proj))
