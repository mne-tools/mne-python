# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from mne.time_frequency import morlet
from mne.preprocessing.ctps_ import (ctps, _prob_kuiper,
                                     _compute_normalized_phase)

###############################################################################
# Generate testing signal

tmin = -0.3
sfreq = 1000.  # Hz
tstep = 1. / sfreq
n_samples = 600
times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

# Generate times series from Morlet wavelet
single_trial = np.zeros((1, len(times)))
Ws = morlet(sfreq, [3], n_cycles=[1])

single_trial[0][:len(Ws[0])] = np.real(Ws[0])
roll_to = 300 - 265  # shift data to center of time window
single_trial = np.roll(single_trial, roll_to)
rng = np.random.RandomState(42)


def get_data(n_trials, j_extent):
    """Generate ground truth and testing data."""
    ground_truth = np.tile(single_trial, n_trials)
    my_shape = n_trials, 1, 600
    random_data = rng.random_sample(my_shape)
    rand_ints = rng.randint(-j_extent, j_extent, n_trials)
    jittered_data = np.array([np.roll(single_trial, i) for i in rand_ints])
    data = np.concatenate([ground_truth.reshape(my_shape),
                           jittered_data.reshape(my_shape),
                           random_data.reshape(my_shape)], 1)

    assert data.shape == (n_trials, 3, 600)
    return data


# vary extent of jittering --> creates phaselocks at the borders if
# 2 * extent != n_samples
iter_test_ctps = enumerate(zip([400, 400], [150, 300], [0.6, 0.2]))


def test_ctps():
    """Test basic ctps functionality."""
    for ii, (n_trials, j_extent, pk_max) in iter_test_ctps:
        data = get_data(n_trials, j_extent)
        ks_dyn, pk_dyn, phase_trial = ctps(data)
        data2 = _compute_normalized_phase(data)
        ks_dyn2, pk_dyn2, phase_trial2 = ctps(data2, is_raw=False)
        for a, b in zip([ks_dyn, pk_dyn, phase_trial],
                        [ks_dyn2, pk_dyn2, data2]):
            assert_array_equal(a, b)
            assert (a.min() >= 0)
            assert (a.max() <= 1)
            assert (b.min() >= 0)
            assert (b.max() <= 1)

        # test for normalization
        assert ((pk_dyn.min() > 0.0) or (pk_dyn.max() < 1.0))
        # test shapes
        assert (phase_trial.shape == data.shape)
        assert (pk_dyn.shape == data.shape[1:])
        # tets ground_truth + random + jittered case
        assert (pk_dyn[0].max() == 1.0)
        assert (len(np.unique(pk_dyn[0])) == 1.0)
        assert (pk_dyn[1].max() < pk_max)
        assert (pk_dyn[2].max() > 0.3)
        if ii < 1:
            pytest.raises(ValueError, ctps, data[:, :, :, None])

    assert (_prob_kuiper(1.0, 400) == 1.0)
    # test vecrosization
    assert_array_equal(_prob_kuiper(np.array([1.0, 1.0]), 400),
                       _prob_kuiper(np.array([1.0, 1.0]), 400))
    assert (_prob_kuiper(0.1, 400) < 0.1)
