# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#         Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD-3-Clause


import os.path as op

import numpy as np
from numpy.testing import assert_allclose
import pytest

from mne import read_source_spaces
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, source_estimate_quantification

data_path = testing.data_path(download=False)
src_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')


@testing.requires_testing_data
def test_metrics():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    times = np.arange(600) / 1000.
    rng = np.random.RandomState(42)
    stc1 = simulate_sparse_stc(src, n_dipoles=2, times=times, random_state=rng)
    stc2 = simulate_sparse_stc(src, n_dipoles=2, times=times, random_state=rng)
    with pytest.warns(FutureWarning, match="source_estimate_quantification"):
        E1_rms = source_estimate_quantification(stc1, stc1, metric='rms')
        E2_rms = source_estimate_quantification(stc2, stc2, metric='rms')
        E1_cos = source_estimate_quantification(stc1, stc1, metric='cosine')
        E2_cos = source_estimate_quantification(stc2, stc2, metric='cosine')
    # ### Tests to add
    assert (E1_rms == 0.)
    assert (E2_rms == 0.)
    assert_allclose(E1_cos, 0., atol=1e-08)
    assert_allclose(E2_cos, 0., atol=1e-08)

    stc_bad = stc2.copy().crop(0, 0.5)
    with pytest.warns(FutureWarning, match="source_estimate_quantification"):
        with pytest.raises(ValueError, match='must have the same size'):
            source_estimate_quantification(stc1, stc_bad)
    stc_bad = stc2.copy()
    stc_bad.tmin -= 0.1
    with pytest.warns(FutureWarning, match="source_estimate_quantification"):
        with pytest.raises(ValueError, match='Times.*must match'):
            source_estimate_quantification(stc1, stc_bad)
    with pytest.warns(FutureWarning, match="source_estimate_quantification"):
        with pytest.raises(ValueError, match="Invalid value for the 'metric'"):
            source_estimate_quantification(stc1, stc2, metric='foo')
