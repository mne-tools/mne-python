# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.signal import hilbert

from mne.connectivity import envelope_correlation


def _compute_corrs_orig(data):
    # This is the version of the code by Sheraz and Denis.
    # For this version (epochs, labels, time) must be -> (labels, time, epochs)
    data = np.transpose(data, (1, 2, 0))
    corr_mats = np.empty((data.shape[0], data.shape[0], data.shape[2]))
    for index, label_data in enumerate(data):
        label_data_orth = np.imag(label_data * (data.conj() / np.abs(data)))
        label_data_orig = np.abs(label_data)
        label_data_cont = np.transpose(
            np.dstack((label_data_orig, np.transpose(label_data_orth,
                                                     (1, 2, 0)))), (1, 2, 0))
        corr_mats[index] = np.array([np.corrcoef(dat)
                                     for dat in label_data_cont])[:, 0, 1:].T
    corr_mats = np.transpose(corr_mats, (2, 0, 1))
    corr = np.mean(np.array([(np.abs(corr_mat) + np.abs(corr_mat).T) / 2.
                             for corr_mat in corr_mats]), axis=0)
    return corr


def test_envelope_correlation():
    """Test the envelope correlation function."""
    rng = np.random.RandomState(0)
    data = rng.randn(2, 4, 64)
    data_hilbert = hilbert(data, axis=-1)
    corr_orig = _compute_corrs_orig(data_hilbert)
    assert (0 < corr_orig).all()
    assert (corr_orig < 1).all()
    # using complex data
    corr = envelope_correlation(data_hilbert)
    assert_allclose(corr, corr_orig)
    # using callable
    corr = envelope_correlation(data_hilbert,
                                combine=lambda data: np.mean(data, axis=0))
    assert_allclose(corr, corr_orig)
    # do Hilbert internally, and don't combine
    corr = envelope_correlation(data, combine=None)
    assert corr.shape == (data.shape[0],) + corr_orig.shape
    corr = np.mean(corr, axis=0)
    assert_allclose(corr, corr_orig)
    # degenerate
    with pytest.raises(ValueError, match='float'):
        envelope_correlation(data.astype(int))
    with pytest.raises(ValueError, match='entry in data must be 2D'):
        envelope_correlation(data[np.newaxis])
    with pytest.raises(ValueError, match='n_nodes mismatch'):
        envelope_correlation([rng.randn(2, 8), rng.randn(3, 8)])
    with pytest.raises(ValueError, match='mean or callable'):
        envelope_correlation(data, 1.)
    with pytest.raises(ValueError, match='Combine option'):
        envelope_correlation(data, 'foo')
    with pytest.raises(ValueError, match='Invalid value.*orthogonalize.*'):
        envelope_correlation(data, orthogonalize='foo')

    corr_plain = envelope_correlation(data, combine=None, orthogonalize=False)
    assert corr_plain.shape == (data.shape[0],) + corr_orig.shape
    assert np.min(corr_plain) < 0
    corr_plain_mean = np.mean(corr_plain, axis=0)
    assert_allclose(np.diag(corr_plain_mean), 1)
    np_corr = np.array([np.corrcoef(np.abs(x)) for x in data_hilbert])
    assert_allclose(corr_plain, np_corr)
