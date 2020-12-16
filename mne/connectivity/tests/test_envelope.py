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
    n_epochs, n_labels, _ = data.shape
    corr = np.zeros((n_labels, n_labels))
    for epoch_data in data:
        for ii in range(n_labels):
            for jj in range(n_labels):
                # Get timeseries for each pair
                x, y = epoch_data[ii], epoch_data[jj]
                x_mag = np.abs(x)
                x_conj_scaled = x.conj()
                x_conj_scaled /= x_mag
                # Calculate orthogonalization
                y_orth_x = (y * x_conj_scaled).imag
                y_orth_x_mag = np.abs(y_orth_x)
                # Estimate correlation
                corr[ii, jj] += np.abs(np.corrcoef(x_mag, y_orth_x_mag)[0, 1])
    corr = (corr + corr.T) / (2. * n_epochs)
    corr.flat[::n_labels + 1] = 0.
    return corr


def test_envelope_correlation():
    """Test the envelope correlation function."""
    rng = np.random.RandomState(0)
    data = rng.randn(2, 4, 64)
    data_hilbert = hilbert(data, axis=-1)
    corr_orig = _compute_corrs_orig(data_hilbert)
    assert (0 <= corr_orig).all()
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

    # check against FieldTrip, which uses the square-log-norm version
    # from scipy.io import savemat
    # savemat('data.mat', dict(data_hilbert=data_hilbert))
    # matlab
    # load data
    # ft_connectivity_powcorr_ortho(reshape(data_hilbert(1,:,:), [4, 64]))
    # ft_connectivity_powcorr_ortho(reshape(data_hilbert(2,:,:), [4, 64]))
    ft_vals = np.array([
        [[np.nan, 0.196734553900236, 0.063173148355451, -0.242638384630448],
         [0.196734553900236, np.nan, 0.041799775495150, -0.088205187548542],
         [0.063173148355451, 0.041799775495150, np.nan, 0.090331428512317],
         [-0.242638384630448, -0.088205187548542, 0.090331428512317, np.nan]],
        [[np.nan, -0.013270857462890, 0.185200598081295, 0.140284351572544],
         [-0.013270857462890, np.nan, 0.150981508043722, -0.000671809276372],
         [0.185200598081295, 0.150981508043722, np.nan, 0.137460244313337],
         [0.140284351572544, -0.000671809276372, 0.137460244313337, np.nan]],
    ], float)
    ft_vals[np.isnan(ft_vals)] = 0
    corr_log = envelope_correlation(
        data, combine=None, log=True, absolute=False)
    assert_allclose(corr_log, ft_vals)
