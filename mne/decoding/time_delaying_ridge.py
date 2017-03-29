# -*- coding: utf-8 -*-
"""TimeDelayingRidge class."""
# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .base import BaseEstimator
from ..filter import next_fast_len


def _compute_corrs(X, y, smin, smax):
    """Compute the auto- and cross-correlations."""
    len_trf = smax - smin
    len_x, n_ch_x = X.shape
    len_y, n_ch_y = y.shape

    n_fft = next_fast_len(max(X.shape[0], y.shape[0]) + len_trf - 1)
    X_in = np.fft.rfft(X.T, n_fft)
    X_out = np.fft.rfft(y.T, n_fft)
    del X, y

    # compute the autocorrelations
    ac = np.zeros((n_ch_x, n_ch_x, len_trf * 2 - 1))
    for ch0 in range(n_ch_x):
        for ch1 in range(ch0, n_ch_x):
            ac_temp = np.fft.irfft(X_in[ch0] * np.conj(X_in[ch1]), n_fft)
            ac[ch0, ch1] = np.concatenate((ac_temp[-len_trf + 1:],
                                           ac_temp[:len_trf]))
            if ch0 != ch1:
                ac[ch1, ch0] = ac[ch0, ch1][::-1]

    # compute the crosscorrelations
    x_y = np.zeros((n_ch_y, n_ch_x, len_trf))
    for ch_in in range(n_ch_x):
        for ch_out in range(n_ch_y):
            cc_temp = np.fft.irfft(X_out[ch_out] * X_in[ch_in].conj(),
                                   n_fft)
            # XXX Need to ensure smax > 0
            if smin < 0:
                x_y[ch_out, ch_in] = np.append(cc_temp[smin:], cc_temp[:smax])
            else:
                x_y[ch_out, ch_in] = cc_temp[smin:smax]

    # make xxt and xy
    x_xt = _make_x_xt(ac)
    x_xt /= len_x
    x_y.shape = (n_ch_y, n_ch_x * len_trf)
    x_y /= len_x
    return x_xt, x_y, n_ch_x


def _fit_corrs(x_xt, x_y, n_ch_x, reg_type, alpha, n_ch_in):
    """Fit the model using correlation matrices."""
    from scipy import linalg
    known_types = ('ridge', 'laplacian')
    if reg_type not in known_types:
        raise ValueError('reg_type must be one of %s, got %s'
                         % (known_types, reg_type))

    # do the regularized solving
    n_ch_out = x_y.shape[0]
    assert x_y.shape[1] % n_ch_x == 0
    n_trf = x_y.shape[1] // n_ch_x
    if reg_type == 'ridge':
        reg = np.eye(x_xt.shape[0])
    else:  # if reg_type == 'laplacian':
        reg = np.eye(n_trf)
        reg.flat[reg.shape[0] + 1:-reg.shape[0] - 1:reg.shape[0] + 1] = 2
        reg.flat[1::reg.shape[0] + 1] = -1
        reg.flat[reg.shape[0]::reg.shape[0] + 1] = -1
        args = [reg] * n_ch_x
        reg = linalg.block_diag(*args)
    mat = x_xt + alpha * reg
    w = linalg.lstsq(mat, x_y.T)[0].T
    w = w.reshape([n_ch_out, n_ch_in, n_trf])
    return w


def _make_x_xt(ac):
    len_trf = (ac.shape[2] + 1) // 2
    n_ch = ac.shape[0]
    xxt = np.zeros([n_ch * len_trf] * 2)
    for ch0 in range(n_ch):
        for ch1 in range(n_ch):
            xxt_temp = np.zeros((len_trf, len_trf))
            xxt_temp[0, :] = ac[ch0, ch1, len_trf - 1:]
            xxt_temp[:, 0] = ac[ch0, ch1, len_trf - 1::-1]
            for ii in range(1, len_trf):
                xxt_temp[ii, ii:] = ac[ch0, ch1, len_trf - 1:-ii]
                xxt_temp[ii:, ii] = ac[ch0, ch1, len_trf - 1:ii - 1:-1]
            xxt[ch0 * len_trf:(ch0 + 1) * len_trf,
                ch1 * len_trf:(ch1 + 1) * len_trf] = xxt_temp
    return xxt


class TimeDelayingRidge(BaseEstimator):
    """Ridge regression of data with time delays.

    Parameters
    ----------
    smin : int
        Minimum sample number (must be < 0).
    smax : float
        Maximum time used to predict the data.
    alpha : float
        The ridge (or laplacian) regularization.
    reg_type : str
        Can be "ridge" (default) or "laplacian".
    """
    def __init__(self, smin, smax, alpha=0., reg_type='ridge'):
        assert smin < 0
        assert smax >= 0
        self._smin = -smax
        self._smax = -smin + 1
        self._alpha = float(alpha)
        self._reg_type = reg_type

    def fit(self, X, y):
        """Estimate the coefficients of the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples, n_outputs)
            The target values.

        Returns
        -------
        self : instance of LinearModel
            Returns the modified instance.
        """
        # These are split into two functions because it's possible that we
        # might want to allow people to do them separately (e.g., to test
        # different regularization parameters).
        x_xt, x_y, n_ch_x = _compute_corrs(X, y, self._smin, self._smax)
        self.coef_ = _fit_corrs(x_xt, x_y, n_ch_x,
                                self._reg_type, self._alpha, n_ch_x)
        self.coef_ = self.coef_[..., ::-1]
        return self

    def predict(self, X):
        """Predict the output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data.

        Returns
        -------
        X : ndarray
            The predicted response.
        """
        out = np.zeros((X.shape[0], self.coef_.shape[0]))
        for oi in range(self.coef_.shape[0]):
            for fi in range(self.coef_.shape[1]):
                # XXX This truncation is not correct... need to shift based
                # on tmin/tmax locations
                temp = np.convolve(X[:, fi], self.coef_[oi, fi])
                out[:, oi] += temp[-self._smin:-self._smax + 1]
        return out
