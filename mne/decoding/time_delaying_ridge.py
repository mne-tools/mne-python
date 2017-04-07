# -*- coding: utf-8 -*-
"""TimeDelayingRidge class."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Ross Maddox <ross.maddox@rochester.edu>
#
# License: BSD (3-clause)

import warnings

import numpy as np

from .base import BaseEstimator
from ..filter import next_fast_len
from ..externals.six import string_types


def _compute_corrs(X, y, smin, smax, fit_intercept):
    """Compute the auto- and cross-correlations."""
    len_trf = smax - smin
    len_x, n_ch_x = X.shape
    len_y, n_ch_y = y.shape
    assert len_x == len_y

    if fit_intercept:
        # We could do this in the Fourier domain, too, but it should
        # be a bit cleaner numerically do do it here.
        X = X - np.mean(X, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)
    n_fft = next_fast_len(X.shape[0] + max(smax, 0) - min(smin, 0) - 1)
    X_fft = np.fft.rfft(X.T, n_fft)
    y_fft = np.fft.rfft(y.T, n_fft)
    # del X, y

    # compute the autocorrelations
    ac = np.zeros((n_ch_x, n_ch_x, len_trf * 2 - 1))
    for ch0 in range(n_ch_x):
        for ch1 in range(ch0, n_ch_x):
            # This is equivalent to:
            # ac_temp = np.correlate(X[:, ch0], X[:, ch1], mode='full')
            # ac_temp = np.roll(ac_temp_2, X.shape[0])
            ac_temp = np.fft.irfft(X_fft[ch0] * X_fft[ch1].conj(), n_fft)
            ac[ch0, ch1] = np.concatenate((ac_temp[-len_trf + 1:],
                                           ac_temp[:len_trf]))
            if ch0 != ch1:
                ac[ch1, ch0] = ac[ch0, ch1][::-1]

    # compute the crosscorrelations
    x_y = np.zeros((n_ch_y, n_ch_x, len_trf))
    for ch_in in range(n_ch_x):
        for ch_out in range(n_ch_y):
            cc_temp = np.fft.irfft(y_fft[ch_out] * X_fft[ch_in].conj(), n_fft)
            if smin < 0 and smax >= 0:
                x_y[ch_out, ch_in] = np.append(cc_temp[smin:], cc_temp[:smax])
            else:
                x_y[ch_out, ch_in] = cc_temp[smin:smax]

    # make xxt and xy
    x_xt = _make_x_xt(ac)
    x_y.shape = (n_ch_y, n_ch_x * len_trf)
    return x_xt, x_y, n_ch_x


def _fit_corrs(x_xt, x_y, n_ch_x, reg_type, alpha, n_ch_in):
    """Fit the model using correlation matrices."""
    from scipy import linalg
    known_types = ('ridge', 'quadratic')
    if isinstance(reg_type, string_types):
        reg_type = (reg_type,) * 2
    if len(reg_type) != 2:
        raise ValueError('reg_type must have two elements, got %s'
                         % (len(reg_type),))
    for r in reg_type:
        if r not in known_types:
            raise ValueError('reg_type entries must be one of %s, got %s'
                             % (known_types, r))

    # do the regularized solving
    n_ch_out = x_y.shape[0]
    assert x_y.shape[1] % n_ch_x == 0
    n_trf = x_y.shape[1] // n_ch_x

    # regularize time
    reg = np.eye(n_trf)
    if reg_type[0] == 'quadratic':
        reg.flat[1::reg.shape[0] + 1] += -1
        reg.flat[reg.shape[0] + 1:-reg.shape[0] - 1:reg.shape[0] + 1] += 1
        reg.flat[reg.shape[0]::reg.shape[0] + 1] += -1
    args = [reg] * n_ch_x
    reg = linalg.block_diag(*args)

    # regularize features
    if reg_type[1] == 'quadratic':
        row_offset = n_trf * n_trf * n_ch_x
        reg.flat[n_trf::n_trf * n_ch_x + 1] += -1
        reg.flat[row_offset + n_trf:-row_offset:n_trf * n_ch_x + 1] += 1
        reg.flat[n_trf * n_trf * n_ch_x::n_trf * n_ch_x + 1] += -1

    mat = x_xt + alpha * reg
    # From sklearn
    try:
        # Note: we must use overwrite_a=False in order to be able to
        #       use the fall-back solution below in case a LinAlgError
        #       is raised
        w = linalg.solve(mat, x_y.T, sym_pos=True, overwrite_a=False)
    except np.linalg.LinAlgError:
        warnings.warn('Singular matrix in solving dual problem. Using '
                      'least-squares solution instead.')
        w = linalg.lstsq(mat, x_y.T, lapack_driver='gelsy')[0]
    w = w.T.reshape([n_ch_out, n_ch_in, n_trf])
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
    tmin : int | float
        The starting lag, in seconds (or samples if ``sfreq`` == 1).
        Negative values correspond to times in the past.
    tmax : int | float
        The ending lag, in seconds (or samples if ``sfreq`` == 1).
        Positive values correspond to times in the future.
        Must be >= tmin.
    sfreq : float
        The sampling frequency used to convert times into samples.
    alpha : float
        The ridge (or quadratic) regularization factor.
    reg_type : str | list
        Can be "ridge" (default) or "quadratic".
        Can also be a 2-element list specifying how to regularize in time
        and across adjacent features.
    fit_intercept : bool
        If True (default), the sample mean is removed before fitting.

    Notes
    -----
    This class is meant to be used with :class:`mne.decoding.ReceptiveField`
    by only implicitly doing the time delaying. For reasonable receptive
    field and input signal sizes, it should be more CPU and memory
    efficient by using frequency-domain methods (FFTs) to compute the
    auto- and cross-correlations.

    See Also
    --------
    mne.decoding.ReceptiveField
    """

    def __init__(self, tmin, tmax, sfreq, alpha=0., reg_type='ridge',
                 fit_intercept=True):  # noqa: D102
        if tmin > tmax:
            raise ValueError('tmin must be <= tmax, got %s and %s'
                             % (tmin, tmax))
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = float(sfreq)
        self.alpha = float(alpha)
        self.reg_type = reg_type
        self.fit_intercept = fit_intercept
        self._estimator_type = "regressor"

    @property
    def _smin(self):
        return int(round(-self.tmax * self.sfreq))

    @property
    def _smax(self):
        return int(round(-self.tmin * self.sfreq)) + 1

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
        self : instance of TimeDelayingRidge
            Returns the modified instance.
        """
        # These are split into two functions because it's possible that we
        # might want to allow people to do them separately (e.g., to test
        # different regularization parameters).
        x_xt, x_y, n_ch_x = _compute_corrs(X, y, self._smin, self._smax,
                                           self.fit_intercept)
        self.coef_ = _fit_corrs(x_xt, x_y, n_ch_x,
                                self.reg_type, self.alpha, n_ch_x)
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
        smin = self._smin
        offset = max(smin, 0)
        for oi in range(self.coef_.shape[0]):
            for fi in range(self.coef_.shape[1]):
                temp = np.convolve(X[:, fi], self.coef_[oi, fi][::-1])
                temp = temp[max(-smin, 0):][:len(out) - offset]
                out[offset:len(temp) + offset, oi] += temp
        return out
