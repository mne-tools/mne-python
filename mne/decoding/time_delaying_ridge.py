# -*- coding: utf-8 -*-
"""TimeDelayingRidge class."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Ross Maddox <ross.maddox@rochester.edu>
#
# License: BSD (3-clause)

import warnings

import numpy as np
from scipy import linalg

from .base import BaseEstimator
from ..filter import next_fast_len
from ..externals.six import string_types


def _compute_corrs(X, y, smin, smax):
    """Compute auto- and cross-correlations."""
    if X.ndim == 2:
        assert y.ndim == 2
        X = X[:, np.newaxis, :]
        y = y[:, np.newaxis, :]
    assert X.shape[:2] == y.shape[:2]
    len_trf = smax - smin
    len_x, n_epochs, n_ch_x = X.shape
    len_y, n_epcohs, n_ch_y = y.shape
    assert len_x == len_y

    n_fft = next_fast_len(X.shape[0] + max(smax, 0) - min(smin, 0) - 1)

    x_xt = np.zeros([n_ch_x * len_trf] * 2)
    x_y = np.zeros((len_trf, n_ch_x, n_ch_y), order='F')
    for ei in range(n_epochs):
        this_X = X[:, ei, :]
        X_fft = np.fft.rfft(this_X, n_fft, axis=0)
        y_fft = np.fft.rfft(y[:, ei, :], n_fft, axis=0)

        # compute the autocorrelations
        for ch0 in range(n_ch_x):
            other_sl = slice(ch0, n_ch_x)
            ac_temp = np.fft.irfft(X_fft[:, ch0][:, np.newaxis] *
                                   X_fft[:, other_sl].conj(), n_fft, axis=0)
            n_other = ac_temp.shape[1]
            row = ac_temp[:len_trf]  # zero and positive lags
            col = ac_temp[-1:-len_trf:-1]  # negative lags
            # Our autocorrelation structure is a Toeplitz matrix, but
            # it's faster to create the Toeplitz ourselves.
            x_xt_temp = np.zeros((len_trf, len_trf, n_other))
            for ii in range(len_trf):
                x_xt_temp[ii, ii:] = row[:len_trf - ii]
                x_xt_temp[ii + 1:, ii] = col[:len_trf - ii - 1]
            row_adjust = np.zeros((len_trf, n_other))
            col_adjust = np.zeros((len_trf, n_other))

            # However, we need to adjust for coeffs that are cut off by
            # the mode="same"-like behavior of the algorithm,
            # i.e. the non-zero delays should not have the same AC value
            # as the zero-delay ones (because they actually have fewer
            # coefficents).
            #
            # These adjustments also follow a Toeplitz structure, but it's
            # computationally more efficient to manually accumulate and
            # subtract from each row and col, rather than accumulate a single
            # adjustment matrix using Toeplitz repetitions then subtract

            # Adjust positive lags where the tail gets cut off
            for idx in range(1, smax):
                ii = idx - smin
                end_sl = slice(X.shape[0] - idx, -smax - min(ii, 0), -1)
                c = (this_X[-idx, other_sl][np.newaxis] *
                     this_X[end_sl, ch0][:, np.newaxis])
                r = this_X[-idx, ch0] * this_X[end_sl, other_sl]
                if ii <= 0:
                    col_adjust += c
                    row_adjust += r
                    if ii == 0:
                        x_xt_temp[0, :] = row - row_adjust
                        x_xt_temp[1:, 0] = col - col_adjust[1:]
                else:
                    col_adjust[:-ii] += c
                    row_adjust[:-ii] += r
                    x_xt_temp[ii, ii:] = row[:-ii] - row_adjust[:-ii]
                    x_xt_temp[ii + 1:, ii] = col[:-ii] - col_adjust[1:-ii]

            # Adjust negative lags where the head gets cut off
            x_xt_temp = x_xt_temp[::-1][:, ::-1]
            row_adjust.fill(0.)
            col_adjust.fill(0.)
            for idx in range(0, -smin):
                ii = idx + smax
                start_sl = slice(idx, -smin + min(ii, 0))
                c = (this_X[idx, other_sl][np.newaxis] *
                     this_X[start_sl, ch0][:, np.newaxis])
                r = this_X[idx, ch0] * this_X[start_sl, other_sl]
                if ii <= 0:
                    col_adjust += c
                    row_adjust += r
                    if ii == 0:
                        x_xt_temp[0, :] -= row_adjust
                        x_xt_temp[1:, 0] -= col_adjust[1:]
                else:
                    col_adjust[:-ii] += c
                    row_adjust[:-ii] += r
                    x_xt_temp[ii, ii:] -= row_adjust[:-ii]
                    x_xt_temp[ii + 1:, ii] -= col_adjust[1:-ii]

            x_xt_temp = x_xt_temp[::-1][:, ::-1]
            for oi in range(n_other):
                ch1 = oi + ch0
                # Store the result
                this_result = x_xt_temp[:, :, oi]
                x_xt[ch0 * len_trf:(ch0 + 1) * len_trf,
                     ch1 * len_trf:(ch1 + 1) * len_trf] += this_result
                if ch0 != ch1:
                    x_xt[ch1 * len_trf:(ch1 + 1) * len_trf,
                         ch0 * len_trf:(ch0 + 1) * len_trf] += this_result.T

            # compute the crosscorrelations
            cc_temp = np.fft.irfft(
                y_fft * X_fft[:, ch0][:, np.newaxis].conj(), n_fft, axis=0)
            if smin < 0 and smax >= 0:
                x_y[:-smin, ch0] += cc_temp[smin:]
                x_y[len_trf - smax:, ch0] += cc_temp[:smax]
            else:
                x_y[:, ch0] += cc_temp[smin:smax]

    x_y = np.reshape(x_y, (n_ch_x * len_trf, n_ch_y), order='F')
    return x_xt, x_y, n_ch_x


def _compute_reg_neighbors(n_ch_x, n_delays, reg_type, method='direct',
                           normed=False):
    """Compute regularization parameter from neighbors."""
    from scipy.sparse.csgraph import laplacian
    known_types = ('ridge', 'laplacian')
    if isinstance(reg_type, string_types):
        reg_type = (reg_type,) * 2
    if len(reg_type) != 2:
        raise ValueError('reg_type must have two elements, got %s'
                         % (len(reg_type),))
    for r in reg_type:
        if r not in known_types:
            raise ValueError('reg_type entries must be one of %s, got %s'
                             % (known_types, r))
    reg_time = (reg_type[0] == 'laplacian' and n_delays > 1)
    reg_chs = (reg_type[1] == 'laplacian' and n_ch_x > 1)
    if not reg_time and not reg_chs:
        return np.eye(n_ch_x * n_delays)
    # regularize time
    if reg_time:
        reg = np.eye(n_delays)
        stride = n_delays + 1
        reg.flat[1::stride] += -1
        reg.flat[n_delays::stride] += -1
        reg.flat[n_delays + 1:-n_delays - 1:stride] += 1
        args = [reg] * n_ch_x
        reg = linalg.block_diag(*args)
    else:
        reg = np.zeros((n_delays * n_ch_x,) * 2)

    # regularize features
    if reg_chs:
        block = n_delays * n_delays
        row_offset = block * n_ch_x
        stride = n_delays * n_ch_x + 1
        reg.flat[n_delays:-row_offset:stride] += -1
        reg.flat[n_delays + row_offset::stride] += 1
        reg.flat[row_offset:-n_delays:stride] += -1
        reg.flat[:-(n_delays + row_offset):stride] += 1
    assert np.array_equal(reg[::-1, ::-1], reg)

    if method == 'direct':
        if normed:
            norm = np.sqrt(np.diag(reg))
            reg /= norm
            reg /= norm[:, np.newaxis]
        return reg
    else:
        # Use csgraph. Note that our -1's above are really the neighbors!
        # If we ever want to allow arbitrary adjacency matrices, this is how
        # we'd want to do it.
        reg = laplacian(-reg, normed=normed)
    return reg


def _fit_corrs(x_xt, x_y, n_ch_x, reg_type, alpha, n_ch_in):
    """Fit the model using correlation matrices."""
    # do the regularized solving
    n_ch_out = x_y.shape[1]
    assert x_y.shape[0] % n_ch_x == 0
    n_delays = x_y.shape[0] // n_ch_x
    reg = _compute_reg_neighbors(n_ch_x, n_delays, reg_type)
    mat = x_xt + alpha * reg
    # From sklearn
    try:
        # Note: we must use overwrite_a=False in order to be able to
        #       use the fall-back solution below in case a LinAlgError
        #       is raised
        w = linalg.solve(mat, x_y, sym_pos=True, overwrite_a=False)
    except np.linalg.LinAlgError:
        warnings.warn('Singular matrix in solving dual problem. Using '
                      'least-squares solution instead.')
        w = linalg.lstsq(mat, x_y, lapack_driver='gelsy')[0]
    w = w.T.reshape([n_ch_out, n_ch_in, n_delays])
    return w


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
        The ridge (or laplacian) regularization factor.
    reg_type : str | list
        Can be "ridge" (default) or "laplacian".
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

    _estimator_type = "regressor"

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

    @property
    def _smin(self):
        return int(round(self.tmin * self.sfreq))

    @property
    def _smax(self):
        return int(round(self.tmax * self.sfreq)) + 1

    def fit(self, X, y):
        """Estimate the coefficients of the linear model.

        Parameters
        ----------
        X : array, shape (n_samples[, n_epochs], n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples[, n_epochs],  n_outputs)
            The target values.

        Returns
        -------
        self : instance of TimeDelayingRidge
            Returns the modified instance.
        """
        if X.ndim == 3:
            assert y.ndim == 3
            assert X.shape[:2] == y.shape[:2]
        else:
            assert X.ndim == 2 and y.ndim == 2
            assert X.shape[0] == y.shape[0]
        # These are split into two functions because it's possible that we
        # might want to allow people to do them separately (e.g., to test
        # different regularization parameters).
        if self.fit_intercept:
            # We could do this in the Fourier domain, too, but it should
            # be a bit cleaner numerically to do it here.
            X_offset = np.mean(X, axis=0)
            y_offset = np.mean(y, axis=0)
            if X.ndim == 3:
                X_offset = X_offset.mean(axis=0)
                y_offset = np.mean(y_offset, axis=0)
            X = X - X_offset
            y = y - y_offset
        else:
            X_offset = y_offset = 0.
        self.cov_, x_y_, n_ch_x = _compute_corrs(X, y, self._smin, self._smax)
        self.coef_ = _fit_corrs(self.cov_, x_y_, n_ch_x,
                                self.reg_type, self.alpha, n_ch_x)
        # This is the sklearn formula from LinearModel (will be 0. for no fit)
        if self.fit_intercept:
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.sum(-1).T)
        else:
            self.intercept_ = 0.
        return self

    def predict(self, X):
        """Predict the output.

        Parameters
        ----------
        X : array, shape (n_samples[, n_epochs], n_features)
            The data.

        Returns
        -------
        X : ndarray
            The predicted response.
        """
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
            singleton = True
        else:
            singleton = False
        out = np.zeros(X.shape[:2] + (self.coef_.shape[0],))
        smin = self._smin
        offset = max(smin, 0)
        for ei in range(X.shape[1]):
            for oi in range(self.coef_.shape[0]):
                for fi in range(self.coef_.shape[1]):
                    temp = np.convolve(X[:, ei, fi], self.coef_[oi, fi])
                    temp = temp[max(-smin, 0):][:len(out) - offset]
                    out[offset:len(temp) + offset, ei, oi] += temp
        out += self.intercept_
        if singleton:
            out = out[:, 0, :]
        return out
