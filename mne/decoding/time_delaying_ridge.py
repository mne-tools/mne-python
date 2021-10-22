# -*- coding: utf-8 -*-
"""TimeDelayingRidge class."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Ross Maddox <ross.maddox@rochester.edu>
#
# License: BSD-3-Clause

import numpy as np

from .base import BaseEstimator
from ..cuda import _setup_cuda_fft_multiply_repeated
from ..filter import next_fast_len
from ..fixes import jit
from ..parallel import check_n_jobs
from ..utils import warn, ProgressBar, logger


def _compute_corrs(X, y, smin, smax, n_jobs=1, fit_intercept=False,
                   edge_correction=True):
    """Compute auto- and cross-correlations."""
    if fit_intercept:
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
    if X.ndim == 2:
        assert y.ndim == 2
        X = X[:, np.newaxis, :]
        y = y[:, np.newaxis, :]
    assert X.shape[:2] == y.shape[:2]
    len_trf = smax - smin
    len_x, n_epochs, n_ch_x = X.shape
    len_y, n_epcohs, n_ch_y = y.shape
    assert len_x == len_y

    n_fft = next_fast_len(2 * X.shape[0] - 1)

    n_jobs, cuda_dict = _setup_cuda_fft_multiply_repeated(
        n_jobs, [1.], n_fft, 'correlation calculations')

    # create our Toeplitz indexer
    ij = np.empty((len_trf, len_trf), int)
    for ii in range(len_trf):
        ij[ii, ii:] = np.arange(len_trf - ii)
        x = np.arange(n_fft - 1, n_fft - len_trf + ii, -1)
        ij[ii + 1:, ii] = x

    x_xt = np.zeros([n_ch_x * len_trf] * 2)
    x_y = np.zeros((len_trf, n_ch_x, n_ch_y), order='F')
    n = n_epochs * (n_ch_x * (n_ch_x + 1) // 2 + n_ch_x)
    logger.info('Fitting %d epochs, %d channels' % (n_epochs, n_ch_x))
    pb = ProgressBar(n, mesg='Sample')
    count = 0
    pb.update(count)
    for ei in range(n_epochs):
        this_X = X[:, ei, :]
        # XXX maybe this is what we should parallelize over CPUs at some point
        X_fft = cuda_dict['rfft'](this_X, n=n_fft, axis=0)
        X_fft_conj = X_fft.conj()
        y_fft = cuda_dict['rfft'](y[:, ei, :], n=n_fft, axis=0)

        for ch0 in range(n_ch_x):
            for oi, ch1 in enumerate(range(ch0, n_ch_x)):
                this_result = cuda_dict['irfft'](
                    X_fft[:, ch0] * X_fft_conj[:, ch1], n=n_fft, axis=0)
                # Our autocorrelation structure is a Toeplitz matrix, but
                # it's faster to create the Toeplitz ourselves than use
                # linalg.toeplitz.
                this_result = this_result[ij]
                # However, we need to adjust for coeffs that are cut off,
                # i.e. the non-zero delays should not have the same AC value
                # as the zero-delay ones (because they actually have fewer
                # coefficients).
                #
                # These adjustments also follow a Toeplitz structure, so we
                # construct a matrix of what has been left off, compute their
                # inner products, and remove them.
                if edge_correction:
                    _edge_correct(this_result, this_X, smax, smin, ch0, ch1)

                # Store the results in our output matrix
                x_xt[ch0 * len_trf:(ch0 + 1) * len_trf,
                     ch1 * len_trf:(ch1 + 1) * len_trf] += this_result
                if ch0 != ch1:
                    x_xt[ch1 * len_trf:(ch1 + 1) * len_trf,
                         ch0 * len_trf:(ch0 + 1) * len_trf] += this_result.T
                count += 1
                pb.update(count)

            # compute the crosscorrelations
            cc_temp = cuda_dict['irfft'](
                y_fft * X_fft_conj[:, slice(ch0, ch0 + 1)], n=n_fft, axis=0)
            if smin < 0 and smax >= 0:
                x_y[:-smin, ch0] += cc_temp[smin:]
                x_y[len_trf - smax:, ch0] += cc_temp[:smax]
            else:
                x_y[:, ch0] += cc_temp[smin:smax]
            count += 1
            pb.update(count)

    x_y = np.reshape(x_y, (n_ch_x * len_trf, n_ch_y), order='F')
    return x_xt, x_y, n_ch_x, X_offset, y_offset


@jit()
def _edge_correct(this_result, this_X, smax, smin, ch0, ch1):
    if smax > 0:
        tail = _toeplitz_dot(this_X[-1:-smax:-1, ch0],
                             this_X[-1:-smax:-1, ch1])
        if smin > 0:
            tail = tail[smin - 1:, smin - 1:]
        this_result[max(-smin + 1, 0):, max(-smin + 1, 0):] -= tail
    if smin < 0:
        head = _toeplitz_dot(this_X[:-smin, ch0],
                             this_X[:-smin, ch1])[::-1, ::-1]
        if smax < 0:
            head = head[:smax, :smax]
        this_result[:-smin, :-smin] -= head


@jit()
def _toeplitz_dot(a, b):
    """Create upper triangular Toeplitz matrices & compute the dot product."""
    # This is equivalent to:
    # a = linalg.toeplitz(a)
    # b = linalg.toeplitz(b)
    # a[np.triu_indices(len(a), 1)] = 0
    # b[np.triu_indices(len(a), 1)] = 0
    # out = np.dot(a.T, b)
    assert a.shape == b.shape and a.ndim == 1
    out = np.outer(a, b)
    for ii in range(1, len(a)):
        out[ii, ii:] += out[ii - 1, ii - 1:-1]
        out[ii + 1:, ii] += out[ii:-1, ii - 1]
    return out


def _compute_reg_neighbors(n_ch_x, n_delays, reg_type, method='direct',
                           normed=False):
    """Compute regularization parameter from neighbors."""
    from scipy import linalg
    from scipy.sparse.csgraph import laplacian
    known_types = ('ridge', 'laplacian')
    if isinstance(reg_type, str):
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
    from scipy import linalg
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
        warn('Singular matrix in solving dual problem. Using '
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
    n_jobs : int | str
        The number of jobs to use. Can be an int (default 1) or ``'cuda'``.

        .. versionadded:: 0.18
    edge_correction : bool
        If True (default), correct the autocorrelation coefficients for
        non-zero delays for the fact that fewer samples are available.
        Disabling this speeds up performance at the cost of accuracy
        depending on the relationship between epoch length and model
        duration. Only used if ``estimator`` is float or None.

        .. versionadded:: 0.18

    See Also
    --------
    mne.decoding.ReceptiveField

    Notes
    -----
    This class is meant to be used with :class:`mne.decoding.ReceptiveField`
    by only implicitly doing the time delaying. For reasonable receptive
    field and input signal sizes, it should be more CPU and memory
    efficient by using frequency-domain methods (FFTs) to compute the
    auto- and cross-correlations.
    """

    _estimator_type = "regressor"

    def __init__(self, tmin, tmax, sfreq, alpha=0., reg_type='ridge',
                 fit_intercept=True, n_jobs=1, edge_correction=True):
        if tmin > tmax:
            raise ValueError('tmin must be <= tmax, got %s and %s'
                             % (tmin, tmax))
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = float(sfreq)
        self.alpha = float(alpha)
        self.reg_type = reg_type
        self.fit_intercept = fit_intercept
        self.edge_correction = edge_correction
        self.n_jobs = n_jobs

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
        n_jobs = check_n_jobs(self.n_jobs, allow_cuda=True)
        # These are split into two functions because it's possible that we
        # might want to allow people to do them separately (e.g., to test
        # different regularization parameters).
        self.cov_, x_y_, n_ch_x, X_offset, y_offset = _compute_corrs(
            X, y, self._smin, self._smax, n_jobs, self.fit_intercept,
            self.edge_correction)
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
        from scipy.signal import fftconvolve

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
                    temp = fftconvolve(X[:, ei, fi], self.coef_[oi, fi])
                    temp = temp[max(-smin, 0):][:len(out) - offset]
                    out[offset:len(temp) + offset, ei, oi] += temp
        out += self.intercept_
        if singleton:
            out = out[:, 0, :]
        return out
