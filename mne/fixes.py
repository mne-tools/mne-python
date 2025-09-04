"""Compatibility fixes for older versions of libraries.

If you add content to this file, please give the version of the package
at which the fix is no longer needed.

# originally copied from scikit-learn

"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# NOTE:
# Imports for SciPy submodules need to stay nested in this module
# because this module is imported many places (but not always used)!

import inspect
import operator as operator_module
import os
import warnings
from math import log

import numpy as np
from packaging.version import parse

###############################################################################
# distutils LooseVersion removed in Python 3.12


def _compare_version(version_a, operator, version_b):
    """Compare two version strings via a user-specified operator.

    Parameters
    ----------
    version_a : str
        First version string.
    operator : '==' | '>' | '<' | '>=' | '<='
        Operator to compare ``version_a`` and ``version_b`` in the form of
        ``version_a operator version_b``.
    version_b : str
        Second version string.

    Returns
    -------
    bool
        The result of the version comparison.
    """
    mapping = {"<": "lt", "<=": "le", "==": "eq", "!=": "ne", ">=": "ge", ">": "gt"}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        ver_a = parse(version_a)
        ver_b = parse(version_b)
        return getattr(operator_module, mapping[operator])(ver_a, ver_b)


###############################################################################
# Misc


def _median_complex(data, axis):
    """Compute marginal median on complex data safely.

    Can be removed when numpy introduces a fix.
    See: https://github.com/scipy/scipy/pull/12676/.
    """
    # np.median must be passed real arrays for the desired result
    if np.iscomplexobj(data):
        data = np.median(np.real(data), axis=axis) + 1j * np.median(
            np.imag(data), axis=axis
        )
    else:
        data = np.median(data, axis=axis)
    return data


def _safe_svd(A, **kwargs):
    """Get around the SVD did not converge error of death."""
    # Intel has a bug with their GESVD driver:
    #     https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/628049  # noqa: E501
    # For SciPy 0.18 and up, we can work around it by using
    # lapack_driver='gesvd' instead.
    from scipy import linalg

    if kwargs.get("overwrite_a", False):
        raise ValueError("Cannot set overwrite_a=True with this function")
    try:
        return linalg.svd(A, **kwargs)
    except np.linalg.LinAlgError as exp:
        from .utils import warn

        warn(f"SVD error ({exp}), attempting to use GESVD instead of GESDD")
        return linalg.svd(A, lapack_driver="gesvd", **kwargs)


def _csc_array_cast(x):
    from scipy.sparse import csc_array

    return csc_array(x)


# Can be replaced with sparse.eye_array once we depend on SciPy >= 1.12
def _eye_array(n, *, format="csr"):  # noqa: A002
    from scipy import sparse

    return sparse.dia_array((np.ones(n), 0), shape=(n, n)).asformat(format)


###############################################################################
# NumPy Generator (NumPy 1.17)


def rng_uniform(rng):
    """Get the uniform/randint from the rng."""
    # prefer Generator.integers, fall back to RandomState.randint
    return getattr(rng, "integers", getattr(rng, "randint", None))


###############################################################################
# Misc utilities


# get_fdata() requires knowing the dtype ahead of time, so let's triage on our
# own instead
def _get_img_fdata(img):
    data = np.asanyarray(img.dataobj)
    dtype = np.complex128 if np.iscomplexobj(data) else np.float64
    return data.astype(dtype)


###############################################################################
# Copied from sklearn to simplify code paths


def empirical_covariance(X, assume_centered=False):
    """Compute the Maximum likelihood covariance estimator.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : Boolean
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Returns
    -------
    covariance : 2D ndarray, shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


class _EstimatorMixin:
    def __sklearn_tags__(self):
        # If we get here, we should have sklearn installed
        from sklearn.utils import Tags, TargetTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
        )

    def _param_names(self):
        return inspect.getfullargspec(self.__init__).args[1:]

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._param_names():
            out[key] = getattr(self, key)
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        param_names = self._param_names()
        for key in params:
            if key in param_names:
                setattr(self, key, params[key])


class EmpiricalCovariance(_EstimatorMixin):
    """Maximum likelihood covariance estimator.

    Read more in the :ref:`User Guide <covariance>`.

    Parameters
    ----------
    store_precision : bool
        Specifies if the estimated precision is stored.

    assume_centered : bool
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)
    """

    def __init__(self, store_precision=True, assume_centered=False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def _set_covariance(self, covariance):
        """Save the covariance and precision estimates.

        Storage is done accordingly to `self.store_precision`.
        Precision stored only if invertible.

        Parameters
        ----------
        covariance : 2D ndarray, shape (n_features, n_features)
            Estimated covariance matrix to be stored, and from which precision
            is computed.
        """
        from scipy import linalg

        # covariance = check_array(covariance)
        # set covariance
        self.covariance_ = covariance
        # set precision
        if self.store_precision:
            self.precision_ = linalg.pinvh(covariance)
        else:
            self.precision_ = None

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        from scipy import linalg

        if self.store_precision:
            precision = self.precision_
        else:
            precision = linalg.pinvh(self.covariance_)
        return precision

    def fit(self, X, y=None):
        """Fit the Maximum Likelihood Estimator covariance model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : ndarray | None
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """  # noqa: E501
        # X = check_array(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(X, assume_centered=self.assume_centered)
        self._set_covariance(covariance)

        return self

    def score(self, X_test, y=None):
        """Compute the log-likelihood of a Gaussian dataset.

        Uses ``self.covariance_`` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).
        y : ndarray | None
            Not used, present for API consistency.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.
        """
        # compute empirical covariance of the test set
        test_cov = empirical_covariance(X_test - self.location_, assume_centered=True)
        # compute log likelihood
        res = log_likelihood(test_cov, self.get_precision())

        return res

    def error_norm(self, comp_cov, norm="frobenius", scaling=True, squared=True):
        """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like, shape = [n_features, n_features]
            The covariance to compare with.
        norm : str
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.
        scaling : bool
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.
        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        The Mean Squared Error (in the sense of the Frobenius norm) between
        `self` and `comp_cov` covariance estimators.
        """
        from scipy import linalg

        # compute the error
        error = comp_cov - self.covariance_
        # compute the error norm
        if norm == "frobenius":
            squared_norm = np.sum(error**2)
        elif norm == "spectral":
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented"
            )
        # optionally scale the error norm
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        # finally get either the squared norm or the norm
        if squared:
            result = squared_norm
        else:
            result = np.sqrt(squared_norm)

        return result

    def mahalanobis(self, observations):
        """Compute the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        observations : array-like, shape = [n_observations, n_features]
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        mahalanobis_distance : array, shape = [n_observations,]
            Squared Mahalanobis distances of the observations.
        """
        precision = self.get_precision()
        # compute mahalanobis distances
        centered_obs = observations - self.location_
        mahalanobis_dist = np.sum(np.dot(centered_obs, precision) * centered_obs, 1)

        return mahalanobis_dist


def log_likelihood(emp_cov, precision):
    """Compute the sample mean of the log_likelihood under a covariance model.

    computes the empirical expected log-likelihood (accounting for the
    normalization terms and scaling), allowing for universal comparison (beyond
    this software package)

    Parameters
    ----------
    emp_cov : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested

    Returns
    -------
    sample mean of the log-likelihood
    """
    p = precision.shape[0]
    log_likelihood_ = -np.sum(emp_cov * precision) + _logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_


# sklearn uses np.linalg for this, but ours is more robust to zero eigenvalues


def _logdet(A):
    """Compute the log det of a positive semidefinite matrix."""
    from scipy import linalg

    vals = linalg.eigvalsh(A)
    # avoid negative (numerical errors) or zero (semi-definite matrix) values
    tol = vals.max() * vals.size * np.finfo(np.float64).eps
    vals = np.where(vals > tol, vals, tol)
    return np.sum(np.log(vals))


def _infer_dimension_(spectrum, n_samples, n_features):
    """Infer the dimension of a dataset of shape (n_samples, n_features).

    The dataset is described by its spectrum `spectrum`.
    """
    n_spectrum = len(spectrum)
    ll = np.empty(n_spectrum)
    for rank in range(n_spectrum):
        ll[rank] = _assess_dimension_(spectrum, rank, n_samples, n_features)
    return ll.argmax()


def _assess_dimension_(spectrum, rank, n_samples, n_features):
    from scipy.special import gammaln

    if rank > len(spectrum):
        raise ValueError("The tested rank cannot exceed the rank of the dataset")

    pu = -rank * log(2.0)
    for i in range(rank):
        pu += gammaln((n_features - i) / 2.0) - log(np.pi) * (n_features - i) / 2.0

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.0

    if rank == n_features:
        pv = 0
        v = 1
    else:
        v = np.sum(spectrum[rank:]) / (n_features - rank)
        pv = -np.log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = log(2.0 * np.pi) * (m + rank + 1.0) / 2.0

    pa = 0.0
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log(
                (spectrum[i] - spectrum[j]) * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
            ) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0

    return ll


def svd_flip(u, v, u_based_decision=True):  # noqa: D103
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, np.arange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[np.arange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        warnings.warn(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum",
            RuntimeWarning,
        )
    return out


###############################################################################
# From nilearn


def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """Crop a colorbar to show from cbar_vmin to cbar_vmax.

    Used when symmetric_cbar=False is used.
    """
    if (cbar_vmin is None) and (cbar_vmax is None):
        return
    cbar_tick_locs = cbar.locator.locs
    if cbar_vmax is None:
        cbar_vmax = cbar_tick_locs.max()
    if cbar_vmin is None:
        cbar_vmin = cbar_tick_locs.min()
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax, len(cbar_tick_locs))

    cbar.ax.set_ylim(cbar_vmin, cbar_vmax)
    X = cbar._mesh()[0]
    X = np.array([X[0], X[-1]])
    Y = np.array([[cbar_vmin, cbar_vmin], [cbar_vmax, cbar_vmax]])
    N = X.shape[0]
    ii = [0, 1, N - 2, N - 1, 2 * N - 1, 2 * N - 2, N + 1, N, 0]
    x = X.T.reshape(-1)[ii]
    y = Y.T.reshape(-1)[ii]
    xy = (
        np.column_stack([y, x])
        if cbar.orientation == "horizontal"
        else np.column_stack([x, y])
    )
    cbar.outline.set_xy(xy)

    cbar.set_ticks(new_tick_locs)
    cbar.update_ticks()


###############################################################################
# Numba (optional requirement)

# Here we choose different defaults to speed things up by default
try:
    import numba

    if _compare_version(numba.__version__, "<", "0.56.4"):
        raise ImportError
    prange = numba.prange

    def jit(nopython=True, nogil=True, fastmath=True, cache=True, **kwargs):  # noqa
        return numba.jit(
            nopython=nopython, nogil=nogil, fastmath=fastmath, cache=cache, **kwargs
        )

except Exception:  # could be ImportError, SystemError, etc.
    has_numba = False
else:
    has_numba = os.getenv("MNE_USE_NUMBA", "true").lower() == "true"


if not has_numba:

    def jit(**kwargs):  # noqa
        def _jit(func):
            return func

        return _jit

    prange = range
    bincount = np.bincount

else:

    @jit()
    def bincount(x, weights, minlength):  # noqa: D103
        out = np.zeros(minlength)
        for idx, w in zip(x, weights):
            out[idx] += w
        return out


###############################################################################
# Matplotlib


# workaround: plt.close() doesn't spawn close_event on Agg backend
# https://github.com/matplotlib/matplotlib/issues/18609
def _close_event(fig):
    """Force calling of the MPL figure close event."""
    from matplotlib import backend_bases

    from .utils import logger

    try:
        fig.canvas.callbacks.process(
            "close_event",
            backend_bases.CloseEvent(name="close_event", canvas=fig.canvas),
        )
        logger.debug(f"Called {fig!r}.canvas.close_event()")
    except ValueError:  # old mpl with Qt
        logger.debug(f"Calling {fig!r}.canvas.close_event() failed")
        pass  # pragma: no cover


###############################################################################
# SciPy 1.14+ minimum_phase half=True option


def minimum_phase(h, method="homomorphic", n_fft=None, *, half=True):
    """Wrap scipy.signal.minimum_phase with half option."""
    # Can be removed once
    from scipy.fft import fft, ifft
    from scipy.signal import minimum_phase as sp_minimum_phase

    assert isinstance(method, str) and method == "homomorphic"

    if "half" in inspect.getfullargspec(sp_minimum_phase).kwonlyargs:
        return sp_minimum_phase(h, method=method, n_fft=n_fft, half=half)
    h = np.asarray(h)
    if np.iscomplexobj(h):
        raise ValueError("Complex filters not supported")
    if h.ndim != 1 or h.size <= 2:
        raise ValueError("h must be 1-D and at least 2 samples long")
    n_half = len(h) // 2
    if not np.allclose(h[-n_half:][::-1], h[:n_half]):
        warnings.warn(
            "h does not appear to by symmetric, conversion may fail",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
    n_fft = int(n_fft)
    if n_fft < len(h):
        raise ValueError(f"n_fft must be at least len(h)=={len(h)}")

    # zero-pad; calculate the DFT
    h_temp = np.abs(fft(h, n_fft))
    # take 0.25*log(|H|**2) = 0.5*log(|H|)
    h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
    np.log(h_temp, out=h_temp)
    if half:  # halving of magnitude spectrum optional
        h_temp *= 0.5
    # IDFT
    h_temp = ifft(h_temp).real
    # multiply pointwise by the homomorphic filter
    # lmin[n] = 2u[n] - d[n]
    # i.e., double the positive frequencies and zero out the negative ones;
    # Oppenheim+Shafer 3rd ed p991 eq13.42b and p1004 fig13.7
    win = np.zeros(n_fft)
    win[0] = 1
    stop = n_fft // 2
    win[1:stop] = 2
    if n_fft % 2:
        win[stop] = 1
    h_temp *= win
    h_temp = ifft(np.exp(fft(h_temp)))
    h_minimum = h_temp.real

    n_out = (n_half + len(h) % 2) if half else len(h)
    return h_minimum[:n_out]


# SciPy 1.15 deprecates sph_harm for sph_harm_y and using it will trigger a
# DeprecationWarning. This is a backport of the new function for older SciPy versions.
def sph_harm_y(n, m, theta, phi, *, diff_n=0):
    """Wrap scipy.special.sph_harm for sph_harm_y."""
    # Can be removed once we no longer support scipy < 1.15.0
    from scipy import special

    if "sph_harm_y" in special.__dict__:
        return special.sph_harm_y(n, m, theta, phi, diff_n=diff_n)
    else:
        return special.sph_harm(m, n, phi, theta)
