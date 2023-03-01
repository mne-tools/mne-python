"""Compatibility fixes for older versions of libraries

If you add content to this file, please give the version of the package
at which the fix is no longer needed.

# originally copied from scikit-learn

"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD

import inspect
from math import log
from pprint import pprint
from io import StringIO
import os
import warnings

import numpy as np


###############################################################################
# distutils

# distutils has been deprecated since Python 3.10 and is scheduled for removal
# from the standard library with the release of Python 3.12. For version
# comparisons, we use setuptools's `parse_version` if available.

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
    from packaging.version import parse
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('ignore')
        return eval(f'parse("{version_a}") {operator} parse("{version_b}")')


###############################################################################
# Misc

def _median_complex(data, axis):
    """Compute marginal median on complex data safely.

    Can be removed when numpy introduces a fix.
    See: https://github.com/scipy/scipy/pull/12676/.
    """
    # np.median must be passed real arrays for the desired result
    if np.iscomplexobj(data):
        data = (np.median(np.real(data), axis=axis)
                + 1j * np.median(np.imag(data), axis=axis))
    else:
        data = np.median(data, axis=axis)
    return data


def _safe_svd(A, **kwargs):
    """Wrapper to get around the SVD did not converge error of death"""
    # Intel has a bug with their GESVD driver:
    #     https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/628049  # noqa: E501
    # For SciPy 0.18 and up, we can work around it by using
    # lapack_driver='gesvd' instead.
    from scipy import linalg
    if kwargs.get('overwrite_a', False):
        raise ValueError('Cannot set overwrite_a=True with this function')
    try:
        return linalg.svd(A, **kwargs)
    except np.linalg.LinAlgError as exp:
        from .utils import warn
        warn('SVD error (%s), attempting to use GESVD instead of GESDD'
                % (exp,))
        return linalg.svd(A, lapack_driver='gesvd', **kwargs)


def _csc_matrix_cast(x):
    from scipy.sparse import csc_matrix
    return csc_matrix(x)


###############################################################################
# Backporting nibabel's read_geometry

def _get_read_geometry():
    """Get the geometry reading function."""
    try:
        import nibabel as nib
        has_nibabel = True
    except ImportError:
        has_nibabel = False
    if has_nibabel:
        from nibabel.freesurfer import read_geometry
    else:
        read_geometry = _read_geometry
    return read_geometry


def _read_geometry(filepath, read_metadata=False, read_stamp=False):
    """Backport from nibabel."""
    from .surface import _fread3, _fread3_many
    volume_info = dict()

    TRIANGLE_MAGIC = 16777214
    QUAD_MAGIC = 16777215
    NEW_QUAD_MAGIC = 16777213
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic in (QUAD_MAGIC, NEW_QUAD_MAGIC):  # Quad file
            nvert = _fread3(fobj)
            nquad = _fread3(fobj)
            (fmt, div) = (">i2", 100.) if magic == QUAD_MAGIC else (">f4", 1.)
            coords = np.fromfile(fobj, fmt, nvert * 3).astype(np.float64) / div
            coords = coords.reshape(-1, 3)
            quads = _fread3_many(fobj, nquad * 4)
            quads = quads.reshape(nquad, 4)
            #
            #   Face splitting follows
            #
            faces = np.zeros((2 * nquad, 3), dtype=np.int64)
            nface = 0
            for quad in quads:
                if (quad[0] % 2) == 0:
                    faces[nface] = quad[0], quad[1], quad[3]
                    nface += 1
                    faces[nface] = quad[2], quad[3], quad[1]
                    nface += 1
                else:
                    faces[nface] = quad[0], quad[1], quad[2]
                    nface += 1
                    faces[nface] = quad[0], quad[2], quad[3]
                    nface += 1

        elif magic == TRIANGLE_MAGIC:  # Triangle file
            create_stamp = fobj.readline().rstrip(b'\n').decode('utf-8')
            fobj.readline()
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)

            if read_metadata:
                volume_info = _read_volume_info(fobj)
        else:
            raise ValueError("File does not appear to be a Freesurfer surface")

    coords = coords.astype(np.float64)

    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn('No volume information contained in the file')
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)

    return ret


###############################################################################
# NumPy Generator (NumPy 1.17)


def rng_uniform(rng):
    """Get the unform/randint from the rng."""
    # prefer Generator.integers, fall back to RandomState.randint
    return getattr(rng, 'integers', getattr(rng, 'randint', None))


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


###############################################################################
# Misc utilities

# get_fdata() requires knowing the dtype ahead of time, so let's triage on our
# own instead
def _get_img_fdata(img):
    data = np.asanyarray(img.dataobj)
    dtype = np.complex128 if np.iscomplexobj(data) else np.float64
    return data.astype(dtype)


def _read_volume_info(fobj):
    """An implementation of nibabel.freesurfer.io._read_volume_info, since old
    versions of nibabel (<=2.1.0) don't have it.
    """
    volume_info = dict()
    head = np.fromfile(fobj, '>i4', 1)
    if not np.array_equal(head, [20]):  # Read two bytes more
        head = np.concatenate([head, np.fromfile(fobj, '>i4', 2)])
        if not np.array_equal(head, [2, 0, 20]):
            warnings.warn("Unknown extension code.")
            return volume_info

    volume_info['head'] = head
    for key in ['valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras',
                'zras', 'cras']:
        pair = fobj.readline().decode('utf-8').split('=')
        if pair[0].strip() != key or len(pair) != 2:
            raise IOError('Error parsing volume info.')
        if key in ('valid', 'filename'):
            volume_info[key] = pair[1].strip()
        elif key == 'volume':
            volume_info[key] = np.array(pair[1].split()).astype(int)
        else:
            volume_info[key] = np.array(pair[1].split()).astype(float)
    # Ignore the rest
    return volume_info


def _serialize_volume_info(volume_info):
    """An implementation of nibabel.freesurfer.io._serialize_volume_info, since
    old versions of nibabel (<=2.1.0) don't have it."""
    keys = ['head', 'valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras',
            'zras', 'cras']
    diff = set(volume_info.keys()).difference(keys)
    if len(diff) > 0:
        raise ValueError('Invalid volume info: %s.' % diff.pop())

    strings = list()
    for key in keys:
        if key == 'head':
            if not (np.array_equal(volume_info[key], [20]) or np.array_equal(
                    volume_info[key], [2, 0, 20])):
                warnings.warn("Unknown extension code.")
            strings.append(np.array(volume_info[key], dtype='>i4').tobytes())
        elif key in ('valid', 'filename'):
            val = volume_info[key]
            strings.append('{} = {}\n'.format(key, val).encode('utf-8'))
        elif key == 'volume':
            val = volume_info[key]
            strings.append('{} = {} {} {}\n'.format(
                key, val[0], val[1], val[2]).encode('utf-8'))
        else:
            val = volume_info[key]
            strings.append('{} = {:0.10g} {:0.10g} {:0.10g}\n'.format(
                key.ljust(6), val[0], val[1], val[2]).encode('utf-8'))
    return b''.join(strings)


##############################################################################
# adapted from scikit-learn


def is_classifier(estimator):
    """Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"


_DEFAULT_TAGS = {
    'non_deterministic': False,
    'requires_positive_X': False,
    'requires_positive_y': False,
    'X_types': ['2darray'],
    'poor_score': False,
    'no_validation': False,
    'multioutput': False,
    "allow_nan": False,
    'stateless': False,
    'multilabel': False,
    '_skip_test': False,
    '_xfail_checks': False,
    'multioutput_only': False,
    'binary_only': False,
    'requires_fit': True,
    'preserves_dtype': [np.float64],
    'requires_y': False,
    'pairwise': False,
}


class BaseEstimator(object):
    """Base class for all estimators in scikit-learn.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
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
            Parameters.

        Returns
        -------
        inst : instance
            The object.
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        params = StringIO()
        pprint(self.get_params(deep=False), params)
        params.seek(0)
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, params.read().strip())

    # __getstate__ and __setstate__ are omitted because they only contain
    # conditionals that are not satisfied by our objects (e.g.,
    # ``if type(self).__module__.startswith('sklearn.')``.

    def _more_tags(self):
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags


# newer sklearn deprecates importing from sklearn.metrics.scoring,
# but older sklearn does not expose check_scoring in sklearn.metrics.
def _get_check_scoring():
    try:
        from sklearn.metrics import check_scoring  # noqa
    except ImportError:
        from sklearn.metrics.scorer import check_scoring  # noqa
    return check_scoring


def _check_fit_params(X, fit_params, indices=None):
    """Check and validate the parameters passed during `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    fit_params : dict
        Dictionary containing the parameters passed at fit.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as
        `X`.

    Returns
    -------
    fit_params_validated : dict
        Validated parameters. We ensure that the values support
        indexing.
    """
    try:
        from sklearn.utils.validation import \
            _check_fit_params as _sklearn_check_fit_params
        return _sklearn_check_fit_params(X, fit_params, indices)
    except ImportError:
        from sklearn.model_selection import _validation

        fit_params_validated = \
            {k: _validation._index_param_value(X, v, indices)
             for k, v in fit_params.items()}
        return fit_params_validated


###############################################################################
# Copied from sklearn to simplify code paths

def empirical_covariance(X, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator


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
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


class EmpiricalCovariance(BaseEstimator):
    """Maximum likelihood covariance estimator

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
        """Saves the covariance and precision estimates

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
        covariance = empirical_covariance(
            X, assume_centered=self.assume_centered)
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
        test_cov = empirical_covariance(
            X_test - self.location_, assume_centered=True)
        # compute log likelihood
        res = log_likelihood(test_cov, self.get_precision())

        return res

    def error_norm(self, comp_cov, norm='frobenius', scaling=True,
                   squared=True):
        """Computes the Mean Squared Error between two covariance estimators.

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
            squared_norm = np.sum(error ** 2)
        elif norm == "spectral":
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented")
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
        """Computes the squared Mahalanobis distances of given observations.

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
        mahalanobis_dist = np.sum(
            np.dot(centered_obs, precision) * centered_obs, 1)

        return mahalanobis_dist


def log_likelihood(emp_cov, precision):
    """Computes the sample mean of the log_likelihood under a covariance model

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
    log_likelihood_ = - np.sum(emp_cov * precision) + _logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.
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
    """Infers the dimension of a dataset of shape (n_samples, n_features)
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
        raise ValueError("The tested rank cannot exceed the rank of the"
                         " dataset")

    pu = -rank * log(2.)
    for i in range(rank):
        pu += (gammaln((n_features - i) / 2.) -
               log(np.pi) * (n_features - i) / 2.)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    if rank == n_features:
        pv = 0
        v = 1
    else:
        v = np.sum(spectrum[rank:]) / (n_features - rank)
        pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank + 1.) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll


def svd_flip(u, v, u_based_decision=True):
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
    """Use high precision for cumsum and check that final value matches sum

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
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


# This shim can be removed once NumPy 1.19.0+ is required (1.18.4 has sign bug)
def svd(a, hermitian=False):
    if hermitian:  # faster
        s, u = np.linalg.eigh(a)
        sgn = np.sign(s)
        s = np.abs(s)
        sidx = np.argsort(s)[..., ::-1]
        sgn = np.take_along_axis(sgn, sidx, axis=-1)
        s = np.take_along_axis(s, sidx, axis=-1)
        u = np.take_along_axis(u, sidx[..., None, :], axis=-1)
        # singular values are unsigned, move the sign into v
        vt = (u * sgn[..., np.newaxis, :]).swapaxes(-2, -1).conj()
        np.abs(s, out=s)
        return u, s, vt
    else:
        return np.linalg.svd(a)


###############################################################################
# From nilearn


def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """
    crop a colorbar to show from cbar_vmin to cbar_vmax
    Used when symmetric_cbar=False is used.
    """
    import matplotlib
    if (cbar_vmin is None) and (cbar_vmax is None):
        return
    cbar_tick_locs = cbar.locator.locs
    if cbar_vmax is None:
        cbar_vmax = cbar_tick_locs.max()
    if cbar_vmin is None:
        cbar_vmin = cbar_tick_locs.min()
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax,
                                len(cbar_tick_locs))

    # matplotlib >= 3.2.0 no longer normalizes axes between 0 and 1
    # See https://matplotlib.org/3.2.1/api/prev_api_changes/api_changes_3.2.0.html
    # _outline was removed in
    # https://github.com/matplotlib/matplotlib/commit/03a542e875eba091a027046d5ec652daa8be6863
    # so we use the code from there
    if _compare_version(matplotlib.__version__, '>=', '3.2.0'):
        cbar.ax.set_ylim(cbar_vmin, cbar_vmax)
        X = cbar._mesh()[0]
        X = np.array([X[0], X[-1]])
        Y = np.array([[cbar_vmin, cbar_vmin], [cbar_vmax, cbar_vmax]])
        N = X.shape[0]
        ii = [0, 1, N - 2, N - 1, 2 * N - 1, 2 * N - 2, N + 1, N, 0]
        x = X.T.reshape(-1)[ii]
        y = Y.T.reshape(-1)[ii]
        xy = (np.column_stack([y, x])
              if cbar.orientation == 'horizontal' else
              np.column_stack([x, y]))
        cbar.outline.set_xy(xy)
    else:
        cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
        outline = cbar.outline.get_xy()
        outline[:2, 1] += cbar.norm(cbar_vmin)
        outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
        outline[6:, 1] += cbar.norm(cbar_vmin)
        cbar.outline.set_xy(outline)

    cbar.set_ticks(new_tick_locs)
    cbar.update_ticks()


###############################################################################
# Numba (optional requirement)

# Here we choose different defaults to speed things up by default
try:
    import numba
    if _compare_version(numba.__version__, '<', '0.48'):
        raise ImportError
    prange = numba.prange
    def jit(nopython=True, nogil=True, fastmath=True, cache=True,
            **kwargs):  # noqa
        return numba.jit(nopython=nopython, nogil=nogil, fastmath=fastmath,
                         cache=cache, **kwargs)
except Exception:  # could be ImportError, SystemError, etc.
    has_numba = False
else:
    has_numba = (os.getenv('MNE_USE_NUMBA', 'true').lower() == 'true')


if not has_numba:
    def jit(**kwargs):  # noqa
        def _jit(func):
            return func
        return _jit
    prange = range
    bincount = np.bincount
    mean = np.mean

else:
    @jit()
    def bincount(x, weights, minlength):  # noqa: D103
        out = np.zeros(minlength)
        for idx, w in zip(x, weights):
            out[idx] += w
        return out

    # fix because Numba does not support axis kwarg for mean
    @jit()
    def _np_apply_along_axis(func1d, axis, arr):
        assert arr.ndim == 2
        assert axis in [0, 1]
        if axis == 0:
            result = np.empty(arr.shape[1])
            for i in range(len(result)):
                result[i] = func1d(arr[:, i])
        else:
            result = np.empty(arr.shape[0])
            for i in range(len(result)):
                result[i] = func1d(arr[i, :])
        return result

    @jit()
    def mean(array, axis):
        return _np_apply_along_axis(np.mean, axis, array)


###############################################################################
# Matplotlib

# workaround: plt.close() doesn't spawn close_event on Agg backend
# https://github.com/matplotlib/matplotlib/issues/18609
# scheduled to be fixed by MPL 3.6
def _close_event(fig):
    """Force calling of the MPL figure close event."""
    from .utils import logger
    from matplotlib import backend_bases
    try:
        fig.canvas.callbacks.process(
            'close_event', backend_bases.CloseEvent(
                name='close_event', canvas=fig.canvas))
        logger.debug(f'Called {fig!r}.canvas.close_event()')
    except ValueError:  # old mpl with Qt
        logger.debug(f'Calling {fig!r}.canvas.close_event() failed')
        pass  # pragma: no cover


def _is_last_row(ax):
    try:
        return ax.get_subplotspec().is_last_row()  # 3.4+
    except AttributeError:
        return ax.is_last_row()
    return ax.get_subplotspec().is_last_row()


def _sharex(ax1, ax2):
    if hasattr(ax1.axes, 'sharex'):
        ax1.axes.sharex(ax2)
    else:
        ax1.get_shared_x_axes().join(ax1, ax2)


###############################################################################
# SciPy deprecation of pinv + pinvh rcond (never worked properly anyway) in 1.7

def pinvh(a, rtol=None):
    """Compute a pseudo-inverse of a Hermitian matrix."""
    s, u = np.linalg.eigh(a)
    del a
    if rtol is None:
        rtol = s.size * np.finfo(s.dtype).eps
    maxS = np.max(np.abs(s))
    above_cutoff = (abs(s) > maxS * rtol)
    psigma_diag = 1.0 / s[above_cutoff]
    u = u[:, above_cutoff]
    return (u * psigma_diag) @ u.conj().T


def pinv(a, rtol=None):
    """Compute a pseudo-inverse of a matrix."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    del a
    maxS = np.max(s)
    if rtol is None:
        rtol = max(vh.shape + u.shape) * np.finfo(u.dtype).eps
    rank = np.sum(s > maxS * rtol)
    u = u[:, :rank]
    u /= s[:rank]
    return (u @ vh[:rank]).conj().T
