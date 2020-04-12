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
from distutils.version import LooseVersion
from math import log
import os
from pathlib import Path
import warnings

import numpy as np
from scipy import linalg
from scipy.linalg import LinAlgError


###############################################################################
# Misc

# helpers to get function arguments
def _get_args(function, varargs=False):
    params = inspect.signature(function).parameters
    args = [key for key, param in params.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    if varargs:
        varargs = [param.name for param in params.values()
                   if param.kind == param.VAR_POSITIONAL]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args


def _safe_svd(A, **kwargs):
    """Wrapper to get around the SVD did not converge error of death"""
    # Intel has a bug with their GESVD driver:
    #     https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/628049  # noqa: E501
    # For SciPy 0.18 and up, we can work around it by using
    # lapack_driver='gesvd' instead.
    if kwargs.get('overwrite_a', False):
        raise ValueError('Cannot set overwrite_a=True with this function')
    try:
        return linalg.svd(A, **kwargs)
    except np.linalg.LinAlgError as exp:
        from .utils import warn
        if 'lapack_driver' in _get_args(linalg.svd):
            warn('SVD error (%s), attempting to use GESVD instead of GESDD'
                 % (exp,))
            return linalg.svd(A, lapack_driver='gesvd', **kwargs)
        else:
            raise


###############################################################################
# Backporting nibabel's read_geometry

def _get_read_geometry():
    """Get the geometry reading function."""
    try:
        import nibabel as nib
        has_nibabel = True
    except ImportError:
        has_nibabel = False
    if has_nibabel and LooseVersion(nib.__version__) > LooseVersion('2.1.0'):
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
            coords = np.fromfile(fobj, fmt, nvert * 3).astype(np.float) / div
            coords = coords.reshape(-1, 3)
            quads = _fread3_many(fobj, nquad * 4)
            quads = quads.reshape(nquad, 4)
            #
            #   Face splitting follows
            #
            faces = np.zeros((2 * nquad, 3), dtype=np.int)
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

    coords = coords.astype(np.float)  # XXX: due to mayavi bug on mac 32bits

    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn('No volume information contained in the file')
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)

    return ret


###############################################################################
# Triaging scipy.signal.windows.dpss (1.1)

def tridisolve(d, e, b, overwrite_b=True):
    """Symmetric tridiagonal system solver, from Golub and Van Loan p157.

    .. note:: Copied from NiTime.

    Parameters
    ----------
    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector

    Returns
    -------
    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b

    """
    N = len(b)
    # work vectors
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in range(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    # This iterative solver can fail sometimes. There is probably a
    # graceful way to solve this, but it should only be a problem
    # in very rare cases. Users of SciPy 1.1+ will never hit this anyway,
    # so not worth spending more time figuring out how to do it faster.
    if dw[N - 1] == 0:
        a = np.diag(d) + np.diag(e[:-1], -1) + np.diag(e[:-1], 1)
        x[:] = linalg.solve(a, b)
    else:
        for k in range(1, N):
            x[k] = x[k] - ew[k - 1] * x[k - 1]
        if dw[N - 1] != 0:
            x[N - 1] = x[N - 1] / dw[N - 1]
        for k in range(N - 2, -1, -1):
            x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    """Perform an inverse iteration.

    This will find the eigenvector corresponding to the given eigenvalue
    in a symmetric tridiagonal system.

    ..note:: Copied from NiTime.

    Parameters
    ----------
    d : ndarray
      main diagonal of the tridiagonal system
    e : ndarray
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : ndarray
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates

    Returns
    -------
    e: ndarray
      The converged eigenvector
    """
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0


def _dpss(N, half_nbw, Kmax):
    """Compute DPSS windows."""
    # here we want to set up an optimization problem to find a sequence
    # whose energy is maximally concentrated within band [-W,W].
    # Thus, the measure lambda(T,W) is the ratio between the energy within
    # that band, and the total energy. This leads to the eigen-system
    # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
    # eigenvalue is the sequence with maximally concentrated energy. The
    # collection of eigenvectors of this system are called Slepian
    # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
    # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
    # concentration
    # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

    # Here I set up an alternative symmetric tri-diagonal eigenvalue
    # problem such that
    # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
    # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
    # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
    # [see Percival and Walden, 1993]
    nidx = np.arange(N, dtype='d')
    W = float(half_nbw) / N
    diagonal = ((N - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
    off_diag = np.zeros_like(nidx)
    off_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.
    # put the diagonals in LAPACK "packed" storage
    ab = np.zeros((2, N), 'd')
    ab[1] = diagonal
    ab[0, 1:] = off_diag[:-1]
    # only calculate the highest Kmax eigenvalues
    w = linalg.eigvals_banded(ab, select='i',
                              select_range=(N - Kmax, N - 1))
    w = w[::-1]

    # find the corresponding eigenvectors via inverse iteration
    t = np.linspace(0, np.pi, N)
    dpss = np.zeros((Kmax, N), 'd')
    for k in range(Kmax):
        dpss[k] = tridi_inverse_iteration(diagonal, off_diag, w[k],
                                          x0=np.sin((k + 1) * t))

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    # rather than test the sign of one point, test the sign of the
    # linear slope up to the first (largest) peak
    pk = np.argmax(np.abs(dpss[1::2, :N // 2]), axis=1)
    for i, p in enumerate(pk):
        if np.sum(dpss[2 * i + 1, :p]) < 0:
            dpss[2 * i + 1] *= -1

    return dpss


def _get_dpss():
    try:
        from scipy.signal.windows import dpss
    except ImportError:
        dpss = _dpss
    return dpss


###############################################################################
# Triaging FFT functions to get fast pocketfft (SciPy 1.4)

try:
    from scipy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq, ifftshift
except ImportError:
    from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq, ifftshift


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

# Deal with nibabel 2.5 img.get_data() deprecation
def _get_img_fdata(img):
    try:
        return img.get_fdata()
    except AttributeError:
        return img.get_data().astype(float)


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


class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

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
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
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
        Returns
        -------
        self
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
        from sklearn.base import _pprint
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    # __getstate__ and __setstate__ are omitted because they only contain
    # conditionals that are not satisfied by our objects (e.g.,
    # ``if type(self).__module__.startswith('sklearn.')``.


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
        if self.store_precision:
            precision = self.precision_
        else:
            precision = linalg.pinvh(self.covariance_)
        return precision

    def fit(self, X, y=None):
        """Fits the Maximum Likelihood Estimator covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples and
          n_features is the number of features.

        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.

        """
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
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : not used, present for API consistence purpose.

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
        (In the sense of the Frobenius norm).

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


###############################################################################
# NumPy einsum backward compat (allow "optimize" arg and fix 1.14.0 bug)
# XXX eventually we should hand-tune our `einsum` calls given our array sizes!

def einsum(*args, **kwargs):
    if 'optimize' not in kwargs:
        kwargs['optimize'] = False
    return np.einsum(*args, **kwargs)


###############################################################################
# From nilearn

def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """
    crop a colorbar to show from cbar_vmin to cbar_vmax

    Used when symmetric_cbar=False is used.
    """
    if (cbar_vmin is None) and (cbar_vmax is None):
        return
    cbar_tick_locs = cbar.locator.locs
    if cbar_vmax is None:
        cbar_vmax = cbar_tick_locs.max()
    if cbar_vmin is None:
        cbar_vmin = cbar_tick_locs.min()
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax,
                                len(cbar_tick_locs))
    cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
    outline = cbar.outline.get_xy()
    outline[:2, 1] += cbar.norm(cbar_vmin)
    outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
    outline[6:, 1] += cbar.norm(cbar_vmin)
    cbar.outline.set_xy(outline)
    cbar.set_ticks(new_tick_locs, update_ticks=True)


###############################################################################
# Matplotlib

def _get_status(checks):
    """Deal with old MPL to get check box statuses."""
    try:
        return list(checks.get_status())
    except AttributeError:
        return [x[0].get_visible() for x in checks.lines]


###############################################################################
# Numba (optional requirement)

# Here we choose different defaults to speed things up by default
try:
    import numba
    if LooseVersion(numba.__version__) < LooseVersion('0.40'):
        raise ImportError
    prange = numba.prange
    def jit(nopython=True, nogil=True, fastmath=True, cache=True,
            **kwargs):  # noqa
        return numba.jit(nopython=nopython, nogil=nogil, fastmath=fastmath,
                         cache=cache, **kwargs)
except ImportError:
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
# Python 3.5 compat with pathlib.Path-like objects

def _fn35(fname):
    try:
        from py._path.common import PathBase
    except ImportError:
        pass
    else:
        if isinstance(fname, PathBase):
            fname = str(fname)
    if isinstance(fname, Path):
        fname = str(fname)
    return fname
