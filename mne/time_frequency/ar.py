# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          The statsmodels folks for AR yule_walker
#
# License: BSD (3-clause)

import numpy as np
from scipy.linalg import toeplitz

from ..io.pick import pick_types
from ..utils import verbose


# XXX : Back ported from statsmodels

def yule_walker(X, order=1, method="unbiased", df=None, inv=False,
                demean=True):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    Parameters
    ----------
    X : array-like
        1d array
    order : integer, optional
        The order of the autoregressive process.  Default is 1.
    method : string, optional
       Method can be "unbiased" or "mle" and this determines denominator in
       estimate of autocorrelation function (ACF) at lag k. If "mle", the
       denominator is n=X.shape[0], if "unbiased" the denominator is n-k.
       The default is unbiased.
    df : integer, optional
       Specifies the degrees of freedom. If `df` is supplied, then it is
       assumed the X has `df` degrees of freedom rather than `n`.  Default is
       None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho
        The autoregressive coefficients
    sigma
        TODO

    """
    # TODO: define R better, look back at notes and technical notes on YW.
    # First link here is useful
    # http://www-stat.wharton.upenn.edu/~steele/Courses/956/ResourceDetails/YuleWalkerAndMore.htm  # noqa
    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError("ACF estimation method must be 'unbiased' or 'MLE'")
    X = np.array(X)
    if demean:
        X -= X.mean()                  # automatically demean's X
    n = df or X.shape[0]

    if method == "unbiased":        # this is df_resid ie., n - p
        def denom(k):
            return n - k
    else:
        def denom(k):
            return n
    if X.ndim > 1 and X.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")
    r = np.zeros(order + 1, np.float64)
    r[0] = (X ** 2).sum() / denom(0)
    for k in range(1, order + 1):
        r[k] = (X[0:-k] * X[k:]).sum() / denom(k)
    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:] * rho).sum()
    if inv:
        return rho, np.sqrt(sigmasq), np.linalg.inv(R)
    else:
        return rho, np.sqrt(sigmasq)


def ar_raw(raw, order, picks, tmin=None, tmax=None):
    """Fit AR model on raw data

    Fit AR models for each channels and returns the models
    coefficients for each of them.

    Parameters
    ----------
    raw : Raw instance
        The raw data
    order : int
        The AR model order
    picks : array-like of int
        The channels indices to include
    tmin : float
        The beginning of time interval in seconds.
    tmax : float
        The end of time interval in seconds.

    Returns
    -------
    coefs : array
        Sets of coefficients for each channel
    """
    start, stop = None, None
    if tmin is not None:
        start = raw.time_as_index(tmin)[0]
    if tmax is not None:
        stop = raw.time_as_index(tmax)[0] + 1
    data, times = raw[picks, start:stop]

    coefs = np.empty((len(data), order))
    for k, d in enumerate(data):
        this_coefs, _ = yule_walker(d, order=order)
        coefs[k, :] = this_coefs
    return coefs


@verbose
def fit_iir_model_raw(raw, order=2, picks=None, tmin=None, tmax=None,
                      verbose=None):
    """Fits an AR model to raw data and creates the corresponding IIR filter

    The computed filter is the average filter for all the picked channels.
    The frequency response is given by:

    .. math::

        H(\\exp^{jw}) = \\frac{1}{a[0] + a[1]\\exp{-jw} + ...
                                  + a[n]\\exp{-jnw}}

    Parameters
    ----------
    raw : Raw object
        an instance of Raw.
    order : int
        order of the FIR filter.
    picks : array-like of int | None
        indices of selected channels. If None, MEG and EEG channels are used.
    tmin : float
        The beginning of time interval in seconds.
    tmax : float
        The end of time interval in seconds.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    b : ndarray
        Numerator filter coefficients.
    a : ndarray
        Denominator filter coefficients
    """
    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True)
    coefs = ar_raw(raw, order=order, picks=picks, tmin=tmin, tmax=tmax)
    mean_coefs = np.mean(coefs, axis=0)  # mean model across channels
    a = np.concatenate(([1.], -mean_coefs))  # filter coefficients
    return np.array([1.]), a
