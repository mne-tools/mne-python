# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          The statsmodels folks for AR yule_walker
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..defaults import _handle_default
from ..io.pick import _pick_data_channels, _picks_by_type, pick_info
from ..utils import verbose


def _yule_walker(X, order=1):
    """Compute Yule-Walker (adapted from statsmodels).

    Operates in-place.
    """
    assert X.ndim == 2
    denom = X.shape[-1] - np.arange(order + 1)
    r = np.zeros(order + 1, np.float64)
    for di, d in enumerate(X):
        d -= d.mean()
        r[0] += np.dot(d, d)
        for k in range(1, order + 1):
            r[k] += np.dot(d[0:-k], d[k:])
    r /= denom * len(X)
    rho = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])
    sigmasq = r[0] - (r[1:] * rho).sum()
    return rho, np.sqrt(sigmasq)


@verbose
def fit_iir_model_raw(raw, order=2, picks=None, tmin=None, tmax=None,
                      verbose=None):
    r"""Fit an AR model to raw data and creates the corresponding IIR filter.

    The computed filter is fitted to data from all of the picked channels,
    with frequency response given by the standard IIR formula:

    .. math::

        H(e^{jw}) = \frac{1}{a[0] + a[1]e^{-jw} + ... + a[n]e^{-jnw}}

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    b : ndarray
        Numerator filter coefficients.
    a : ndarray
        Denominator filter coefficients
    """
    from ..cov import _apply_scaling_array
    start, stop = None, None
    if tmin is not None:
        start = raw.time_as_index(tmin)[0]
    if tmax is not None:
        stop = raw.time_as_index(tmax)[0] + 1
    if picks is None:
        picks = _pick_data_channels(raw.info, exclude='bads')
    data = raw[picks, start:stop][0]
    # rescale data to similar levels
    picks_list = _picks_by_type(pick_info(raw.info, picks))
    scalings = _handle_default('scalings_cov_rank', None)
    _apply_scaling_array(data, picks_list=picks_list, scalings=scalings)
    # do the fitting
    coeffs, _ = _yule_walker(data, order=order)
    return np.array([1.]), np.concatenate(([1.], -coeffs))
