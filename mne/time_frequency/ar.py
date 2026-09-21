# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy import linalg

from .._fiff.pick import _picks_by_type, _picks_to_idx, pick_info
from ..defaults import _handle_default
from ..utils import _apply_scaling_array, verbose_static


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


@verbose_static("picks_good_data")
def fit_iir_model_raw(raw, order=2, picks=None, tmin=None, tmax=None, verbose=None):
    r"""Fit an AR model to raw data and creates the corresponding IIR filter.

    The computed filter is fitted to data from all of the picked channels,
    with frequency response given by the standard IIR formula:

    .. math::

        H(e^{jw}) = \frac{1}{a[0] + a[1]e^{-jw} + ... + a[n]e^{-jnw}}

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    order : int
        Order of the FIR filter.
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick good data channels. Note
        that channels in ``info['bads']`` *will be included* if their names or
        indices are explicitly provided.
    tmin : float
        The beginning of time interval in seconds.
    tmax : float
        The end of time interval in seconds.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    b : ndarray
        Numerator filter coefficients.
    a : ndarray
        Denominator filter coefficients.
    """
    start, stop = None, None
    if tmin is not None:
        start = raw.time_as_index(tmin)[0]
    if tmax is not None:
        stop = raw.time_as_index(tmax)[0] + 1
    picks = _picks_to_idx(raw.info, picks)
    data = raw[picks, start:stop][0]
    # rescale data to similar levels
    picks_list = _picks_by_type(pick_info(raw.info, picks))
    scalings = _handle_default("scalings_cov_rank", None)
    _apply_scaling_array(data, picks_list=picks_list, scalings=scalings)
    # do the fitting
    coeffs, _ = _yule_walker(data, order=order)
    return np.array([1.0]), np.concatenate(([1.0], -coeffs))
