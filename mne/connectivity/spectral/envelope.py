# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Sheraz Khan <sheraz@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np


def compute_corr(x, y):
    """Correlate 2 matrices along last axis.

    Parameters
    ----------
    x : np.ndarray of shape(n_time_series, n_times)
        The first set of vectors.
    y : np.ndarray of shape(n_time_series, n_times)
        The second set of vectors.

    Retrurns
    --------
    r : np.ndarray of shape(n_time_series,)
        The correlation betwen x and y.
    """
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) *
                    np.sum(ym * ym, axis=-1))
    r = np.sum(xm * ym, axis=-1) / r_den
    return r


def _orthogonalize(a, b):
    """Orthogonalize x on y."""
    return np.imag(a * (b.conj() / np.abs(b)))


def compute_envelope_correlation(data):
    """Compute power envelope correlation with orthogonalization.

    .. Note:
        Input should be bandpass filtered power envelopes,
        either obtained from Hilbert transform or the
        rectified signal.

    Parameters
    ----------
    epochs_data : np.ndarray of shape(n_sources, n_times)
        Channels, source time series.

    Returns
    -------
    corr : np.ndarray of shape (n_labels, n_labels)
        The connectivity matrix.
    """
    n_features = data.shape[0]
    corr = np.zeros((n_features, n_features), dtype=np.float)
    for ii, x in enumerate(data):
        jj = ii + 1
        y = data[jj:]
        x_, y_ = _orthogonalize(a=x, b=y), _orthogonalize(a=y, b=x)
        # take abs, sign is ambiguous.
        this_corr = np.mean((
            np.abs(compute_corr(np.abs(x), y_)),
            np.abs(compute_corr(np.abs(y), x_))), axis=0)
        corr[ii:jj, jj:] = this_corr

    corr.flat[::n_features + 1] = 0  # orthogonalized correlation should be 0
    return corr + corr.T  # mirror lower diagonal
