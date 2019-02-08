# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..filter import next_fast_len


def envelope_correlation(data):
    """Compute the envelope correlation.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity.
        The array-like object can also be a list/generator of array,
        each with shape (n_signals, n_times). If it's float data,
        the Hilbert transform will be applied; if it's complex data,
        it's assumed the Hilbert has already been applied.

    Returns
    -------
    corr : ndarray, shape (n_nodes, n_nodes)
        The pairwise orthogonal envelope correlations.
        This matrix is symmetric.

    Notes
    -----
    This function computes the orthogonal envelope correlation between
    time series.
    """
    from scipy.signal import hilbert
    corrs = list()
    n_nodes = None
    for data_ in data:
        if data_.ndim != 2:
            raise ValueError('Each entry in data must be 2D, got shape %s'
                             % (data_.shape,))
        this_n_nodes, n_times = data_.shape
        if n_nodes is None:
            n_nodes = this_n_nodes
        if this_n_nodes != n_nodes:
            raise ValueError('n_nodes mismatch between epochs, got %s and %s'
                             % (n_nodes, this_n_nodes))
        # Get the complex envelope (allowing complex inputs allows people
        # to do raw.apply_hilbert if they want)
        if data_.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            data_ = hilbert(data_, N=n_fft, axis=-1)[..., :n_times]
        if data_.dtype not in (np.complex64, np.complex128):
            raise ValueError('data.dtype must be float or complex, got %s'
                             % (data_.dtype,))
        data_mag = np.abs(data_)
        data_orth = np.einsum('it,jt->ijt', data_,
                              data_.conj() / data_mag).imag
        # subtract means
        data_mag -= np.mean(data_mag, axis=-1, keepdims=True)
        data_orth -= np.mean(data_orth, axis=-1, keepdims=True)
        # compute variances using dot products
        data_mag_var = np.sum(data_mag * data_mag, axis=-1, keepdims=True)
        data_orth_var = np.einsum('ijt,ijt->ij', data_orth, data_orth)
        # correlation is dot product divided by variances
        corr = np.einsum('it,ijt->ij', data_mag, data_orth)
        corr /= np.sqrt(data_mag_var)
        corr /= np.sqrt(data_orth_var)
        # we always make the matrix symmetric
        corr = np.abs(corr)
        corrs.append((corr.T + corr) / 2.)
    corr = np.median(corrs, axis=0)
    return corr
