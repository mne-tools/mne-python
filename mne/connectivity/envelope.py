# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..filter import next_fast_len
from ..utils import _validate_type, verbose


@verbose
def envelope_correlation(data, combine='mean', verbose=None):
    """Compute the envelope correlation.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity.
        The array-like object can also be a list/generator of array,
        each with shape (n_signals, n_times). If it's float data,
        the Hilbert transform will be applied; if it's complex data,
        it's assumed the Hilbert has already been applied.
    combine : 'mean' | callable | None
        How to combine correlation estimates across epochs.
        Default is 'mean'. Can be None to return without combining.
        If callable, it must accept one positional input.
        For example::

            combine = lambda data: np.median(data, axis=0)
    %(verbose)s

    Returns
    -------
    corr : ndarray, shape ([n_epochs, ]n_nodes, n_nodes)
        The pairwise orthogonal envelope correlations.
        This matrix is symmetric. If combine is None, the array
        with have three dimensions, the first of which is ``n_epochs``.

    Notes
    -----
    This function computes the power envelope correlation between
    orthogonalized signals [1]_ [2]_.

    References
    ----------
    .. [1] Hipp JF, Hawellek DJ, Corbetta M, Siegel M, Engel AK (2012)
           Large-scale cortical correlation structure of spontaneous
           oscillatory activity. Nature Neuroscience 15:884–890
    .. [2] Khan S et al. (2018). Maturation trajectories of cortical
           resting-state networks depend on the mediating frequency band.
           Neuroimage 174:57–68
    """
    from scipy.signal import hilbert
    corrs = list()
    n_nodes = None
    if not callable(combine):
        _validate_type(combine, (str, None), 'combine')
    for epoch_data in data:
        if epoch_data.ndim != 2:
            raise ValueError('Each entry in data must be 2D, got shape %s'
                             % (epoch_data.shape,))
        this_n_nodes, n_times = epoch_data.shape
        if n_nodes is None:
            n_nodes = this_n_nodes
        if this_n_nodes != n_nodes:
            raise ValueError('n_nodes mismatch between epochs, got %s and %s'
                             % (n_nodes, this_n_nodes))
        # Get the complex envelope (allowing complex inputs allows people
        # to do raw.apply_hilbert if they want)
        if epoch_data.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            epoch_data = hilbert(epoch_data, N=n_fft, axis=-1)[..., :n_times]
        if epoch_data.dtype not in (np.complex64, np.complex128):
            raise ValueError('data.dtype must be float or complex, got %s'
                             % (epoch_data.dtype,))
        data_mag = np.abs(epoch_data)
        data_conj_scaled = epoch_data.conj()
        data_conj_scaled /= data_mag
        # subtract means
        data_mag_nomean = data_mag - np.mean(data_mag, axis=-1, keepdims=True)
        # compute variances using linalg.norm (square, sum, sqrt) since mean=0
        data_mag_std = np.linalg.norm(data_mag_nomean, axis=-1)
        data_mag_std[data_mag_std == 0] = 1
        corr = np.empty((n_nodes, n_nodes))
        for li, label_data in enumerate(epoch_data):
            label_data_orth = (label_data * data_conj_scaled).imag
            label_data_orth -= np.mean(label_data_orth, axis=-1, keepdims=True)
            label_data_orth_std = np.linalg.norm(label_data_orth, axis=-1)
            label_data_orth_std[label_data_orth_std == 0] = 1
            # correlation is dot product divided by variances
            corr[li] = np.dot(label_data_orth, data_mag_nomean[li])
            corr[li] /= data_mag_std[li]
            corr[li] /= label_data_orth_std
        # Make it symmetric (it isn't at this point)
        corr = np.abs(corr)
        corrs.append((corr.T + corr) / 2.)
        del corr
    if isinstance(combine, str):
        if combine == 'mean':
            corr = np.mean(corrs, axis=0)
        else:
            raise ValueError('Unknown combine option %r' % (combine,))
    elif callable(combine):
        corr = combine(corrs)
    else:  # None
        corr = np.array(corrs)
    return corr
