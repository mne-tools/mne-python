
import numpy as np
from numpy.polynomial.legendre import legval
from numpy.linalg import solve

from ..epochs import _BaseEpochs
from ..evoked import Evoked


def _calc_g(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline.
    num_lterms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) / (n ** stiffnes * (n + 1) ** stiffnes * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, factors)


def _calc_h(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline. Also referred to as `m`.
    num_lterms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) ** 2 /
               (n ** stiffnes * (n + 1) ** stiffnes * -(4 * np.pi))
               for n in range(1, num_lterms + 1)]
    return legval(cosang, factors)


def current_source_density(inst, g_matrix=None, h_matrix=None, lambda=1.0e-5):
    """ Current Source Density (CSD) transformation

    Transormation based on spherical spline surface Laplacian as suggested by
    Perrin et al. (1989, 1990), published in appendix of Kayser J, Tenke CE,
    Clin Neurophysiol 2006;117(2):348-368)

    Implementation of algorithms described by Perrin, Pernier, Bertrand, and
    Echallier in Electroenceph Clin Neurophysiol 1989;72(2):184-187, and
    Corrigenda EEG 02274 in Electroenceph Clin Neurophysiol 1990;76:565.

    Parameters
    ----------
    inst : instance of Epochs or Evoked
        The data to be transformed.
    g_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the g_function will be computed from the data (default).
    h_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the h_function will be computed from the data (default).
    lambda : float

    """
    if copy is True:
        out = inst.copy()
    else:
        out = inst
    if isinstance(out, _BaseEpochs):
        data = np.zeros(len(out.events), out.info['nchan'], len(out.times))
        for ii, e in enumerate(out):
            data[ii] = _compute_csd(e, g_matrix=g_matrix, h_matrix=h_matrix,
                                    lambda=lambda)
            out._data = data
            out.preload = True
    elif isinstance(out, Evoked):
        out.data = _compute_csd(out.data, g_matrix=g_matrix, h_matrix=h_matrix,
                                lambda=lambda)
    return out


def interpolate(data, datapos, targetpos, stiffnes=4, num_lterms=50):
    """Spherical spline interpolation of time series.

    Parameters
    ----------
    data : array-like, shape = [n_epochs, n_channels, n_samples]
        Time series data
    datapos : array-like, shape = [n_channels, 3]
        3D positions of the data channels. Each position vector is assumed to
        lie on the unit sphere (must me normalized to 1).
    targetpos : array-like, shape = [n_channels, 3]
        3D positions of interpolation points. Each position vector is assumed
        to lie on the unit sphere (must me normalized to 1).
    stiffnes : float
        stiffnes of the spline
    num_lterms : int
        number of Legendre terms to evaluate
    """

    data = np.asarray(data)
    datapos = np.asarray(datapos)
    targetpos = np.asarray(targetpos)

    _, n_channels, n_samples = data.shape

    source_cosangles = np.dot(datapos, datapos.T)
    source_splines = _calc_g(source_cosangles, stiffnes, num_lterms)

    source_splines = np.ones((1 + n_channels, 1 + n_channels))
    source_splines[-1, 0] = 0
    source_splines[0:-1, 1:] = _calc_g(source_cosangles, stiffnes, num_lterms)

    transfer_cosangles = np.dot(targetpos, datapos.T)
    transfer_splines = _calc_g(transfer_cosangles, stiffnes, num_lterms)

    output = []
    for epoch in data:
        z = np.concatenate((epoch, np.zeros((1, n_samples))))
        coefficients = solve(source_splines, z)
        output.append(np.dot(transfer_splines, coefficients[1:, :]) + coefficients[0, :])
    return np.array(output)
