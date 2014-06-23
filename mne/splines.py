
import numpy as np
from numpy.polynomial.legendre import legval
from numpy.linalg import solve, pinv

from .epochs import _BaseEpochs
from .evoked import Evoked

from .io.pick import pick_types


def _get_positions(info, picks):
    """Helper to get positions"""
    positions = np.array([c['loc'][:3] for c in info['chs']])[picks]
    return positions


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
    factors = [(2 * n + 1) /
               (n ** (stiffnes - 1) * (n + 1) ** (stiffnes - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, factors)


def _compute_csd(data, G, H, lambda2, head):
    """compute the CSD"""
    n_channels, n_times = data.shape
    Z = data - np.mean(data, 0)[None]  # XXX? compute average reference
    X = data
    head **= 2  # or rescale data to head sphere

    # regularize if desired
    if lambda2 is None:
        G.flat[::len(G) + 1] += lambda2

    # compute the CSD
    Gi = pinv(G)
    TC = Gi.sum(0)
    sgi = np.sum(TC)  # compute sum total
    for this_time in range(n_times):
        Cp = np.dot(Gi, Z[:, this_time])  # compute preliminary C vector
        c0 = np.sum(Cp) / sgi  # common constant across electrodes
        C = Cp - np.dot(c0, TC.T)  # compute final C vector
        for this_chan in range(n_channels):  # compute all CSDs ...
        # ... and scale to head size
            X[this_chan, this_time] = np.sum(C * H[this_chan].T) / head
    return X


def current_source_density(inst, ch_type='eeg', g_matrix=None, h_matrix=None,
                           lambda2=1e-5, head=1.0, copy=True):
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
    ch_type : str
        The channel type.
    g_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the g_function will be computed from the data (default).
    h_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the h_function will be computed from the data (default).
    lambda2 : float
        Regularization parameter, produces smoothnes. Defaults to 1e-5.
    head : float
        The head radius (unit sphere). Defaults to 1.
    copy : bool
        Whether to overwrite instance data or create a copy.

    Returns
    -------
    inst_csd : instance of Epochs or Evoked
        The transformed data. Output type will match input type.
    """

    if copy is True:
        out = inst.copy()
    else:
        out = inst
    if ch_type == 'eeg':
        picks = pick_types(inst.info, meg=False, eeg=True)
    else:
        raise ValueError('currently only eeg is supportedd')

    if g_matrix is None or h_matrix is None:
        pos = _get_positions(inst.info, picks)

    G = _calc_g(np.dot(pos, pos.T)) if g_matrix is None else g_matrix
    H = _calc_h(np.dot(pos, pos.T)) if h_matrix is None else h_matrix

    if isinstance(out, _BaseEpochs):
        data = np.zeros(len(out.events), len(picks), len(out.times))
        for ii, e in enumerate(out):
            data[ii] = _compute_csd(e[picks], G=G, H=H, lambda2=lambda2,
                                    head=head)
            out._data = data
            out.preload = True
    elif isinstance(out, Evoked):
        out.data = _compute_csd(out.data[picks], G=G, H=H, lambda2=lambda2,
                                head=head)
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
