import numpy as np
from numpy.polynomial.legendre import legval
from scipy import linalg

from .. utils import logger
from ..io.pick import pick_types


def _calc_g(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline.
    num_lterms : int
        number of Legendre terms to evaluate.

    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [(2 * n + 1) / (n ** stiffnes * (n + 1) ** stiffnes * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _calc_h(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline. Also referred to as `m`.
    num_lterms : int
        number of Legendre terms to evaluate.
    H : np.ndrarray of float, shape(n_channels, n_channels)
        The H matrix.
    """
    factors = [(2 * n + 1) /
               (n ** (stiffnes - 1) * (n + 1) ** (stiffnes - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines

    Implementation based on [1]

    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpoloate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regulrization parameter. Defaults to 1e-5.

    Returns
    -------
    interpolation : np.ndarray of float, shape(len(pos_from), len(pos_to))
        The interpolation matrix that maps good signals to the location
        of bad signals.

    Referneces
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """

    pos_from = pos_from.copy()
    pos_to = pos_to.copy()

    # normalize sensor positions to sphere
    pos_from /= np.sqrt(np.sum(pos_from ** 2, 1))[:, None]
    pos_to /= np.sqrt(np.sum(pos_to ** 2, 1))[:, None]

    # cosine angles between source positions
    cosang_from = pos_from.dot(pos_from.T)
    cosang_to_from = pos_to.dot(pos_from.T)
    G_from = _calc_g(cosang_from)
    G_to_from, H_to_from = (f(cosang_to_from) for f in (_calc_g, _calc_h))

    if alpha is not None:
        G_from.flat[::len(G_from) + 1] += alpha

    C_inv = linalg.pinv(G_from)
    interpolation = G_to_from.dot(C_inv)
    return interpolation


def interpolate_bads_eeg(inst):
    """Interpolate bad channels

    Operates in place.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    """
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    from mne.evoked import Evoked
    from mne.preprocessing.maxfilter import _fit_sphere
    if 'eeg' not in inst:
        raise ValueError('This interpolation function requires EEG channels')
    if len(inst.info['bads']) == 0:
        raise ValueError('No bad channels to interpolate')
    if getattr(inst, 'preload', None) is False:
        raise ValueError('Data must be preloaded')

    bads_idx = np.array([ch in inst.info['bads'] for ch in inst.ch_names])
    goods_idx = np.invert(bads_idx)

    picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])
    pos = inst.get_channel_positions(picks)
    pos_good = pos[goods_idx]
    pos_bad = pos[bads_idx]

    # test spherical fit
    radius, center = _fit_sphere(pos_good)
    distance = np.sqrt(np.sum((pos_good - center) ** 2, 1))
    distance = np.mean(np.abs(distance) / radius)
    if 1.0 - distance > 0.1:
        logger.warn('Your spherical fit is poor, interpolation results are '
                    'likely to be inaccurate.')

    logger.info('Computing interpolation matrix from {0} sensor '
                'positions'.format(len(pos_good)))

    interpolation = make_interpolation_matrix(pos_good, pos_bad)

    logger.info('Interpolating {0} sensors'.format(len(pos_bad)))
    if getattr(inst, 'preload', None) is False:
        raise ValueError('Data must be preloaded')

    if isinstance(inst, _BaseRaw):
        inst._data[bads_idx] = interpolation.dot(inst._data[goods_idx])
    elif isinstance(inst, _BaseEpochs):
        tmp = np.dot(interpolation[:, np.newaxis, :],
                     inst._data[:, goods_idx, :])
        if np.sum(bads_idx) == 1:
            tmp = tmp[0]
        else:
            tmp = np.squeeze(tmp)
        inst._data[:, bads_idx, :] = np.transpose(tmp, (1, 0, 2))
    elif isinstance(inst, Evoked):
        inst.data[bads_idx] = interpolation.dot(inst.data[goods_idx])
    else:
        raise ValueError('Inputs of type {0} are not supported'
                         .format(type(inst)))
    return inst
