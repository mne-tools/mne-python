# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.polynomial.legendre import legval
from scipy import linalg

from ..utils import logger
from ..io.pick import pick_types
from ..surface import _normalize_vectors
from ..bem import _fit_sphere


def _calc_g(cosang, stiffness=4, num_lterms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline.
    num_lterms : int
        number of Legendre terms to evaluate.

    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness *
                              4 * np.pi) for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _calc_h(cosang, stiffness=4, num_lterms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline. Also referred to as `m`.
    num_lterms : int
        number of Legendre terms to evaluate.
    H : np.ndrarray of float, shape(n_channels, n_channels)
        The H matrix.
    """
    factors = [(2 * n + 1) /
               (n ** (stiffness - 1) * (n + 1) ** (stiffness - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines

    Implementation based on [1]

    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpoloate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regularization parameter. Defaults to 1e-5.

    Returns
    -------
    interpolation : np.ndarray of float, shape(len(pos_from), len(pos_to))
        The interpolation matrix that maps good signals to the location
        of bad signals.

    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """

    pos_from = pos_from.copy()
    pos_to = pos_to.copy()

    # normalize sensor positions to sphere
    _normalize_vectors(pos_from)
    _normalize_vectors(pos_to)

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


def _make_interpolator(inst, bad_channels):
    """Find indexes and interpolation matrix to interpolate bad channels

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    """
    bads_idx = np.zeros(len(inst.ch_names), dtype=np.bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=np.bool)

    picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])
    bads_idx[picks] = [inst.ch_names[ch] in bad_channels for ch in picks]
    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    if bads_idx.sum() != len(bad_channels):
        logger.warning('Channel interpolation is currently only implemented '
                       'for EEG. The MEG channels marked as bad will remain '
                       'untouched.')

    pos = inst.get_channel_positions(picks)

    # Make sure only EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]

    pos_good = pos[goods_idx_pos]
    pos_bad = pos[bads_idx_pos]

    # test spherical fit
    radius, center = _fit_sphere(pos_good)
    distance = np.sqrt(np.sum((pos_good - center) ** 2, 1))
    distance = np.mean(distance / radius)
    if np.abs(1. - distance) > 0.1:
        logger.warning('Your spherical fit is poor, interpolation results are '
                       'likely to be inaccurate.')

    logger.info('Computing interpolation matrix from {0} sensor '
                'positions'.format(len(pos_good)))

    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    return goods_idx, bads_idx, interpolation


def _interpolate_bads_eeg(inst):
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

    if 'eeg' not in inst:
        raise ValueError('This interpolation function requires EEG channels.')
    if len(inst.info['bads']) == 0:
        raise ValueError('No bad channels to interpolate.')
    if getattr(inst, 'preload', None) is False:
        raise ValueError('Data must be preloaded.')

    goods_idx, bads_idx, interpolation = _make_interpolator(inst,
                                                            inst.info['bads'])

    logger.info('Interpolating {0} sensors'.format(bads_idx.sum()))
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
            tmp = tmp[:, 0, ...]
        inst._data[:, bads_idx, :] = np.transpose(tmp, (1, 0, 2))
    elif isinstance(inst, Evoked):
        inst.data[bads_idx] = interpolation.dot(inst.data[goods_idx])
    else:
        raise ValueError('Inputs of type {0} are not supported'
                         .format(type(inst)))
    return inst


def _interpolate_bads_eeg_epochs(epochs, bad_channels_by_epoch=None):
    """Interpolate bad channels per epoch

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    bad_channels_by_epoch : list of list of str
        Bad channel names specified for each epoch. For example, for an Epochs
        instance containing 3 epochs: ``[['F1'], [], ['F3', 'FZ']]``
    """
    if len(bad_channels_by_epoch) != len(epochs):
        raise ValueError("Unequal length of epochs (%i) and "
                         "bad_channels_by_epoch (%i)"
                         % (len(epochs), len(bad_channels_by_epoch)))

    interp_cache = {}
    for i, bad_channels in enumerate(bad_channels_by_epoch):
        if not bad_channels:
            continue

        # find interpolation matrix
        key = tuple(sorted(bad_channels))
        if key in interp_cache:
            goods_idx, bads_idx, interpolation = interp_cache[key]
        else:
            goods_idx, bads_idx, interpolation = interp_cache[key] \
                = _make_interpolator(epochs, key)

        # apply interpolation
        logger.info('Interpolating %i sensors on epoch %i', bads_idx.sum(), i)
        epochs._data[i, bads_idx, :] = np.dot(interpolation,
                                              epochs._data[i, goods_idx, :])
