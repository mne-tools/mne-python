# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.polynomial.legendre import legval
from scipy import linalg

from ..fixes import einsum
from ..utils import logger, warn, verbose
from ..io.pick import pick_types, pick_channels, pick_info
from ..surface import _normalize_vectors
from ..forward import _map_meg_channels


def _calc_h(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffnes of the spline. Also referred to as `m`.
    n_legendre_terms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) /
               (n ** (stiffness - 1) * (n + 1) ** (stiffness - 1) * 4 * np.pi)
               for n in range(1, n_legendre_terms + 1)]
    return legval(cosang, [0] + factors)


def _calc_g(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline.
    n_legendre_terms : int
        number of Legendre terms to evaluate.

    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness *
                              4 * np.pi)
               for n in range(1, n_legendre_terms + 1)]
    return legval(cosang, [0] + factors)


def _make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines.

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
    G_to_from = _calc_g(cosang_to_from)

    if alpha is not None:
        G_from.flat[::len(G_from) + 1] += alpha

    n_channels = G_from.shape[0]  # G_from should be square matrix
    C = np.r_[np.c_[G_from, np.ones((n_channels, 1))],
              np.c_[np.ones((1, n_channels)), 0]]
    C_inv = linalg.pinv(C)

    interpolation = np.c_[G_to_from,
                          np.ones((G_to_from.shape[0], 1))].dot(C_inv[:, :-1])
    return interpolation


def _do_interp_dots(inst, interpolation, goods_idx, bads_idx):
    """Dot product of channel mapping matrix to channel data."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked

    if isinstance(inst, (BaseRaw, Evoked)):
        inst._data[bads_idx] = interpolation.dot(inst._data[goods_idx])
    elif isinstance(inst, BaseEpochs):
        inst._data[:, bads_idx, :] = einsum(
            'ij,xjy->xiy', interpolation, inst._data[:, goods_idx, :])
    else:
        raise ValueError('Inputs of type {} are not supported'
                         .format(type(inst)))


@verbose
def _interpolate_bads_eeg(inst, origin, verbose=None):
    """Interpolate bad EEG channels.

    Operates in place.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    """
    bads_idx = np.zeros(len(inst.ch_names), dtype=np.bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=np.bool)

    picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])
    inst.info._check_consistency()
    bads_idx[picks] = [inst.ch_names[ch] in inst.info['bads'] for ch in picks]

    if len(picks) == 0 or bads_idx.sum() == 0:
        return

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = inst._get_channel_positions(picks)

    # Make sure only EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]

    # test spherical fit
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1. - distance) > 0.1:
        warn('Your spherical fit is poor, interpolation results are '
             'likely to be inaccurate.')

    pos_good = pos[goods_idx_pos] - origin
    pos_bad = pos[bads_idx_pos] - origin
    logger.info('Computing interpolation matrix from {} sensor '
                'positions'.format(len(pos_good)))
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    logger.info('Interpolating {} sensors'.format(len(pos_bad)))
    _do_interp_dots(inst, interpolation, goods_idx, bads_idx)


@verbose
def _interpolate_bads_meg(inst, mode='accurate', origin=(0., 0., 0.04),
                          verbose=None, ref_meg=False):
    """Interpolate bad channels from data in good channels.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used for interpolation. `'fast'` should
        be sufficient for most applications.
    origin : array-like, shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'``, which means a head-digitization-based origin
        fit. Default is ``(0., 0., 0.04)``.
    %(verbose)s
    ref_meg : bool
        Should always be False; only exists for testing purpose.
    """
    picks_meg = pick_types(inst.info, meg=True, eeg=False,
                           ref_meg=ref_meg, exclude=[])
    picks_good = pick_types(inst.info, meg=True, eeg=False,
                            ref_meg=ref_meg, exclude='bads')
    meg_ch_names = [inst.info['ch_names'][p] for p in picks_meg]
    bads_meg = [ch for ch in inst.info['bads'] if ch in meg_ch_names]

    # select the bad meg channel to be interpolated
    if len(bads_meg) == 0:
        picks_bad = []
    else:
        picks_bad = pick_channels(inst.info['ch_names'], bads_meg,
                                  exclude=[])

    # return without doing anything if there are no meg channels
    if len(picks_meg) == 0 or len(picks_bad) == 0:
        return
    info_from = pick_info(inst.info, picks_good)
    info_to = pick_info(inst.info, picks_bad)
    mapping = _map_meg_channels(info_from, info_to, mode=mode, origin=origin)
    _do_interp_dots(inst, mapping, picks_good, picks_bad)
