# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

import numpy as np
from numpy.polynomial.legendre import legval

from ..utils import logger, warn, verbose
from ..io.meas_info import _simplify_info
from ..io.pick import pick_types, pick_channels, pick_info
from ..surface import _normalize_vectors
from ..forward import _map_meg_or_eeg_channels
from ..utils import _check_option, _validate_type


def _calc_h(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffnes of the spline. Also referred to as ``m``.
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
    from scipy import linalg
    pos_from = pos_from.copy()
    pos_to = pos_to.copy()
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]

    # normalize sensor positions to sphere
    _normalize_vectors(pos_from)
    _normalize_vectors(pos_to)

    # cosine angles between source positions
    cosang_from = pos_from.dot(pos_from.T)
    cosang_to_from = pos_to.dot(pos_from.T)
    G_from = _calc_g(cosang_from)
    G_to_from = _calc_g(cosang_to_from)
    assert G_from.shape == (n_from, n_from)
    assert G_to_from.shape == (n_to, n_from)

    if alpha is not None:
        G_from.flat[::len(G_from) + 1] += alpha

    C = np.vstack([np.hstack([G_from, np.ones((n_from, 1))]),
                   np.hstack([np.ones((1, n_from)), [[0]]])])
    C_inv = linalg.pinv(C)

    interpolation = np.hstack([G_to_from, np.ones((n_to, 1))]) @ C_inv[:, :-1]
    assert interpolation.shape == (n_to, n_from)
    return interpolation


def _do_interp_dots(inst, interpolation, goods_idx, bads_idx):
    """Dot product of channel mapping matrix to channel data."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), 'inst')
    inst._data[..., bads_idx, :] = np.matmul(
        interpolation, inst._data[..., goods_idx, :])


@verbose
def _interpolate_bads_eeg(inst, origin, exclude=None, verbose=None):
    if exclude is None:
        exclude = list()
    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)

    picks = pick_types(inst.info, meg=False, eeg=True, exclude=exclude)
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


def _interpolate_bads_meg(inst, mode='accurate', origin=(0., 0., 0.04),
                          verbose=None, ref_meg=False):
    return _interpolate_bads_meeg(
        inst, mode, origin, ref_meg=ref_meg, eeg=False, verbose=verbose)


@verbose
def _interpolate_bads_meeg(inst, mode='accurate', origin=(0., 0., 0.04),
                           meg=True, eeg=True, ref_meg=False,
                           exclude=(), verbose=None):
    bools = dict(meg=meg, eeg=eeg)
    info = _simplify_info(inst.info)
    for ch_type, do in bools.items():
        if not do:
            continue
        kw = dict(meg=False, eeg=False)
        kw[ch_type] = True
        picks_type = pick_types(info, ref_meg=ref_meg, exclude=exclude, **kw)
        picks_good = pick_types(info, ref_meg=ref_meg, exclude='bads', **kw)
        use_ch_names = [inst.info['ch_names'][p] for p in picks_type]
        bads_type = [ch for ch in inst.info['bads'] if ch in use_ch_names]
        if len(bads_type) == 0 or len(picks_type) == 0:
            continue
        # select the bad channels to be interpolated
        picks_bad = pick_channels(inst.info['ch_names'], bads_type,
                                  exclude=[])
        if ch_type == 'eeg':
            picks_to = picks_type
            bad_sel = np.in1d(picks_type, picks_bad)
        else:
            picks_to = picks_bad
            bad_sel = slice(None)
        info_from = pick_info(inst.info, picks_good)
        info_to = pick_info(inst.info, picks_to)
        mapping = _map_meg_or_eeg_channels(
            info_from, info_to, mode=mode, origin=origin)
        mapping = mapping[bad_sel]
        _do_interp_dots(inst, mapping, picks_good, picks_bad)


@verbose
def _interpolate_bads_nirs(inst, method='nearest', exclude=(), verbose=None):
    from scipy.spatial.distance import pdist, squareform
    from mne.preprocessing.nirs import _validate_nirs_info

    if len(pick_types(inst.info, fnirs=True, exclude=())) == 0:
        return

    # Returns pick of all nirs and ensures channels are correctly ordered
    picks_nirs = _validate_nirs_info(inst.info)
    nirs_ch_names = [inst.info['ch_names'][p] for p in picks_nirs]
    nirs_ch_names = [ch for ch in nirs_ch_names if ch not in exclude]
    bads_nirs = [ch for ch in inst.info['bads'] if ch in nirs_ch_names]
    if len(bads_nirs) == 0:
        return
    picks_bad = pick_channels(inst.info['ch_names'], bads_nirs, exclude=[])
    bads_mask = [p in picks_bad for p in picks_nirs]

    chs = [inst.info['chs'][i] for i in picks_nirs]
    locs3d = np.array([ch['loc'][:3] for ch in chs])

    _check_option('fnirs_method', method, ['nearest'])

    if method == 'nearest':

        dist = pdist(locs3d)
        dist = squareform(dist)

        for bad in picks_bad:
            dists_to_bad = dist[bad]
            # Ignore distances to self
            dists_to_bad[dists_to_bad == 0] = np.inf
            # Ignore distances to other bad channels
            dists_to_bad[bads_mask] = np.inf
            # Find closest remaining channels for same frequency
            closest_idx = np.argmin(dists_to_bad) + (bad % 2)
            inst._data[bad] = inst._data[closest_idx]

        inst.info['bads'] = [ch for ch in inst.info['bads'] if ch in exclude]

    return inst
