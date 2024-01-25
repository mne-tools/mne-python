# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Ana Radanovic <radanovica@protonmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import pinv
from scipy.spatial.distance import pdist, squareform

from .._fiff.meas_info import _simplify_info
from .._fiff.pick import pick_channels, pick_info, pick_types
from ..surface import _normalize_vectors
from ..utils import _validate_type, logger, verbose, warn


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
    factors = [
        (2 * n + 1) / (n ** (stiffness - 1) * (n + 1) ** (stiffness - 1) * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
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
    factors = [
        (2 * n + 1) / (n**stiffness * (n + 1) ** stiffness * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return legval(cosang, [0] + factors)


def _make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines.

    Implementation based on [1]

    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpolate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpolate.
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
        G_from.flat[:: len(G_from) + 1] += alpha

    C = np.vstack(
        [
            np.hstack([G_from, np.ones((n_from, 1))]),
            np.hstack([np.ones((1, n_from)), [[0]]]),
        ]
    )
    C_inv = pinv(C)

    interpolation = np.hstack([G_to_from, np.ones((n_to, 1))]) @ C_inv[:, :-1]
    assert interpolation.shape == (n_to, n_from)
    return interpolation


def _do_interp_dots(inst, interpolation, goods_idx, bads_idx):
    """Dot product of channel mapping matrix to channel data."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..io import BaseRaw

    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), "inst")
    inst._data[..., bads_idx, :] = np.matmul(
        interpolation, inst._data[..., goods_idx, :]
    )


@verbose
def _interpolate_bads_eeg(inst, origin, exclude=None, ecog=False, verbose=None):
    if exclude is None:
        exclude = list()
    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)

    picks = pick_types(inst.info, meg=False, eeg=not ecog, ecog=ecog, exclude=exclude)
    inst.info._check_consistency()
    bads_idx[picks] = [inst.ch_names[ch] in inst.info["bads"] for ch in picks]

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
    if np.abs(1.0 - distance) > 0.1:
        warn(
            "Your spherical fit is poor, interpolation results are "
            "likely to be inaccurate."
        )

    pos_good = pos[goods_idx_pos] - origin
    pos_bad = pos[bads_idx_pos] - origin
    logger.info(f"Computing interpolation matrix from {len(pos_good)} sensor positions")
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    logger.info(f"Interpolating {len(pos_bad)} sensors")
    _do_interp_dots(inst, interpolation, goods_idx, bads_idx)


@verbose
def _interpolate_bads_ecog(inst, origin, exclude=None, verbose=None):
    _interpolate_bads_eeg(inst, origin, exclude=exclude, ecog=True, verbose=verbose)


def _interpolate_bads_meg(
    inst, mode="accurate", origin=(0.0, 0.0, 0.04), verbose=None, ref_meg=False
):
    return _interpolate_bads_meeg(
        inst, mode, origin, ref_meg=ref_meg, eeg=False, verbose=verbose
    )


@verbose
def _interpolate_bads_nan(
    inst,
    ch_type,
    ref_meg=False,
    exclude=(),
    *,
    verbose=None,
):
    info = _simplify_info(inst.info)
    picks_type = pick_types(info, ref_meg=ref_meg, exclude=exclude, **{ch_type: True})
    use_ch_names = [inst.info["ch_names"][p] for p in picks_type]
    bads_type = [ch for ch in inst.info["bads"] if ch in use_ch_names]
    if len(bads_type) == 0 or len(picks_type) == 0:
        return
    # select the bad channels to be interpolated
    picks_bad = pick_channels(inst.info["ch_names"], bads_type, exclude=[])
    inst._data[..., picks_bad, :] = np.nan


@verbose
def _interpolate_bads_meeg(
    inst,
    mode="accurate",
    origin=(0.0, 0.0, 0.04),
    meg=True,
    eeg=True,
    ref_meg=False,
    exclude=(),
    *,
    method=None,
    verbose=None,
):
    from ..forward import _map_meg_or_eeg_channels

    if method is None:
        method = {"meg": "MNE", "eeg": "MNE"}
    bools = dict(meg=meg, eeg=eeg)
    info = _simplify_info(inst.info)
    for ch_type, do in bools.items():
        if not do:
            continue
        kw = dict(meg=False, eeg=False)
        kw[ch_type] = True
        picks_type = pick_types(info, ref_meg=ref_meg, exclude=exclude, **kw)
        picks_good = pick_types(info, ref_meg=ref_meg, exclude="bads", **kw)
        use_ch_names = [inst.info["ch_names"][p] for p in picks_type]
        bads_type = [ch for ch in inst.info["bads"] if ch in use_ch_names]
        if len(bads_type) == 0 or len(picks_type) == 0:
            continue
        # select the bad channels to be interpolated
        picks_bad = pick_channels(inst.info["ch_names"], bads_type, exclude=[])

        # do MNE based interpolation
        if ch_type == "eeg":
            picks_to = picks_type
            bad_sel = np.isin(picks_type, picks_bad)
        else:
            picks_to = picks_bad
            bad_sel = slice(None)
        info_from = pick_info(inst.info, picks_good)
        info_to = pick_info(inst.info, picks_to)
        mapping = _map_meg_or_eeg_channels(info_from, info_to, mode=mode, origin=origin)
        mapping = mapping[bad_sel]
        _do_interp_dots(inst, mapping, picks_good, picks_bad)


@verbose
def _interpolate_bads_nirs(inst, exclude=(), verbose=None):
    from mne.preprocessing.nirs import _validate_nirs_info

    if len(pick_types(inst.info, fnirs=True, exclude=())) == 0:
        return

    # Returns pick of all nirs and ensures channels are correctly ordered
    picks_nirs = _validate_nirs_info(inst.info)
    nirs_ch_names = [inst.info["ch_names"][p] for p in picks_nirs]
    nirs_ch_names = [ch for ch in nirs_ch_names if ch not in exclude]
    bads_nirs = [ch for ch in inst.info["bads"] if ch in nirs_ch_names]
    if len(bads_nirs) == 0:
        return
    picks_bad = pick_channels(inst.info["ch_names"], bads_nirs, exclude=[])
    bads_mask = [p in picks_bad for p in picks_nirs]

    chs = [inst.info["chs"][i] for i in picks_nirs]
    locs3d = np.array([ch["loc"][:3] for ch in chs])

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

    # TODO: this seems like a bug because it does not respect reset_bads
    inst.info["bads"] = [ch for ch in inst.info["bads"] if ch in exclude]

    return inst


def _find_seeg_electrode_shaft(pos, tol=2e-3):
    # 1) find nearest neighbor to define the electrode shaft line
    # 2) find all contacts on the same line

    dist = squareform(pdist(pos))
    np.fill_diagonal(dist, np.inf)

    shafts = list()
    for i, n1 in enumerate(pos):
        if any([i in shaft for shaft in shafts]):
            continue
        n2 = pos[np.argmin(dist[i])]  # 1
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        shaft_dists = np.linalg.norm(
            np.cross((pos - n1), (pos - n2)), axis=1
        ) / np.linalg.norm(n2 - n1)
        shafts.append(np.where(shaft_dists < tol)[0])  # 2
    return shafts


@verbose
def _interpolate_bads_seeg(inst, exclude=None, tol=2e-3, verbose=None):
    if exclude is None:
        exclude = list()
    picks = pick_types(inst.info, meg=False, seeg=True, exclude=exclude)
    inst.info._check_consistency()
    bads_idx = np.isin(np.array(inst.ch_names)[picks], inst.info["bads"])

    if len(picks) == 0 or bads_idx.sum() == 0:
        return

    pos = inst._get_channel_positions(picks)

    # Make sure only sEEG are used
    bads_idx_pos = bads_idx[picks]

    shafts = _find_seeg_electrode_shaft(pos, tol=tol)

    # interpolate the bad contacts
    picks_bad = list(np.where(bads_idx_pos)[0])
    for shaft in shafts:
        bads_shaft = np.array([idx for idx in picks_bad if idx in shaft])
        if bads_shaft.size == 0:
            continue
        goods_shaft = shaft[np.isin(shaft, bads_shaft, invert=True)]
        if goods_shaft.size < 2:
            raise RuntimeError(
                f"{goods_shaft.size} good contact(s) found in a line "
                f" with {np.array(inst.ch_names)[bads_shaft]}, "
                "at least 2 are required for interpolation. "
                "Dropping this channel/these channels is recommended."
            )
        logger.debug(
            f"Interpolating {np.array(inst.ch_names)[bads_shaft]} using "
            f"data from {np.array(inst.ch_names)[goods_shaft]}"
        )
        bads_shaft_idx = np.where(np.isin(shaft, bads_shaft))[0]
        goods_shaft_idx = np.where(~np.isin(shaft, bads_shaft))[0]
        n1, n2 = pos[shaft][:2]
        ts = np.array(
            [
                -np.dot(n1 - n0, n2 - n1) / np.linalg.norm(n2 - n1) ** 2
                for n0 in pos[shaft]
            ]
        )
        if np.any(np.diff(ts) < 0):
            ts *= -1
        y = np.arange(inst._data.shape[-1])
        inst._data[bads_shaft] = RectBivariateSpline(
            x=ts[goods_shaft_idx], y=y, z=inst._data[goods_shaft]
        )(x=ts[bads_shaft_idx], y=y)  # 3
