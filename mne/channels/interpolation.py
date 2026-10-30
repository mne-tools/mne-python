# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import pinv

from .._fiff.meas_info import Info, _simplify_info, create_info
from .._fiff.pick import pick_channels, pick_info, pick_types
from .._fiff.proj import _has_eeg_average_ref_proj, make_eeg_average_ref_proj
from ..bem import _check_origin
from ..surface import _normalize_vectors
from ..utils import (
    _check_option,
    _validate_type,
    _verbose_control,
    logger,
    warn,
)


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


def _make_interpolation_matrix_from_g(G_from, G_to_from, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines from G matrices."""
    n_from = G_from.shape[0]
    n_to = G_to_from.shape[0]
    G_from = G_from.copy()
    if alpha is not None:
        G_from.flat[:: n_from + 1] += alpha

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
    _normalize_vectors(pos_from)
    _normalize_vectors(pos_to)

    cosang_from = pos_from.dot(pos_from.T)
    cosang_to_from = pos_to.dot(pos_from.T)
    G_from = _calc_g(cosang_from)
    G_to_from = _calc_g(cosang_to_from)
    return _make_interpolation_matrix_from_g(G_from, G_to_from, alpha=alpha)


def _do_interp_dots(inst, interpolation, goods_idx, bads_idx):
    """Dot product of channel mapping matrix to channel data."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..io import BaseRaw

    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), "inst")
    inst._data[..., bads_idx, :] = np.matmul(
        interpolation, inst._data[..., goods_idx, :]
    )


class InterpolationData(dict):
    """Channel interpolation data and geometry.

    This dictionary contains precomputed bads-independent interpolation data
    keyed by channel type (e.g. ``'eeg'``, ``'meg'``).

    Parameters
    ----------
    eeg : dict | None
        Precomputed EEG interpolation geometry.
    meg : dict | None
        Precomputed MEG interpolation geometry.
    """

    def __init__(
        self,
        *,
        eeg=None,
        meg=None,
    ):
        super().__init__()
        if eeg is not None:
            self["eeg"] = eeg
        if meg is not None:
            self["meg"] = meg

    def __repr__(self):  # noqa: D105
        types = list(self.keys())
        if not types:
            return "<InterpolationData | empty>"
        info_strs = []
        for t in types:
            val = self[t]
            if isinstance(val, dict) and "ch_names" in val:
                info_strs.append(f"{t} ({len(val['ch_names'])} channels)")
            else:
                info_strs.append(str(t))
        return f"<InterpolationData | {', '.join(info_strs)}>"

    def copy(self):
        """Return a copy of the InterpolationData.

        Returns
        -------
        inst : instance of InterpolationData
            A copy of the interpolation data.
        """
        return deepcopy(self)


def _compute_spline_interpolation(info, origin, exclude=(), ecog=False):
    """Compute spherical spline interpolation matrix for EEG or ECoG."""
    if exclude is None:
        exclude = list()
    bads_idx = np.zeros(len(info["ch_names"]), dtype=bool)
    goods_idx = np.zeros(len(info["ch_names"]), dtype=bool)

    picks = pick_types(info, meg=False, eeg=not ecog, ecog=ecog, exclude=exclude)
    info._check_consistency()
    bads_idx[picks] = [info["ch_names"][ch] in info["bads"] for ch in picks]

    if len(picks) == 0 or bads_idx.sum() == 0:
        return None

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = info._get_channel_positions(picks)
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
    return goods_idx, bads_idx, interpolation


def _compute_eeg_interpolation_data(info, origin="auto"):
    origin = _check_origin(origin, info)
    picks = pick_types(info, meg=False, eeg=True, exclude=())
    if len(picks) == 0:
        return None
    ch_names = [info["ch_names"][p] for p in picks]
    pos = info._get_channel_positions(picks)
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1.0 - distance) > 0.1:
        warn(
            "Your spherical fit is poor, interpolation results are "
            "likely to be inaccurate."
        )
    pos_norm = (pos - origin).copy()
    _normalize_vectors(pos_norm)
    cosang = pos_norm.dot(pos_norm.T)
    G = _calc_g(cosang)
    return {
        "kind": "eeg",
        "G": G,
        "pos": pos,
        "origin": origin,
        "ch_names": ch_names,
    }


def _compute_meg_interpolation_data(info, origin="auto", mode="accurate"):
    from ..forward._field_interpolation import _setup_dots
    from ..forward._lead_dots import _do_self_dots
    from ..forward._make_forward import _create_meg_coils, _read_coil_defs

    picks = pick_types(info, meg=True, ref_meg=False, exclude=())
    if len(picks) == 0:
        return None
    ch_names = [info["ch_names"][p] for p in picks]
    origin = _check_origin(origin, info)
    chs = [info["chs"][p] for p in picks]
    templates = _read_coil_defs(verbose=False)
    coils = _create_meg_coils(chs, "normal", info["dev_head_t"], templates)
    int_rad, noise, leg_fun, n_fact = _setup_dots(mode, info, coils, "meg")
    dots = _do_self_dots(
        int_rad, False, coils, origin, "meg", leg_fun, n_fact, n_jobs=None
    )
    return {
        "kind": "meg",
        "dots": dots,
        "dev_head_t": deepcopy(info["dev_head_t"]),
        "origin": origin,
        "mode": mode,
        "ch_names": ch_names,
        "int_rad": int_rad,
    }


@_verbose_control
def compute_interpolation_data(
    inst_or_info,
    *,
    ch_types=("eeg", "meg"),
    origin="auto",
    mode="accurate",
    verbose=None,
):
    """Precompute bads-independent interpolation data for EEG and/or MEG.

    Parameters
    ----------
    inst_or_info : instance of Raw | Epochs | Evoked | Info
        The data instance or Info containing channel positions.
    ch_types : str | tuple of str
        Channel type(s) to compute interpolation data for. Can be ``'eeg'``,
        ``'meg'``, or a sequence containing any combination of them.
        Defaults to ``('eeg', 'meg')``. Only channel types present in the
        provided instance/info will be computed.
    origin : array-like, shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'`` (default), which means a head-digitization-based
        origin fit.
    mode : str
        Either ``'accurate'`` (default) or ``'fast'``, determines the quality
        of the Legendre polynomial expansion used for MEG.
    verbose : bool | str | int | None
        Control verbosity of the logging output.

    Returns
    -------
    interpolation_data : instance of InterpolationData
        Dictionary containing precomputed interpolation data keyed by channel type
        (e.g., ``'eeg'`` and/or ``'meg'``). Can be passed to
        :meth:`~mne.io.Raw.interpolate_bads`, :meth:`~mne.Epochs.interpolate_bads`,
        or :meth:`~mne.Evoked.interpolate_bads` via ``interpolation=...``.

    Notes
    -----
    For EEG, this precomputes the full spherical spline :math:`G` matrix evaluated
    via Legendre polynomials between all EEG sensor positions.
    For MEG, this precomputes the coil lead field dot products matrix across all
    MEG channels.
    Because these precomputations depend only on channel geometry and not on
    the set of bad channels, they can be reused across different sets of bad
    channels (such as in leave-one-out cross-validation or per-epoch artifact
    rejection) with minimal computation per iteration.

    .. versionadded:: 1.14
    """
    info = getattr(inst_or_info, "info", inst_or_info)
    _validate_type(info, (Info,), "inst_or_info", "Raw, Epochs, Evoked, or Info")

    if isinstance(ch_types, str):
        ch_types = (ch_types,)
    for ct in ch_types:
        _check_option("ch_types", ct, ("eeg", "meg"))

    data = InterpolationData()
    has_eeg = len(pick_types(info, meg=False, eeg=True, exclude=())) > 0
    has_meg = len(pick_types(info, meg=True, ref_meg=False, exclude=())) > 0

    for ct in ch_types:
        if ct == "eeg":
            if has_eeg:
                data["eeg"] = _compute_eeg_interpolation_data(info, origin=origin)
            elif len(ch_types) == 1:
                raise RuntimeError("No EEG channels found in info.")
        elif ct == "meg":
            if has_meg:
                data["meg"] = _compute_meg_interpolation_data(
                    info, origin=origin, mode=mode
                )
            elif len(ch_types) == 1:
                raise RuntimeError("No MEG channels found in info.")

    if not data:
        warn("No EEG or MEG channels found to compute interpolation data.")
    return data


def _interpolate_bads_eeg_from_data(inst, interp_data, exclude=()):
    picks_type = pick_types(inst.info, meg=False, eeg=True, exclude=())
    curr_names = [inst.info["ch_names"][p] for p in picks_type]
    if (
        interp_data.get("ch_names") is not None
        and interp_data["ch_names"] != curr_names
    ):
        raise ValueError("Channel names do not match precomputed data for eeg.")
    if interp_data.get("pos") is not None:
        curr_pos = inst.info._get_channel_positions(picks_type)
        if not np.allclose(curr_pos, interp_data["pos"], atol=1e-5):
            raise ValueError("Channel positions do not match precomputed data for eeg.")

    bads_type = [
        ch for ch in inst.info["bads"] if ch in curr_names and ch not in exclude
    ]
    if not bads_type:
        return set()
    goods_type = [
        ch for ch in curr_names if ch not in inst.info["bads"] and ch not in exclude
    ]
    if not goods_type:
        raise RuntimeError("No good EEG channels available for interpolation.")

    idx_from = [curr_names.index(ch) for ch in goods_type]
    idx_to = [curr_names.index(ch) for ch in bads_type]
    G = interp_data["G"]
    G_from = G[idx_from, :][:, idx_from]
    G_to_from = G[idx_to, :][:, idx_from]
    mapping = _make_interpolation_matrix_from_g(G_from, G_to_from)

    picks_good = pick_channels(inst.info["ch_names"], goods_type)
    picks_bad = pick_channels(inst.info["ch_names"], bads_type)
    _do_interp_dots(inst, mapping, picks_good, picks_bad)
    return set(bads_type)


def _interpolate_bads_meg_from_data(inst, interp_data, exclude=()):
    from ..forward._field_interpolation import (
        _compute_mapping_matrix,
        _make_field_mapping_noise,
    )

    picks_type = pick_types(inst.info, meg=True, ref_meg=False, exclude=())
    curr_names = [inst.info["ch_names"][p] for p in picks_type]
    if (
        interp_data.get("ch_names") is not None
        and interp_data["ch_names"] != curr_names
    ):
        raise ValueError("Channel names do not match precomputed data for meg.")
    if interp_data.get("dev_head_t") is not None:
        if not np.allclose(
            inst.info["dev_head_t"]["trans"],
            interp_data["dev_head_t"]["trans"],
            atol=1e-5,
        ):
            raise ValueError("dev_head_t does not match precomputed data for meg.")

    bads_type = [
        ch for ch in inst.info["bads"] if ch in curr_names and ch not in exclude
    ]
    if not bads_type:
        return set()
    goods_type = [
        ch for ch in curr_names if ch not in inst.info["bads"] and ch not in exclude
    ]
    if not goods_type:
        raise RuntimeError("No good MEG channels available for interpolation.")

    idx_from = [curr_names.index(ch) for ch in goods_type]
    idx_to = [curr_names.index(ch) for ch in bads_type]
    dots = interp_data["dots"]
    self_dots = dots[idx_from, :][:, idx_from]
    surface_dots = dots[idx_to, :][:, idx_from]

    picks_good = pick_channels(inst.info["ch_names"], goods_type)
    picks_bad = pick_channels(inst.info["ch_names"], bads_type)
    info_from = pick_info(inst.info, picks_good)

    fmd = dict(
        kind="meg",
        ch_names=goods_type,
        origin=interp_data["origin"],
        noise=_make_field_mapping_noise(info_from),
        self_dots=self_dots,
        surface_dots=surface_dots,
        int_rad=interp_data.get("int_rad", 0.06),
        miss=1e-4,
        pinv_method="tsvd",
    )
    mapping = _compute_mapping_matrix(fmd, info_from)
    _do_interp_dots(inst, mapping, picks_good, picks_bad)
    return set(bads_type)


@_verbose_control
def _interpolate_bads_eeg(inst, origin, exclude=None, ecog=False, verbose=None):
    res = _compute_spline_interpolation(inst.info, origin, exclude=exclude, ecog=ecog)
    if res is None:
        return
    goods_idx, bads_idx, interpolation = res
    logger.info(f"Interpolating {bads_idx.sum()} sensors")
    _do_interp_dots(inst, interpolation, goods_idx, bads_idx)


@_verbose_control
def _interpolate_bads_ecog(inst, *, origin, exclude=None, verbose=None):
    _interpolate_bads_eeg(inst, origin, exclude=exclude, ecog=True, verbose=verbose)


def _interpolate_bads_meg(
    inst, mode="accurate", *, origin, verbose=None, ref_meg=False
):
    return _interpolate_bads_meeg(
        inst, mode, ref_meg=ref_meg, eeg=False, origin=origin, verbose=verbose
    )


@_verbose_control
def _interpolate_bads_nan(
    inst,
    *,
    ch_type,
    ref_meg=False,
    exclude=(),
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


@_verbose_control
def _interpolate_bads_meeg(
    inst,
    mode="accurate",
    *,
    meg=True,
    eeg=True,
    ref_meg=False,
    exclude=(),
    origin,
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


@_verbose_control
def _interpolate_bads_nirs(inst, exclude=(), verbose=None):
    from scipy.spatial.distance import pdist, squareform

    from ..preprocessing.nirs import _validate_nirs_info

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
    bads_mask = np.isin(picks_nirs, picks_bad)

    chs = [inst.info["chs"][i] for i in picks_nirs]
    locs3d = np.array([ch["loc"][:3] for ch in chs])
    invalid_mask = np.isnan(locs3d).any(axis=1) | (np.abs(locs3d) <= 1e-16).all(axis=1)
    if np.any(bads_mask & ~invalid_mask) and not np.any(~bads_mask & ~invalid_mask):
        raise ValueError(
            "No good fNIRS channel with a valid position to interpolate from."
        )

    dist = pdist(locs3d)
    dist = squareform(dist)

    for bad_raw_idx in picks_bad:
        # `bad_raw_idx` is the index of the bad channel in `inst`
        # `bad_dist_idx` is the index of the bad channel in `dist`
        bad_dist_idx = np.where(picks_nirs == bad_raw_idx)[0][0]
        if invalid_mask[bad_dist_idx]:
            inst._data[..., bad_raw_idx, :] = np.nan
            continue
        dists_to_bad = dist[bad_dist_idx].copy()
        # Ignore distances to self
        dists_to_bad[dists_to_bad == 0] = np.inf
        # Ignore distances to other bad channels
        dists_to_bad[bads_mask] = np.inf
        # Ignore channels without valid positions
        dists_to_bad[invalid_mask] = np.inf
        # Find closest remaining channels for same frequency
        closest_dist_idx = np.argmin(dists_to_bad) + (bad_dist_idx % 2)
        closest_raw_idx = picks_nirs[closest_dist_idx]
        inst._data[..., bad_raw_idx, :] = inst._data[..., closest_raw_idx, :]

    return inst


def _find_seeg_electrode_shaft(pos, tol_shaft=0.002, tol_spacing=1):
    # 1) find nearest neighbor to define the electrode shaft line
    # 2) find all contacts on the same line
    # 3) remove contacts with large distances

    from scipy.spatial.distance import pdist, squareform

    dist = squareform(pdist(pos))
    np.fill_diagonal(dist, np.inf)

    shafts = list()
    shaft_ts = list()
    for i, n1 in enumerate(pos):
        if any([i in shaft for shaft in shafts]):
            continue
        n2 = pos[np.argmin(dist[i])]  # 1
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        shaft_dists = np.linalg.norm(
            np.cross((pos - n1), (pos - n2)), axis=1
        ) / np.linalg.norm(n2 - n1)
        shaft = np.where(shaft_dists < tol_shaft)[0]  # 2
        shaft_prev = None
        for _ in range(10):  # avoid potential cycles
            if np.array_equal(shaft, shaft_prev):
                break
            shaft_prev = shaft
            # compute median shaft line
            v = np.median(
                [
                    pos[i] - pos[j]
                    for idx, i in enumerate(shaft)
                    for j in shaft[idx + 1 :]
                ],
                axis=0,
            )
            c = np.median(pos[shaft], axis=0)
            # recompute distances
            shaft_dists = np.linalg.norm(
                np.cross((pos - c), (pos - c + v)), axis=1
            ) / np.linalg.norm(v)
            shaft = np.where(shaft_dists < tol_shaft)[0]
        ts = np.array([np.dot(c - n0, v) / np.linalg.norm(v) ** 2 for n0 in pos[shaft]])
        shaft_order = np.argsort(ts)
        shaft = shaft[shaft_order]
        ts = ts[shaft_order]

        # only include the largest group with spacing with the error tolerance
        # avoid interpolating across spans between contacts
        t_diffs = np.diff(ts)
        t_diff_med = np.median(t_diffs)
        spacing_errors = (t_diffs - t_diff_med) / t_diff_med
        groups = list()
        group = [shaft[0]]
        for j in range(len(shaft) - 1):
            if spacing_errors[j] > tol_spacing:
                groups.append(group)
                group = [shaft[j + 1]]
            else:
                group.append(shaft[j + 1])
        groups.append(group)
        group = [group for group in groups if i in group][0]
        ts = ts[np.isin(shaft, group)]
        shaft = np.array(group, dtype=int)

        shafts.append(shaft)
        shaft_ts.append(ts)
    return shafts, shaft_ts


@_verbose_control
def _interpolate_bads_seeg(
    inst, exclude=None, tol_shaft=0.002, tol_spacing=1, verbose=None
):
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

    shafts, shaft_ts = _find_seeg_electrode_shaft(
        pos, tol_shaft=tol_shaft, tol_spacing=tol_spacing
    )

    # interpolate the bad contacts
    picks_bad = list(np.where(bads_idx_pos)[0])
    for shaft, ts in zip(shafts, shaft_ts):
        bads_shaft = np.array([idx for idx in picks_bad if idx in shaft])
        if bads_shaft.size == 0:
            continue
        goods_shaft = shaft[np.isin(shaft, bads_shaft, invert=True)]
        if goods_shaft.size < 4:  # cubic spline requires 3 channels
            msg = "No shaft" if shaft.size < 4 else "Not enough good channels"
            no_shaft_chs = " and ".join(np.array(inst.ch_names)[bads_shaft])
            raise RuntimeError(
                f"{msg} found in a line with {no_shaft_chs} "
                "at least 3 good channels on the same line "
                f"are required for interpolation, {goods_shaft.size} found. "
                f"Dropping {no_shaft_chs} is recommended."
            )
        logger.debug(
            f"Interpolating {np.array(inst.ch_names)[bads_shaft]} using "
            f"data from {np.array(inst.ch_names)[goods_shaft]}"
        )
        bads_shaft_idx = np.where(np.isin(shaft, bads_shaft))[0]
        goods_shaft_idx = np.where(~np.isin(shaft, bads_shaft))[0]

        z = inst._data[..., goods_shaft, :]
        is_epochs = z.ndim == 3
        if is_epochs:
            z = z.swapaxes(0, 1)
            z = z.reshape(z.shape[0], -1)
        y = np.arange(z.shape[-1])
        out = RectBivariateSpline(x=ts[goods_shaft_idx], y=y, z=z)(
            x=ts[bads_shaft_idx], y=y
        )
        if is_epochs:
            out = out.reshape(bads_shaft.size, inst._data.shape[0], -1)
            out = out.swapaxes(0, 1)
        inst._data[..., bads_shaft, :] = out


def _interpolate_to_eeg(inst, sensors, origin, method, reg):
    """Interpolate EEG data to a new montage."""
    from ..forward._field_interpolation import _map_meg_or_eeg_channels

    # Get target positions from the montage
    ch_pos = sensors.get_positions().get("ch_pos", {})
    target_ch_names = list(ch_pos)
    if not target_ch_names:
        raise ValueError("The provided sensors configuration has no channel positions.")

    # Identify EEG channel
    picks_good_eeg = pick_types(inst.info, eeg=True, exclude="bads")
    if len(picks_good_eeg) == 0:
        raise ValueError("No good EEG channels available for interpolation.")

    # Create destination info for new EEG channels
    # TODO: Maybe copy? This will remove potentially useful metadata...
    info_to = create_info(target_ch_names, sfreq=inst.info["sfreq"], ch_types="eeg")
    info_to.set_montage(sensors)

    # Compute the interpolation mapping
    if method == "spline":
        origin_val = _check_origin(origin, inst.info)
        pos_from = inst.info._get_channel_positions(picks_good_eeg) - origin_val
        # Use info_to (rather than ch_pos directly) so that the target positions
        # are in the head frame, and center both sets on the fitted origin
        pos_to = info_to._get_channel_positions() - origin_val

        def _check_pos_sphere(pos):
            d = np.linalg.norm(pos, axis=-1)
            d_norm = np.mean(d / np.mean(d))
            if np.abs(1.0 - d_norm) > 0.1:
                warn("Your spherical fit is poor; interpolation may be inaccurate.")

        _check_pos_sphere(pos_from)
        _check_pos_sphere(pos_to)
        mapping = _make_interpolation_matrix(pos_from, pos_to, alpha=reg)

    else:
        assert method == "MNE"
        info_eeg = pick_info(inst.info, picks_good_eeg)
        # If the original info has an average EEG reference projector but
        # the destination info does not, update info_interp via a temporary RawArray.
        if _has_eeg_average_ref_proj(inst.info) and not _has_eeg_average_ref_proj(
            info_to
        ):
            # add an average reference projector.
            info_to["projs"].append(make_eeg_average_ref_proj(info_to))
        mapping = _map_meg_or_eeg_channels(
            info_eeg, info_to, mode="accurate", origin=origin
        )

    return _remap_add(inst, mapping, info_to, ch_type="eeg")


def _interpolate_to_meg(inst, sensors, origin, mode):
    """Interpolate MEG data to a canonical sensor configuration."""
    from ..forward._field_interpolation import _map_meg_or_eeg_channels
    from .montage import read_meg_canonical_info

    # Get MEG channels from source
    picks_meg_good = pick_types(inst.info, meg=True, ref_meg=False, exclude="bads")
    if len(picks_meg_good) == 0:
        raise ValueError("No good MEG channels available for interpolation.")

    # Load target sensor configuration as info file
    info_cano = read_meg_canonical_info(sensors)

    # Get source MEG info
    info_from = pick_info(inst.info, picks_meg_good)

    # Update target info to accommodate the desired channel info
    # and reset some channel and machine-related fields to avoid
    # confusion later on.
    # NOTE: We don't change the original 'dev_head_t'.
    #       Some keys require as default an empty list.
    info_to = deepcopy(info_from)
    with info_to._unlock():
        info_to.update(
            {
                "chs": info_cano["chs"],
                "ch_names": info_cano["ch_names"],
                "nchan": info_cano["nchan"],
                "device_info": None,
                "helium_info": None,
                "gantry_angle": None,
                "ctf_head_t": None,
                "dev_ctf_t": None,
                "bads": [],
                "projs": [],
                "comps": [],
            }
        )

    # Compute field interpolation mapping
    origin_val = _check_origin(origin, inst.info)
    mapping = _map_meg_or_eeg_channels(info_from, info_to, mode=mode, origin=origin_val)
    return _remap_add(inst, mapping, info_to, ch_type="meg")


def _remap_add(inst, mapping, info_to, ch_type):
    # Comments here refer to EEG, but in principle it could instead be MEG!
    assert ch_type in ("eeg", "meg")
    from ..epochs import BaseEpochs, EpochsArray
    from ..evoked import Evoked, EvokedArray
    from ..io import RawArray
    from ..io.base import BaseRaw

    # Get original channel order
    orig_names = inst.info["ch_names"]

    # Get the full list of EEG channel indices (including bad channels)
    pick_kwargs = {ch_type: True}
    picks_interp_all = pick_types(inst.info, exclude=[], **pick_kwargs)
    picks_interp_good = pick_types(inst.info, exclude="bads", **pick_kwargs)
    interp_names_orig = [orig_names[i] for i in picks_interp_all]

    # Interpolate EEG data
    data_interp = mapping @ inst.get_data(picks=picks_interp_good)

    # Create a new instance for the interpolated EEG channels
    if isinstance(inst, BaseRaw):
        inst_interp = RawArray(data_interp, info_to, first_samp=inst.first_samp)
    elif isinstance(inst, BaseEpochs):
        inst_interp = EpochsArray(data_interp, info_to, tmin=inst.tmin)
    else:
        assert isinstance(inst, Evoked)
        inst_interp = EvokedArray(data_interp, info_to, tmin=inst.tmin)

    # Identify non-EEG channels in original order
    non_interp_names_ordered = [ch for ch in orig_names if ch not in interp_names_orig]

    # Merge only if non-EEG channels exist
    if not non_interp_names_ordered:
        return inst_interp

    inst_non_interp = inst.copy().pick(non_interp_names_ordered).load_data()
    inst_out = inst_non_interp.add_channels([inst_interp], force_update_info=True)

    # Reorder channels
    # Insert the entire new EEG block at the position of the first EEG channel.
    orig_names_arr = np.array(orig_names)
    mask_interp = np.isin(orig_names_arr, interp_names_orig)
    if mask_interp.any():
        first_interp_index = np.where(mask_interp)[0][0]
        pre = orig_names_arr[:first_interp_index]
        new_interp = np.array(info_to["ch_names"])
        post = orig_names_arr[first_interp_index:]
        post = post[~np.isin(orig_names_arr[first_interp_index:], interp_names_orig)]
        new_order = np.concatenate((pre, new_interp, post)).tolist()
    else:
        new_order = orig_names
    inst_out.reorder_channels(new_order)
    return inst_out
