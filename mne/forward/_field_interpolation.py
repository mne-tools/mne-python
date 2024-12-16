# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The computations in this code were primarily derived from Matti Hämäläinen's
# C code.

import inspect
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d

from .._fiff.constants import FIFF
from .._fiff.meas_info import _simplify_info
from .._fiff.pick import pick_info, pick_types
from .._fiff.proj import _has_eeg_average_ref_proj, make_projector
from ..bem import _check_origin
from ..cov import make_ad_hoc_cov
from ..epochs import BaseEpochs, EpochsArray
from ..evoked import Evoked, EvokedArray
from ..fixes import _safe_svd
from ..surface import get_head_surf, get_meg_helmet_surf
from ..transforms import _find_trans, _get_trans, transform_surface_to
from ..utils import _check_fname, _check_option, _pl, _reg_pinv, logger, verbose
from ._lead_dots import (
    _do_cross_dots,
    _do_self_dots,
    _do_surface_dots,
    _get_legen_table,
)
from ._make_forward import _create_eeg_els, _create_meg_coils, _read_coil_defs


def _setup_dots(mode, info, coils, ch_type):
    """Set up dot products."""
    int_rad = 0.06
    noise = make_ad_hoc_cov(info, dict(mag=20e-15, grad=5e-13, eeg=1e-6))
    n_coeff, interp = (50, "nearest") if mode == "fast" else (100, "linear")
    lut, n_fact = _get_legen_table(ch_type, False, n_coeff, verbose=False)
    lut_fun = interp1d(np.linspace(-1, 1, lut.shape[0]), lut, interp, axis=0)
    return int_rad, noise, lut_fun, n_fact


def _compute_mapping_matrix(fmd, info):
    """Do the hairy computations."""
    logger.info("    Preparing the mapping matrix...")
    # assemble a projector and apply it to the data
    ch_names = fmd["ch_names"]
    projs = info.get("projs", list())
    proj_op = make_projector(projs, ch_names)[0]
    proj_dots = np.dot(proj_op.T, np.dot(fmd["self_dots"], proj_op))

    noise_cov = fmd["noise"]
    # Whiten
    if not noise_cov["diag"]:
        raise NotImplementedError  # this shouldn't happen
    whitener = np.diag(1.0 / np.sqrt(noise_cov["data"].ravel()))
    whitened_dots = np.dot(whitener.T, np.dot(proj_dots, whitener))

    # SVD is numerically better than the eigenvalue composition even if
    # mat is supposed to be symmetric and positive definite
    if fmd.get("pinv_method", "tsvd") == "tsvd":
        inv, fmd["nest"] = _pinv_trunc(whitened_dots, fmd["miss"])
    else:
        assert fmd["pinv_method"] == "tikhonov", fmd["pinv_method"]
        inv, fmd["nest"] = _pinv_tikhonov(whitened_dots, fmd["miss"])

    # Sandwich with the whitener
    inv_whitened = np.dot(whitener.T, np.dot(inv, whitener))

    # Take into account that the lead fields used to compute
    # d->surface_dots were unprojected
    inv_whitened_proj = proj_op.T @ inv_whitened

    # Finally sandwich in the selection matrix
    # This one picks up the correct lead field projection
    mapping_mat = np.dot(fmd["surface_dots"], inv_whitened_proj)

    # Optionally apply the average electrode reference to the final field map
    if fmd["kind"] == "eeg" and _has_eeg_average_ref_proj(info):
        logger.info(
            "    The map has an average electrode reference "
            f"({mapping_mat.shape[0]} channels)"
        )
        mapping_mat -= np.mean(mapping_mat, axis=0)
    return mapping_mat


def _pinv_trunc(x, miss):
    """Compute pseudoinverse, truncating at most "miss" fraction of varexp."""
    u, s, v = _safe_svd(x, full_matrices=False)

    # Eigenvalue truncation
    varexp = np.cumsum(s)
    varexp /= varexp[-1]
    n = np.where(varexp >= (1.0 - miss))[0][0] + 1
    logger.info(
        "    Truncating at %d/%d components to omit less than %g " "(%0.2g)",
        n,
        len(s),
        miss,
        1.0 - varexp[n - 1],
    )
    s = 1.0 / s[:n]
    inv = ((u[:, :n] * s) @ v[:n]).T
    return inv, n


def _pinv_tikhonov(x, reg):
    # _reg_pinv requires square Hermitian, which we have here
    inv, _, n = _reg_pinv(x, reg=reg, rank=None)
    logger.info(
        f"    Truncating at {n}/{len(x)} components and regularizing "
        f"with α={reg:0.1e}"
    )
    return inv, n


def _map_meg_or_eeg_channels(info_from, info_to, mode, origin, miss=None):
    """Find mapping from one set of channels to another.

    Parameters
    ----------
    info_from : instance of Info
        The measurement data to interpolate from.
    info_to : instance of Info
        The measurement info to interpolate to.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.
    origin : array-like, shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'``, which means a head-digitization-based origin
        fit. Default is ``(0., 0., 0.04)``.

    Returns
    -------
    mapping : array, shape (n_to, n_from)
        A mapping matrix.
    """
    # no need to apply trans because both from and to coils are in device
    # coordinates
    info_kinds = set(ch["kind"] for ch in info_to["chs"])
    info_kinds |= set(ch["kind"] for ch in info_from["chs"])
    if FIFF.FIFFV_REF_MEG_CH in info_kinds:  # refs same as MEG
        info_kinds |= set([FIFF.FIFFV_MEG_CH])
        info_kinds -= set([FIFF.FIFFV_REF_MEG_CH])
    info_kinds = sorted(info_kinds)
    # This should be guaranteed by the callers
    assert len(info_kinds) == 1 and info_kinds[0] in (
        FIFF.FIFFV_MEG_CH,
        FIFF.FIFFV_EEG_CH,
    )
    kind = "eeg" if info_kinds[0] == FIFF.FIFFV_EEG_CH else "meg"

    #
    # Step 1. Prepare the coil definitions
    #
    if kind == "meg":
        templates = _read_coil_defs(verbose=False)
        coils_from = _create_meg_coils(
            info_from["chs"], "normal", info_from["dev_head_t"], templates
        )
        coils_to = _create_meg_coils(
            info_to["chs"], "normal", info_to["dev_head_t"], templates
        )
        pinv_method = "tsvd"
        miss = 1e-4
    else:
        coils_from = _create_eeg_els(info_from["chs"])
        coils_to = _create_eeg_els(info_to["chs"])
        pinv_method = "tikhonov"
        miss = 1e-1
        if _has_eeg_average_ref_proj(info_from) and not _has_eeg_average_ref_proj(
            info_to
        ):
            raise RuntimeError(
                "info_to must have an average EEG reference projector if "
                "info_from has one"
            )
    origin = _check_origin(origin, info_from)
    #
    # Step 2. Calculate the dot products
    #
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, info_from, coils_from, kind)
    logger.info(
        f"    Computing dot products for {len(coils_from)} "
        f"{kind.upper()} channel{_pl(coils_from)}..."
    )
    self_dots = _do_self_dots(
        int_rad, False, coils_from, origin, kind, lut_fun, n_fact, n_jobs=None
    )
    logger.info(
        f"    Computing cross products for {len(coils_from)} → "
        f"{len(coils_to)} {kind.upper()} channel{_pl(coils_to)}..."
    )
    cross_dots = _do_cross_dots(
        int_rad, False, coils_from, coils_to, origin, kind, lut_fun, n_fact
    ).T

    ch_names = [c["ch_name"] for c in info_from["chs"]]
    fmd = dict(
        kind=kind,
        ch_names=ch_names,
        origin=origin,
        noise=noise,
        self_dots=self_dots,
        surface_dots=cross_dots,
        int_rad=int_rad,
        miss=miss,
        pinv_method=pinv_method,
    )

    #
    # Step 3. Compute the mapping matrix
    #
    mapping = _compute_mapping_matrix(fmd, info_from)
    return mapping


def _as_meg_type_inst(inst, ch_type="grad", mode="fast"):
    """Compute virtual evoked using interpolated fields in mag/grad channels.

    Parameters
    ----------
    inst : instance of mne.Evoked or mne.Epochs
        The evoked or epochs object.
    ch_type : str
        The destination channel type. It can be 'mag' or 'grad'.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.

    Returns
    -------
    inst : instance of mne.EvokedArray or mne.EpochsArray
        The transformed evoked object containing only virtual channels.
    """
    _check_option("ch_type", ch_type, ["mag", "grad"])

    # pick the original and destination channels
    pick_from = pick_types(inst.info, meg=True, eeg=False, ref_meg=False)
    pick_to = pick_types(inst.info, meg=ch_type, eeg=False, ref_meg=False)

    if len(pick_to) == 0:
        raise ValueError(
            "No channels matching the destination channel type"
            " found in info. Please pass an evoked containing"
            "both the original and destination channels. Only the"
            " locations of the destination channels will be used"
            " for interpolation."
        )

    info_from = pick_info(inst.info, pick_from)
    info_to = pick_info(inst.info, pick_to)
    # XXX someday we should probably expose the origin
    mapping = _map_meg_or_eeg_channels(
        info_from, info_to, origin=(0.0, 0.0, 0.04), mode=mode
    )

    # compute data by multiplying by the 'gain matrix' from
    # original sensors to virtual sensors
    if hasattr(inst, "get_data"):
        kwargs = dict()
        if "copy" in inspect.getfullargspec(inst.get_data).kwonlyargs:
            kwargs["copy"] = False
        data = inst.get_data(**kwargs)
    else:
        data = inst.data

    ndim = data.ndim
    if ndim == 2:
        data = data[np.newaxis, :, :]

    data_ = np.empty((data.shape[0], len(mapping), data.shape[2]), dtype=data.dtype)
    for d, d_ in zip(data, data_):
        d_[:] = np.dot(mapping, d[pick_from])

    # keep only the destination channel types
    info = pick_info(inst.info, sel=pick_to, copy=True)

    # change channel names to emphasize they contain interpolated data
    for ch in info["chs"]:
        ch["ch_name"] += "_v"
    info._update_redundant()
    info._check_consistency()
    if isinstance(inst, Evoked):
        assert ndim == 2
        data_ = data_[0]  # undo new axis
        inst_ = EvokedArray(
            data_, info, tmin=inst.times[0], comment=inst.comment, nave=inst.nave
        )
    else:
        assert isinstance(inst, BaseEpochs)
        inst_ = EpochsArray(
            data_,
            info,
            tmin=inst.tmin,
            events=inst.events,
            event_id=inst.event_id,
            metadata=inst.metadata,
        )

    return inst_


@verbose
def _make_surface_mapping(
    info,
    surf,
    ch_type="meg",
    trans=None,
    mode="fast",
    n_jobs=None,
    origin=(0.0, 0.0, 0.04),
    verbose=None,
):
    """Re-map M/EEG data to a surface.

    Parameters
    ----------
    %(info_not_none)s
    surf : dict
        The surface to map the data to. The required fields are `'rr'`,
        `'nn'`, and `'coord_frame'`. Must be in head coordinates.
    ch_type : str
        Must be either `'meg'` or `'eeg'`, determines the type of field.
    trans : None | dict
        If None, no transformation applied. Should be a Head<->MRI
        transformation.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.
    %(n_jobs)s
    origin : array-like, shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        The default is ``'auto'``, which means a head-digitization-based
        origin fit.
    %(verbose)s

    Returns
    -------
    mapping : array
        A n_vertices x n_sensors array that remaps the MEG or EEG data,
        as `new_data = np.dot(mapping, data)`.
    """
    if not all(key in surf for key in ["rr", "nn"]):
        raise KeyError('surf must have both "rr" and "nn"')
    if "coord_frame" not in surf:
        raise KeyError(
            'The surface coordinate frame must be specified in surf["coord_frame"]'
        )
    _check_option("mode", mode, ["accurate", "fast"])

    # deal with coordinate frames here -- always go to "head" (easiest)
    orig_surf = surf
    surf = transform_surface_to(deepcopy(surf), "head", trans)
    origin = _check_origin(origin, info)

    #
    # Step 1. Prepare the coil definitions
    # Do the dot products, assume surf in head coords
    #
    _check_option("ch_type", ch_type, ["meg", "eeg"])
    if ch_type == "meg":
        picks = pick_types(info, meg=True, eeg=False, ref_meg=False)
        logger.info("Prepare MEG mapping...")
    else:
        picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
        logger.info("Prepare EEG mapping...")
    if len(picks) == 0:
        raise RuntimeError("cannot map, no channels found")
    # XXX this code does not do any checking for compensation channels,
    # but it seems like this must be intentional from the ref_meg=False
    # (presumably from the C code)
    dev_head_t = info["dev_head_t"]
    info = pick_info(_simplify_info(info), picks)
    info["dev_head_t"] = dev_head_t

    # create coil defs in head coordinates
    if ch_type == "meg":
        # Put them in head coordinates
        coils = _create_meg_coils(info["chs"], "normal", info["dev_head_t"])
        type_str = "coils"
        miss = 1e-4  # Smoothing criterion for MEG
    else:  # EEG
        coils = _create_eeg_els(info["chs"])
        type_str = "electrodes"
        miss = 1e-3  # Smoothing criterion for EEG

    #
    # Step 2. Calculate the dot products
    #
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, info, coils, ch_type)
    logger.info("Computing dot products for %i %s...", len(coils), type_str)
    self_dots = _do_self_dots(
        int_rad, False, coils, origin, ch_type, lut_fun, n_fact, n_jobs
    )
    sel = np.arange(len(surf["rr"]))  # eventually we should do sub-selection
    logger.info("Computing dot products for %i surface locations...", len(sel))
    surface_dots = _do_surface_dots(
        int_rad, False, coils, surf, sel, origin, ch_type, lut_fun, n_fact, n_jobs
    )

    #
    # Step 4. Return the result
    #
    fmd = dict(
        kind=ch_type,
        surf=surf,
        ch_names=info["ch_names"],
        coils=coils,
        origin=origin,
        noise=noise,
        self_dots=self_dots,
        surface_dots=surface_dots,
        int_rad=int_rad,
        miss=miss,
    )
    logger.info("Field mapping data ready")

    fmd["data"] = _compute_mapping_matrix(fmd, info)
    # bring the original back, whatever coord frame it was in
    fmd["surf"] = orig_surf

    # Remove some unnecessary fields
    del fmd["self_dots"]
    del fmd["surface_dots"]
    del fmd["int_rad"]
    del fmd["miss"]
    return fmd


@verbose
def make_field_map(
    evoked,
    trans="auto",
    subject=None,
    subjects_dir=None,
    ch_type=None,
    mode="fast",
    meg_surf="helmet",
    origin=(0.0, 0.0, 0.04),
    n_jobs=None,
    *,
    head_source=("bem", "head"),
    verbose=None,
):
    """Compute surface maps used for field display in 3D.

    Parameters
    ----------
    evoked : Evoked | Epochs | Raw
        The measurement file. Need to have info attribute.
    %(trans)s ``"auto"`` (default) will load trans from the FreeSurfer
        directory specified by ``subject`` and ``subjects_dir`` parameters.

        .. versionchanged:: 0.19
            Support for ``'fsaverage'`` argument.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None, map for EEG data will not be available.
    subjects_dir : path-like
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    ch_type : None | ``'eeg'`` | ``'meg'``
        If None, a map for each available channel type will be returned.
        Else only the specified type will be used.
    mode : ``'accurate'`` | ``'fast'``
        Either ``'accurate'`` or ``'fast'``, determines the quality of the
        Legendre polynomial expansion used. ``'fast'`` should be sufficient
        for most applications.
    meg_surf : 'helmet' | 'head'
        Should be ``'helmet'`` or ``'head'`` to specify in which surface
        to compute the MEG field map. The default value is ``'helmet'``.
    origin : array-like, shape (3,) | 'auto'
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'``, which means a head-digitization-based origin
        fit. Default is ``(0., 0., 0.04)``.

        .. versionadded:: 0.11
    %(n_jobs)s
    %(head_source)s

        .. versionadded:: 1.1
    %(verbose)s

    Returns
    -------
    surf_maps : list
        The surface maps to be used for field plots. The list contains
        separate ones for MEG and EEG (if both MEG and EEG are present).
    """
    info = evoked.info

    if ch_type is None:
        types = [t for t in ["eeg", "meg"] if t in evoked]
    else:
        _check_option("ch_type", ch_type, ["eeg", "meg"])
        types = [ch_type]

    if subjects_dir is not None:
        subjects_dir = _check_fname(
            subjects_dir,
            overwrite="read",
            must_exist=True,
            name="subjects_dir",
            need_dir=True,
        )
    if isinstance(trans, str) and trans == "auto":
        # let's try to do this in MRI coordinates so they're easy to plot
        trans = _find_trans(subject, subjects_dir)
    trans, trans_type = _get_trans(trans, fro="head", to="mri")

    if "eeg" in types and trans_type == "identity":
        logger.info("No trans file available. EEG data ignored.")
        types.remove("eeg")

    if len(types) == 0:
        raise RuntimeError("No data available for mapping.")

    _check_option("meg_surf", meg_surf, ["helmet", "head"])

    surfs = []
    for this_type in types:
        if this_type == "meg" and meg_surf == "helmet":
            surf = get_meg_helmet_surf(info, trans)
        else:
            surf = get_head_surf(subject, source=head_source, subjects_dir=subjects_dir)
        surfs.append(surf)

    surf_maps = list()

    for this_type, this_surf in zip(types, surfs):
        this_map = _make_surface_mapping(
            evoked.info,
            this_surf,
            this_type,
            trans,
            n_jobs=n_jobs,
            origin=origin,
            mode=mode,
        )
        surf_maps.append(this_map)

    return surf_maps
