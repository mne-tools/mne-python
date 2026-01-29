# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from copy import deepcopy

import numpy as np

from ..utils import (
    _check_option,
    _ensure_int,
    _validate_type,
    fill_doc,
    logger,
    verbose,
)
from .constants import FIFF


def get_channel_type_constants(include_defaults=False):
    """Return all known channel types, and associated FIFF constants.

    Parameters
    ----------
    include_defaults : bool
        Whether to include default values for "unit" and "coil_type" for all
        entries (see Notes). Defaults are generally based on values normally
        present for a VectorView MEG system. Defaults to ``False``.

    Returns
    -------
    channel_types : dict
        The keys are channel type strings, and the values are dictionaries of
        FIFF constants for "kind", and possibly "unit" and "coil_type".

    Notes
    -----
    Values which might vary within a channel type across real data
    recordings are excluded unless ``include_defaults=True``. For example,
    "ref_meg" channels may have coil type
    ``FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD``, ``FIFFV_COIL_VV_MAG_T3``, etc
    (depending on the recording system), so no "coil_type" entry is given
    for "ref_meg" unless ``include_defaults`` is requested.
    """
    base = dict(
        grad=dict(kind=FIFF.FIFFV_MEG_CH, unit=FIFF.FIFF_UNIT_T_M),
        mag=dict(kind=FIFF.FIFFV_MEG_CH, unit=FIFF.FIFF_UNIT_T),
        ref_meg=dict(kind=FIFF.FIFFV_REF_MEG_CH),
        eeg=dict(
            kind=FIFF.FIFFV_EEG_CH, unit=FIFF.FIFF_UNIT_V, coil_type=FIFF.FIFFV_COIL_EEG
        ),
        seeg=dict(
            kind=FIFF.FIFFV_SEEG_CH,
            unit=FIFF.FIFF_UNIT_V,
            coil_type=FIFF.FIFFV_COIL_EEG,
        ),
        dbs=dict(
            kind=FIFF.FIFFV_DBS_CH, unit=FIFF.FIFF_UNIT_V, coil_type=FIFF.FIFFV_COIL_EEG
        ),
        ecog=dict(
            kind=FIFF.FIFFV_ECOG_CH,
            unit=FIFF.FIFF_UNIT_V,
            coil_type=FIFF.FIFFV_COIL_EEG,
        ),
        eog=dict(kind=FIFF.FIFFV_EOG_CH, unit=FIFF.FIFF_UNIT_V),
        emg=dict(kind=FIFF.FIFFV_EMG_CH, unit=FIFF.FIFF_UNIT_V),
        ecg=dict(kind=FIFF.FIFFV_ECG_CH, unit=FIFF.FIFF_UNIT_V),
        resp=dict(kind=FIFF.FIFFV_RESP_CH, unit=FIFF.FIFF_UNIT_V),
        bio=dict(kind=FIFF.FIFFV_BIO_CH, unit=FIFF.FIFF_UNIT_V),
        misc=dict(kind=FIFF.FIFFV_MISC_CH, unit=FIFF.FIFF_UNIT_V),
        stim=dict(kind=FIFF.FIFFV_STIM_CH),
        exci=dict(kind=FIFF.FIFFV_EXCI_CH),
        syst=dict(kind=FIFF.FIFFV_SYST_CH),
        ias=dict(kind=FIFF.FIFFV_IAS_CH),
        gof=dict(kind=FIFF.FIFFV_GOODNESS_FIT),
        dipole=dict(kind=FIFF.FIFFV_DIPOLE_WAVE),
        chpi=dict(
            kind=[
                FIFF.FIFFV_QUAT_0,
                FIFF.FIFFV_QUAT_1,
                FIFF.FIFFV_QUAT_2,
                FIFF.FIFFV_QUAT_3,
                FIFF.FIFFV_QUAT_4,
                FIFF.FIFFV_QUAT_5,
                FIFF.FIFFV_QUAT_6,
                FIFF.FIFFV_HPI_G,
                FIFF.FIFFV_HPI_ERR,
                FIFF.FIFFV_HPI_MOV,
            ]
        ),
        fnirs_cw_amplitude=dict(
            kind=FIFF.FIFFV_FNIRS_CH,
            unit=FIFF.FIFF_UNIT_V,
            coil_type=FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE,
        ),
        fnirs_fd_ac_amplitude=dict(
            kind=FIFF.FIFFV_FNIRS_CH,
            unit=FIFF.FIFF_UNIT_V,
            coil_type=FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE,
        ),
        fnirs_fd_phase=dict(
            kind=FIFF.FIFFV_FNIRS_CH,
            unit=FIFF.FIFF_UNIT_RAD,
            coil_type=FIFF.FIFFV_COIL_FNIRS_FD_PHASE,
        ),
        fnirs_od=dict(kind=FIFF.FIFFV_FNIRS_CH, coil_type=FIFF.FIFFV_COIL_FNIRS_OD),
        hbo=dict(
            kind=FIFF.FIFFV_FNIRS_CH,
            unit=FIFF.FIFF_UNIT_MOL,
            coil_type=FIFF.FIFFV_COIL_FNIRS_HBO,
        ),
        hbr=dict(
            kind=FIFF.FIFFV_FNIRS_CH,
            unit=FIFF.FIFF_UNIT_MOL,
            coil_type=FIFF.FIFFV_COIL_FNIRS_HBR,
        ),
        csd=dict(
            kind=FIFF.FIFFV_EEG_CH,
            unit=FIFF.FIFF_UNIT_V_M2,
            coil_type=FIFF.FIFFV_COIL_EEG_CSD,
        ),
        temperature=dict(kind=FIFF.FIFFV_TEMPERATURE_CH, unit=FIFF.FIFF_UNIT_CEL),
        gsr=dict(kind=FIFF.FIFFV_GALVANIC_CH, unit=FIFF.FIFF_UNIT_S),
        eyegaze=dict(
            kind=FIFF.FIFFV_EYETRACK_CH, coil_type=FIFF.FIFFV_COIL_EYETRACK_POS
        ),
        pupil=dict(
            kind=FIFF.FIFFV_EYETRACK_CH, coil_type=FIFF.FIFFV_COIL_EYETRACK_PUPIL
        ),
    )
    if include_defaults:
        coil_none = dict(coil_type=FIFF.FIFFV_COIL_NONE)
        unit_none = dict(unit=FIFF.FIFF_UNIT_NONE)
        defaults = dict(
            grad=dict(coil_type=FIFF.FIFFV_COIL_VV_PLANAR_T1),
            mag=dict(coil_type=FIFF.FIFFV_COIL_VV_MAG_T3),
            ref_meg=dict(coil_type=FIFF.FIFFV_COIL_VV_MAG_T3, unit=FIFF.FIFF_UNIT_T),
            misc=dict(**coil_none, **unit_none),  # NB: overwrites UNIT_V
            stim=dict(unit=FIFF.FIFF_UNIT_V, **coil_none),
            eog=coil_none,
            ecg=coil_none,
            emg=coil_none,
            bio=coil_none,
            fnirs_od=unit_none,
            pupil=unit_none,
            eyegaze=dict(unit=FIFF.FIFF_UNIT_PX),
        )
        for key, value in defaults.items():
            base[key].update(value)
    return base


_first_rule = {
    FIFF.FIFFV_MEG_CH: "meg",
    FIFF.FIFFV_REF_MEG_CH: "ref_meg",
    FIFF.FIFFV_EEG_CH: "eeg",
    FIFF.FIFFV_STIM_CH: "stim",
    FIFF.FIFFV_EOG_CH: "eog",
    FIFF.FIFFV_EMG_CH: "emg",
    FIFF.FIFFV_ECG_CH: "ecg",
    FIFF.FIFFV_RESP_CH: "resp",
    FIFF.FIFFV_MISC_CH: "misc",
    FIFF.FIFFV_EXCI_CH: "exci",
    FIFF.FIFFV_IAS_CH: "ias",
    FIFF.FIFFV_SYST_CH: "syst",
    FIFF.FIFFV_SEEG_CH: "seeg",
    FIFF.FIFFV_DBS_CH: "dbs",
    FIFF.FIFFV_BIO_CH: "bio",
    FIFF.FIFFV_QUAT_0: "chpi",
    FIFF.FIFFV_QUAT_1: "chpi",
    FIFF.FIFFV_QUAT_2: "chpi",
    FIFF.FIFFV_QUAT_3: "chpi",
    FIFF.FIFFV_QUAT_4: "chpi",
    FIFF.FIFFV_QUAT_5: "chpi",
    FIFF.FIFFV_QUAT_6: "chpi",
    FIFF.FIFFV_HPI_G: "chpi",
    FIFF.FIFFV_HPI_ERR: "chpi",
    FIFF.FIFFV_HPI_MOV: "chpi",
    FIFF.FIFFV_DIPOLE_WAVE: "dipole",
    FIFF.FIFFV_GOODNESS_FIT: "gof",
    FIFF.FIFFV_ECOG_CH: "ecog",
    FIFF.FIFFV_FNIRS_CH: "fnirs",
    FIFF.FIFFV_TEMPERATURE_CH: "temperature",
    FIFF.FIFFV_GALVANIC_CH: "gsr",
    FIFF.FIFFV_EYETRACK_CH: "eyetrack",
}
# How to reduce our categories in channel_type (originally)
_second_rules = {
    "meg": ("unit", {FIFF.FIFF_UNIT_T_M: "grad", FIFF.FIFF_UNIT_T: "mag"}),
    "fnirs": (
        "coil_type",
        {
            FIFF.FIFFV_COIL_FNIRS_HBO: "hbo",
            FIFF.FIFFV_COIL_FNIRS_HBR: "hbr",
            FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE: "fnirs_cw_amplitude",
            FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE: "fnirs_fd_ac_amplitude",
            FIFF.FIFFV_COIL_FNIRS_FD_PHASE: "fnirs_fd_phase",
            FIFF.FIFFV_COIL_FNIRS_OD: "fnirs_od",
        },
    ),
    "eeg": (
        "coil_type",
        {
            FIFF.FIFFV_COIL_EEG: "eeg",
            FIFF.FIFFV_COIL_EEG_BIPOLAR: "eeg",
            FIFF.FIFFV_COIL_NONE: "eeg",  # MNE-C backward compat
            FIFF.FIFFV_COIL_EEG_CSD: "csd",
        },
    ),
    "eyetrack": (
        "coil_type",
        {
            FIFF.FIFFV_COIL_EYETRACK_POS: "eyegaze",
            FIFF.FIFFV_COIL_EYETRACK_PUPIL: "pupil",
        },
    ),
}


@fill_doc
def channel_type(info, idx):
    """Get channel type.

    Parameters
    ----------
    %(info_not_none)s
    idx : int
        Index of channel.

    Returns
    -------
    type : str
        Type of channel. Will be one of::

            {'bio', 'chpi', 'dbs', 'dipole', 'ecg', 'ecog', 'eeg', 'emg',
            'eog', 'exci', 'eyetrack', 'fnirs', 'gof', 'gsr', 'ias', 'misc',
            'meg', 'ref_meg', 'resp', 'seeg', 'stim', 'syst', 'temperature'}
    """
    # This is faster than the original _channel_type_old now in test_pick.py
    # because it uses (at most!) two dict lookups plus one conditional
    # to get the channel type string.
    ch = info["chs"][idx]
    try:
        first_kind = _first_rule[ch["kind"]]
    except KeyError:
        raise ValueError(
            f'Unknown channel type ({ch["kind"]}) for channel "{ch["ch_name"]}"'
        )
    if first_kind in _second_rules:
        key, second_rule = _second_rules[first_kind]
        first_kind = second_rule[ch[key]]
    return first_kind


@verbose
def pick_channels(ch_names, include, exclude=(), ordered=True, *, verbose=None):
    """Pick channels by names.

    Returns the indices of ``ch_names`` in ``include`` but not in ``exclude``.

    Parameters
    ----------
    ch_names : list of str
        List of channels.
    include : list of str
        List of channels to include (if empty include all available).

        .. note:: This is to be treated as a set. The order of this list
           is not used or maintained in ``sel``.

    exclude : list of str
        List of channels to exclude (if empty do not exclude any channel).
        Defaults to [].
    %(ordered)s
    %(verbose)s

    Returns
    -------
    sel : array of int
        Indices of good channels.

    See Also
    --------
    pick_channels_regexp, pick_types
    """
    if len(np.unique(ch_names)) != len(ch_names):
        raise RuntimeError("ch_names is not a unique list, picking is unsafe")
    _validate_type(ordered, bool, "ordered")
    _check_excludes_includes(include)
    _check_excludes_includes(exclude)
    if not isinstance(include, list):
        include = list(include)
    if len(include) == 0:
        include = list(ch_names)
    if not isinstance(exclude, list):
        exclude = list(exclude)
    sel, missing = list(), list()
    for name in include:
        if name in ch_names:
            if name not in exclude:
                sel.append(ch_names.index(name))
        else:
            missing.append(name)
    if len(missing) and ordered:
        raise ValueError(
            f"Missing channels from ch_names required by include:\n{missing}"
        )
    if not ordered:
        sel = np.unique(sel)
    return np.array(sel, int)


def pick_channels_regexp(ch_names, regexp):
    """Pick channels using regular expression.

    Returns the indices of the good channels in ch_names.

    Parameters
    ----------
    ch_names : list of str
        List of channels.

    regexp : str
        The regular expression. See python standard module for regular
        expressions.

    Returns
    -------
    sel : array of int
        Indices of good channels.

    See Also
    --------
    pick_channels

    Examples
    --------
    >>> pick_channels_regexp(['MEG 2331', 'MEG 2332', 'MEG 2333'], 'MEG ...1')
    [0]
    >>> pick_channels_regexp(['MEG 2331', 'MEG 2332', 'MEG 2333'], 'MEG *')
    [0, 1, 2]
    """
    r = re.compile(regexp)
    return [k for k, name in enumerate(ch_names) if r.match(name)]


def _triage_meg_pick(ch, meg):
    """Triage an MEG pick type."""
    if meg is True:
        return True
    elif ch["unit"] == FIFF.FIFF_UNIT_T_M:
        if meg == "grad":
            return True
        elif meg == "planar1" and ch["ch_name"].endswith("2"):
            return True
        elif meg == "planar2" and ch["ch_name"].endswith("3"):
            return True
    elif meg == "mag" and ch["unit"] == FIFF.FIFF_UNIT_T:
        return True
    return False


def _triage_fnirs_pick(ch, fnirs, warned):
    """Triage an fNIRS pick type."""
    if fnirs is True:
        return True
    elif ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_HBO and "hbo" in fnirs:
        return True
    elif ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_HBR and "hbr" in fnirs:
        return True
    elif (
        ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE
        and "fnirs_cw_amplitude" in fnirs
    ):
        return True
    elif (
        ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE
        and "fnirs_fd_ac_amplitude" in fnirs
    ):
        return True
    elif (
        ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_FD_PHASE and "fnirs_fd_phase" in fnirs
    ):
        return True
    elif ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_OD and "fnirs_od" in fnirs:
        return True
    return False


def _triage_eyetrack_pick(ch, eyetrack):
    """Triage an eyetrack pick type."""
    if eyetrack is False:
        return False
    elif eyetrack is True:
        return True
    elif ch["coil_type"] == FIFF.FIFFV_COIL_EYETRACK_PUPIL and "pupil" in eyetrack:
        return True
    elif ch["coil_type"] == FIFF.FIFFV_COIL_EYETRACK_POS and "eyegaze" in eyetrack:
        return True
    return False


def _check_meg_type(meg, allow_auto=False):
    """Ensure a valid meg type."""
    if isinstance(meg, str):
        allowed_types = ["grad", "mag", "planar1", "planar2"]
        allowed_types += ["auto"] if allow_auto else []
        if meg not in allowed_types:
            raise ValueError(
                f"meg value must be one of {allowed_types} or bool, not {meg}"
            )


def _check_info_exclude(info, exclude):
    _validate_type(info, "info")
    info._check_consistency()
    if exclude is None:
        raise ValueError('exclude must be a list of strings or "bads"')
    elif exclude == "bads":
        exclude = info.get("bads", [])
    elif not isinstance(exclude, list | tuple):
        raise ValueError(
            'exclude must either be "bads" or a list of strings.'
            " If only one channel is to be excluded, use "
            "[ch_name] instead of passing ch_name."
        )
    return exclude


@fill_doc
def pick_types(
    info,
    meg=False,
    eeg=False,
    stim=False,
    eog=False,
    ecg=False,
    emg=False,
    ref_meg="auto",
    *,
    misc=False,
    resp=False,
    chpi=False,
    exci=False,
    ias=False,
    syst=False,
    seeg=False,
    dipole=False,
    gof=False,
    bio=False,
    ecog=False,
    fnirs=False,
    csd=False,
    dbs=False,
    temperature=False,
    gsr=False,
    eyetrack=False,
    include=(),
    exclude="bads",
    selection=None,
):
    """Pick channels by type and names.

    Parameters
    ----------
    %(info_not_none)s
    %(pick_types_params)s

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    # NOTE: Changes to this function's signature should also be changed in
    # PickChannelsMixin
    _validate_type(meg, (bool, str), "meg")

    exclude = _check_info_exclude(info, exclude)
    nchan = info["nchan"]
    pick = np.zeros(nchan, dtype=bool)

    _check_meg_type(ref_meg, allow_auto=True)
    _check_meg_type(meg)
    if isinstance(ref_meg, str) and ref_meg == "auto":
        ref_meg = (
            "comps" in info
            and info["comps"] is not None
            and len(info["comps"]) > 0
            and meg is not False
        )

    for param in (
        eeg,
        stim,
        eog,
        ecg,
        emg,
        misc,
        resp,
        chpi,
        exci,
        ias,
        syst,
        seeg,
        dipole,
        gof,
        bio,
        ecog,
        csd,
        dbs,
        temperature,
        gsr,
    ):
        if not isinstance(param, bool):
            w = (
                "Parameters for all channel types (with the exception of "
                '"meg", "ref_meg", "fnirs", and "eyetrack") must be of type '
                "bool, not {}."
            )
            raise ValueError(w.format(type(param)))

    param_dict = dict(
        eeg=eeg,
        stim=stim,
        eog=eog,
        ecg=ecg,
        emg=emg,
        misc=misc,
        resp=resp,
        chpi=chpi,
        exci=exci,
        ias=ias,
        syst=syst,
        seeg=seeg,
        dbs=dbs,
        dipole=dipole,
        gof=gof,
        bio=bio,
        ecog=ecog,
        csd=csd,
        temperature=temperature,
        gsr=gsr,
        eyetrack=eyetrack,
    )

    # avoid triage if possible
    if isinstance(meg, bool):
        for key in ("grad", "mag"):
            param_dict[key] = meg
    if isinstance(fnirs, bool):
        for key in _FNIRS_CH_TYPES_SPLIT:
            param_dict[key] = fnirs
    warned = [False]
    for k in range(nchan):
        ch_type = channel_type(info, k)
        try:
            pick[k] = param_dict[ch_type]
        except KeyError:  # not so simple
            assert (
                ch_type
                in ("grad", "mag", "ref_meg")
                + _FNIRS_CH_TYPES_SPLIT
                + _EYETRACK_CH_TYPES_SPLIT
            )
            if ch_type in ("grad", "mag"):
                pick[k] = _triage_meg_pick(info["chs"][k], meg)
            elif ch_type == "ref_meg":
                pick[k] = _triage_meg_pick(info["chs"][k], ref_meg)
            elif ch_type in ("eyegaze", "pupil"):
                pick[k] = _triage_eyetrack_pick(info["chs"][k], eyetrack)
            else:  # ch_type in ('hbo', 'hbr')
                pick[k] = _triage_fnirs_pick(info["chs"][k], fnirs, warned)

    # restrict channels to selection if provided
    if selection is not None:
        # the selection only restricts these types of channels
        sel_kind = [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH, FIFF.FIFFV_EEG_CH]
        for k in np.where(pick)[0]:
            if (
                info["chs"][k]["kind"] in sel_kind
                and info["ch_names"][k] not in selection
            ):
                pick[k] = False

    myinclude = [info["ch_names"][k] for k in range(nchan) if pick[k]]
    myinclude += include

    if len(myinclude) == 0:
        sel = np.array([], int)
    else:
        sel = pick_channels(info["ch_names"], myinclude, exclude, ordered=False)

    return sel


@verbose
def pick_info(info, sel=(), copy=True, verbose=None):
    """Restrict an info structure to a selection of channels.

    Parameters
    ----------
    %(info_not_none)s
    sel : list of int | None
        Indices of channels to include. If None, all channels
        are included.
    copy : bool
        If copy is False, info is modified inplace.
    %(verbose)s

    Returns
    -------
    res : dict
        Info structure restricted to a selection of channels.
    """
    # avoid circular imports
    from .meas_info import _bad_chans_comp

    info._check_consistency()
    info = info.copy() if copy else info
    if sel is None:
        return info
    elif len(sel) == 0:
        raise ValueError("No channels match the selection.")
    ch_set = set(info["ch_names"][k] for k in sel)
    n_unique = len(ch_set)
    if n_unique != len(sel):
        raise ValueError(
            f"Found {n_unique} / {len(sel)} unique names, sel is not unique"
        )

    # make sure required the compensation channels are present
    if len(info.get("comps", [])) > 0:
        ch_names = [info["ch_names"][idx] for idx in sel]
        _, comps_missing = _bad_chans_comp(info, ch_names)
        if len(comps_missing) > 0:
            logger.info(
                f"Removing {len(info['comps'])} compensators from info because "
                "not all compensation channels were picked."
            )
            with info._unlock():
                info["comps"] = []
    with info._unlock():
        info["chs"] = [info["chs"][k] for k in sel]
    info._update_redundant()
    info["bads"] = [ch for ch in info["bads"] if ch in info["ch_names"]]
    if "comps" in info:
        comps = deepcopy(info["comps"])
        for c in comps:
            row_idx = [
                k for k, n in enumerate(c["data"]["row_names"]) if n in info["ch_names"]
            ]
            row_names = [c["data"]["row_names"][i] for i in row_idx]
            rowcals = c["rowcals"][row_idx]
            c["rowcals"] = rowcals
            c["data"]["nrow"] = len(row_names)
            c["data"]["row_names"] = row_names
            c["data"]["data"] = c["data"]["data"][row_idx]
        with info._unlock():
            info["comps"] = comps
    if info.get("custom_ref_applied", False) and not _electrode_types(info):
        with info._unlock():
            info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    # remove unused projectors
    if info.get("projs", False):
        projs = list()
        for p in info["projs"]:
            if any(ch_name in ch_set for ch_name in p["data"]["col_names"]):
                projs.append(p)
        if len(projs) != len(info["projs"]):
            with info._unlock():
                info["projs"] = projs
    info._check_consistency()

    return info


def _has_kit_refs(info, picks):
    """Determine if KIT ref channels are chosen.

    This is currently only used by make_forward_solution, which cannot
    run when KIT reference channels are included.
    """
    for p in picks:
        if info["chs"][p]["coil_type"] == FIFF.FIFFV_COIL_KIT_REF_MAG:
            return True
    return False


@verbose
def pick_channels_forward(
    orig, include=(), exclude=(), ordered=True, copy=True, *, verbose=None
):
    """Pick channels from forward operator.

    Parameters
    ----------
    orig : dict
        A forward solution.
    include : list of str
        List of channels to include (if empty, include all available).
        Defaults to [].
    exclude : list of str | 'bads'
        Channels to exclude (if empty, do not exclude any). Defaults to [].
        If 'bads', then exclude bad channels in orig.
    %(ordered)s
    copy : bool
        If True (default), make a copy.

        .. versionadded:: 0.19
    %(verbose)s

    Returns
    -------
    res : dict
        Forward solution restricted to selected channels. If include and
        exclude are empty it returns orig without copy.
    """
    orig["info"]._check_consistency()
    if len(include) == 0 and len(exclude) == 0:
        return orig.copy() if copy else orig
    exclude = _check_excludes_includes(exclude, info=orig["info"], allow_bads=True)

    # Allow for possibility of channel ordering in forward solution being
    # different from that of the M/EEG file it is based on.
    sel_sol = pick_channels(
        orig["sol"]["row_names"], include=include, exclude=exclude, ordered=ordered
    )
    sel_info = pick_channels(
        orig["info"]["ch_names"], include=include, exclude=exclude, ordered=ordered
    )

    fwd = deepcopy(orig) if copy else orig

    # Check that forward solution and original data file agree on #channels
    if len(sel_sol) != len(sel_info):
        raise ValueError(
            "Forward solution and functional data appear to "
            "have different channel names, please check."
        )

    #   Do we have something?
    nuse = len(sel_sol)
    if nuse == 0:
        raise ValueError("Nothing remains after picking")

    logger.info(f"    {nuse:d} out of {fwd['nchan']} channels remain after picking")

    #   Pick the correct rows of the forward operator using sel_sol
    fwd["sol"]["data"] = fwd["sol"]["data"][sel_sol, :]
    fwd["_orig_sol"] = fwd["_orig_sol"][sel_sol, :]
    fwd["sol"]["nrow"] = nuse

    ch_names = [fwd["sol"]["row_names"][k] for k in sel_sol]
    fwd["nchan"] = nuse
    fwd["sol"]["row_names"] = ch_names

    # Pick the appropriate channel names from the info-dict using sel_info
    with fwd["info"]._unlock():
        fwd["info"]["chs"] = [fwd["info"]["chs"][k] for k in sel_info]
    fwd["info"]._update_redundant()
    fwd["info"]["bads"] = [b for b in fwd["info"]["bads"] if b in ch_names]

    if fwd["sol_grad"] is not None:
        fwd["sol_grad"]["data"] = fwd["sol_grad"]["data"][sel_sol, :]
        fwd["_orig_sol_grad"] = fwd["_orig_sol_grad"][sel_sol, :]
        fwd["sol_grad"]["nrow"] = nuse
        fwd["sol_grad"]["row_names"] = [
            fwd["sol_grad"]["row_names"][k] for k in sel_sol
        ]

    return fwd


def pick_types_forward(
    orig,
    meg=False,
    eeg=False,
    ref_meg=True,
    seeg=False,
    ecog=False,
    dbs=False,
    include=(),
    exclude=(),
):
    """Pick by channel type and names from a forward operator.

    Parameters
    ----------
    orig : dict
        A forward solution.
    meg : bool | str
        If True include MEG channels. If string it can be 'mag', 'grad',
        'planar1' or 'planar2' to select only magnetometers, all gradiometers,
        or a specific type of gradiometer.
    eeg : bool
        If True include EEG channels.
    ref_meg : bool
        If True include CTF / 4D reference channels.
    seeg : bool
        If True include stereotactic EEG channels.
    ecog : bool
        If True include electrocorticography channels.
    dbs : bool
        If True include deep brain stimulation channels.
    include : list of str
        List of additional channels to include. If empty do not include any.
    exclude : list of str | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in orig['info']['bads'].

    Returns
    -------
    res : dict
        Forward solution restricted to selected channel types.
    """
    info = orig["info"]
    sel = pick_types(
        info,
        meg,
        eeg,
        ref_meg=ref_meg,
        seeg=seeg,
        ecog=ecog,
        dbs=dbs,
        include=include,
        exclude=exclude,
    )
    if len(sel) == 0:
        raise ValueError("No valid channels found")
    include_ch_names = [info["ch_names"][k] for k in sel]

    return pick_channels_forward(orig, include_ch_names)


@fill_doc
def channel_indices_by_type(info, picks=None, *, exclude=()):
    """Get indices of channels by type.

    Parameters
    ----------
    %(info_not_none)s
    %(picks_all)s
    exclude : list | str
        Set of channels to exclude, only used when picking based on
        types (e.g., exclude="bads" when picks="meg").

        .. versionadded:: 1.10.0

    Returns
    -------
    idx_by_type : dict
        A dictionary that maps each channel type to a (possibly empty) list of
        channel indices.
    """
    idx_by_type = {
        key: list()
        for key in _PICK_TYPES_KEYS
        if key not in ("meg", "fnirs", "eyetrack")
    }
    idx_by_type.update(
        mag=list(),
        grad=list(),
        hbo=list(),
        hbr=list(),
        fnirs_cw_amplitude=list(),
        fnirs_fd_ac_amplitude=list(),
        fnirs_fd_phase=list(),
        fnirs_od=list(),
        eyegaze=list(),
        pupil=list(),
    )
    picks = _picks_to_idx(info, picks, none="all", exclude=exclude, allow_empty=True)
    for k in picks:
        ch_type = channel_type(info, k)
        for key in idx_by_type.keys():
            if ch_type == key:
                idx_by_type[key].append(k)
    return idx_by_type


@verbose
def pick_channels_cov(
    orig, include=(), exclude="bads", ordered=True, copy=True, *, verbose=None
):
    """Pick channels from covariance matrix.

    Parameters
    ----------
    orig : Covariance
        A covariance.
    include : list of str, (optional)
        List of channels to include (if empty, include all available).
    exclude : list of str, (optional) | 'bads'
        Channels to exclude (if empty, do not exclude any). Defaults to 'bads'.
    %(ordered)s
    copy : bool
        If True (the default), return a copy of the covariance matrix with the
        modified channels. If False, channels are modified in-place.

        .. versionadded:: 0.20.0
    %(verbose)s

    Returns
    -------
    res : dict
        Covariance solution restricted to selected channels.
    """
    if copy:
        orig = orig.copy()
        # A little peculiarity of the cov objects is that these two fields
        # should not be copied over when None.
        if "method" in orig and orig["method"] is None:
            del orig["method"]
        if "loglik" in orig and orig["loglik"] is None:
            del orig["loglik"]

    exclude = orig["bads"] if exclude == "bads" else exclude
    sel = pick_channels(
        orig["names"], include=include, exclude=exclude, ordered=ordered
    )
    data = orig["data"][sel][:, sel] if not orig["diag"] else orig["data"][sel]
    names = [orig["names"][k] for k in sel]
    bads = [name for name in orig["bads"] if name in orig["names"]]

    orig["data"] = data
    orig["names"] = names
    orig["bads"] = bads
    orig["dim"] = len(data)

    return orig


def _mag_grad_dependent(info):
    """Determine of mag and grad should be dealt with jointly."""
    # right now just uses SSS, could be computed / checked from cov
    # but probably overkill
    return any(
        ph.get("max_info", {}).get("sss_info", {}).get("in_order", 0)
        for ph in info.get("proc_history", [])
    )


@fill_doc
def _contains_ch_type(info, ch_type):
    """Check whether a certain channel type is in an info object.

    Parameters
    ----------
    %(info_not_none)s
    ch_type : str
        the channel type to be checked for

    Returns
    -------
    has_ch_type : bool
        Whether the channel type is present or not.
    """
    _validate_type(ch_type, "str", "ch_type")

    meg_extras = list(_MEG_CH_TYPES_SPLIT)
    fnirs_extras = list(_FNIRS_CH_TYPES_SPLIT)
    et_extras = list(_EYETRACK_CH_TYPES_SPLIT)
    valid_channel_types = sorted(
        [key for key in _PICK_TYPES_KEYS if key != "meg"]
        + meg_extras
        + fnirs_extras
        + et_extras
    )
    _check_option("ch_type", ch_type, valid_channel_types)
    if info is None:
        raise ValueError(
            f'Cannot check for channels of type "{ch_type}" because info is None'
        )
    return any(ch_type == channel_type(info, ii) for ii in range(info["nchan"]))


@fill_doc
def _picks_by_type(info, meg_combined=False, ref_meg=False, exclude="bads"):
    """Get data channel indices as separate list of tuples.

    Parameters
    ----------
    %(info_not_none)s
    meg_combined : bool | 'auto'
        Whether to return combined picks for grad and mag.
        Can be 'auto' to choose based on Maxwell filtering status.
    ref_meg : bool
        If True include CTF / 4D reference channels
    exclude : list of str | str
        List of channels to exclude. If 'bads' (default), exclude channels
        in info['bads'].

    Returns
    -------
    picks_list : list of tuples
        The list of tuples of picks and the type string.
    """
    _validate_type(ref_meg, bool, "ref_meg")
    exclude = _check_info_exclude(info, exclude)
    if meg_combined == "auto":
        meg_combined = _mag_grad_dependent(info)

    picks_list = {ch_type: list() for ch_type in _DATA_CH_TYPES_SPLIT}
    for k in range(info["nchan"]):
        if info["chs"][k]["ch_name"] not in exclude:
            this_type = channel_type(info, k)
            try:
                picks_list[this_type].append(k)
            except KeyError:
                # This annoyance is due to differences in pick_types
                # and channel_type behavior
                if this_type == "ref_meg":
                    ch = info["chs"][k]
                    if _triage_meg_pick(ch, ref_meg):
                        if ch["unit"] == FIFF.FIFF_UNIT_T:
                            picks_list["mag"].append(k)
                        elif ch["unit"] == FIFF.FIFF_UNIT_T_M:
                            picks_list["grad"].append(k)
                else:
                    pass  # not a data channel type
    picks_list = [
        (ch_type, np.array(picks_list[ch_type], int))
        for ch_type in _DATA_CH_TYPES_SPLIT
    ]
    assert _DATA_CH_TYPES_SPLIT[:2] == ("mag", "grad")
    if meg_combined and len(picks_list[0][1]) and len(picks_list[1][1]):
        picks_list.insert(
            0,
            (
                "meg",
                np.unique(np.concatenate([picks_list.pop(0)[1], picks_list.pop(0)[1]])),
            ),
        )
    picks_list = [p for p in picks_list if len(p[1])]
    return picks_list


def _check_excludes_includes(chs, info=None, allow_bads=False):
    """Ensure that inputs to exclude/include are list-like or "bads".

    Parameters
    ----------
    chs : any input, should be list, tuple, set, str
        The channels passed to include or exclude.
    allow_bads : bool
        Allow the user to supply "bads" as a string for auto exclusion.

    Returns
    -------
    chs : list
        Channels to be excluded/excluded. If allow_bads, and chs=="bads",
        this will be the bad channels found in 'info'.
    """
    from .meas_info import Info

    if not isinstance(chs, list | tuple | set | np.ndarray):
        if allow_bads is True:
            if not isinstance(info, Info):
                raise ValueError("Supply an info object if allow_bads is true")
            elif chs != "bads":
                raise ValueError('If chs is a string, it must be "bads"')
            else:
                chs = info["bads"]
        else:
            raise ValueError(
                'include/exclude must be list, tuple, ndarray, or "bads". You provided '
                f"type {type(chs)}."
            )
    return chs


_PICK_TYPES_DATA_DICT = dict(
    meg=True,
    eeg=True,
    csd=True,
    stim=False,
    eog=False,
    ecg=False,
    emg=False,
    misc=False,
    resp=False,
    chpi=False,
    exci=False,
    ias=False,
    syst=False,
    seeg=True,
    dipole=False,
    gof=False,
    bio=False,
    ecog=True,
    fnirs=True,
    dbs=True,
    temperature=False,
    gsr=False,
    eyetrack=True,
)
_PICK_TYPES_KEYS = tuple(list(_PICK_TYPES_DATA_DICT) + ["ref_meg"])
_MEG_CH_TYPES_SPLIT = ("mag", "grad", "planar1", "planar2")
_FNIRS_CH_TYPES_SPLIT = (
    "hbo",
    "hbr",
    "fnirs_cw_amplitude",
    "fnirs_fd_ac_amplitude",
    "fnirs_fd_phase",
    "fnirs_od",
)
_EYETRACK_CH_TYPES_SPLIT = ("eyegaze", "pupil")
_DATA_CH_TYPES_ORDER_DEFAULT = (
    (
        "mag",
        "grad",
        "eeg",
        "csd",
        "eog",
        "ecg",
        "resp",
        "emg",
        "ref_meg",
        "misc",
        "stim",
        "chpi",
        "exci",
        "ias",
        "syst",
        "seeg",
        "bio",
        "ecog",
        "dbs",
        "temperature",
        "gsr",
        "gof",
        "dipole",
    )
    + _FNIRS_CH_TYPES_SPLIT
    + _EYETRACK_CH_TYPES_SPLIT
    + ("whitened",)
)

# Valid data types, ordered for consistency, used in viz/evoked.
_VALID_CHANNEL_TYPES = (
    (
        "eeg",
        "grad",
        "mag",
        "seeg",
        "eog",
        "ecg",
        "resp",
        "emg",
        "dipole",
        "gof",
        "bio",
        "ecog",
        "dbs",
    )
    + _FNIRS_CH_TYPES_SPLIT
    + _EYETRACK_CH_TYPES_SPLIT
    + ("misc", "csd")
)
_DATA_CH_TYPES_SPLIT = (
    "mag",
    "grad",
    "eeg",
    "csd",
    "seeg",
    "ecog",
    "dbs",
) + _FNIRS_CH_TYPES_SPLIT
# Electrode types (e.g., can be average-referenced together or separately)
_ELECTRODE_CH_TYPES = ("eeg", "ecog", "seeg", "dbs")


def _electrode_types(info, *, exclude="bads"):
    return [
        ch_type
        for ch_type in _ELECTRODE_CH_TYPES
        if len(pick_types(info, exclude=exclude, **{ch_type: True}))
    ]


def _pick_data_channels(info, exclude="bads", with_ref_meg=True, with_aux=False):
    """Pick only data channels."""
    kwargs = _PICK_TYPES_DATA_DICT
    if with_aux:
        kwargs = kwargs.copy()
        kwargs.update(eog=True, ecg=True, emg=True, bio=True)
    return pick_types(info, ref_meg=with_ref_meg, exclude=exclude, **kwargs)


def _pick_data_or_ica(info, exclude=()):
    """Pick only data or ICA channels."""
    if any(ch_name.startswith("ICA") for ch_name in info["ch_names"]):
        picks = pick_types(info, exclude=exclude, misc=True)
    else:
        picks = _pick_data_channels(info, exclude=exclude, with_ref_meg=True)
    return picks


def _picks_to_idx(
    info,
    picks,
    none="data",
    exclude="bads",
    allow_empty=False,
    with_ref_meg=True,
    return_kind=False,
    picks_on="channels",
):
    """Convert and check pick validity.

    Parameters
    ----------
    picks_on : str
        'channels' (default) for error messages about selection of channels.
        'components' for error messages about selection of components.
    """
    from .meas_info import Info

    picked_ch_type_or_generic = False
    #
    # None -> all, data, or data_or_ica (ndarray of int)
    #
    if isinstance(info, Info):
        n_chan = info["nchan"]
    else:
        info = _ensure_int(info, "info", "an int or Info")
        n_chan = info
    assert n_chan >= 0

    orig_picks = picks
    # We do some extra_repr gymnastics to avoid calling repr(orig_picks) too
    # soon as it can be a performance bottleneck (repr on ndarray is slow)
    extra_repr = ""
    if picks is None:
        if isinstance(info, int):  # special wrapper for no real info
            picks = np.arange(n_chan)
            extra_repr = ", treated as range({n_chan})"
        else:
            picks = none  # let _picks_str_to_idx handle it
            extra_repr = f'None, treated as "{none}"'

    #
    # slice
    #
    if isinstance(picks, slice):
        picks = np.arange(n_chan)[picks]

    #
    # -> ndarray of int (and make a copy)
    #
    picks = np.atleast_1d(picks)  # this works even for picks == 'something'
    picks = np.array([], dtype=int) if len(picks) == 0 else picks
    if picks.ndim != 1:
        raise ValueError(f"picks must be 1D, got {picks.ndim}D")
    if picks.dtype.char in ("S", "U"):
        picks = _picks_str_to_idx(
            info,
            picks,
            exclude,
            with_ref_meg,
            return_kind,
            extra_repr,
            allow_empty,
            orig_picks,
        )
        if return_kind:
            picked_ch_type_or_generic = picks[1]
            picks = picks[0]
    if picks.dtype.kind not in ["i", "u"]:
        extra_ch = " or list of str (names)" if picks_on == "channels" else ""
        msg = (
            f"picks must be a list of int (indices){extra_ch}. "
            f"The provided data type {picks.dtype} is invalid."
        )
        raise TypeError(msg)
    del extra_repr
    picks = picks.astype(int)

    #
    # ensure we have (optionally non-empty) ndarray of valid int
    #
    if len(picks) == 0 and not allow_empty:
        raise ValueError(
            f"No appropriate {picks_on} found for the given picks ({orig_picks!r})"
        )
    if (picks < -n_chan).any():
        raise IndexError(f"All picks must be >= {-n_chan}, got {repr(orig_picks)}")
    if (picks >= n_chan).any():
        raise IndexError(
            f"All picks must be < n_{picks_on} ({n_chan}), got {repr(orig_picks)}"
        )
    picks %= n_chan  # ensure positive
    if return_kind:
        return picks, picked_ch_type_or_generic
    return picks


def _picks_str_to_idx(
    info, picks, exclude, with_ref_meg, return_kind, extra_repr, allow_empty, orig_picks
):
    """Turn a list of str into ndarray of int."""
    # special case for _picks_to_idx w/no info: shouldn't really happen
    if isinstance(info, int):
        raise ValueError(
            "picks as str can only be used when measurement info is available"
        )

    #
    # first: check our special cases
    #

    picks_generic = list()
    if len(picks) == 1:
        if picks[0] in ("all", "data", "data_or_ica"):
            if picks[0] == "all":
                use_exclude = info["bads"] if exclude == "bads" else exclude
                picks_generic = pick_channels(
                    info["ch_names"], info["ch_names"], exclude=use_exclude
                )
            elif picks[0] == "data":
                picks_generic = _pick_data_channels(
                    info, exclude=exclude, with_ref_meg=with_ref_meg
                )
            elif picks[0] == "data_or_ica":
                picks_generic = _pick_data_or_ica(info, exclude=exclude)
            if len(picks_generic) == 0 and orig_picks is None and not allow_empty:
                raise ValueError(
                    f"picks ({repr(orig_picks) + extra_repr}) yielded no channels, "
                    "consider passing picks explicitly"
                )

    #
    # second: match all to channel names
    #

    bad_names = []
    picks_name = list()
    for pick in picks:
        try:
            picks_name.append(info["ch_names"].index(pick))
        except ValueError:
            bad_names.append(pick)

    #
    # third: match all to types
    #
    bad_type = None
    picks_type = list()
    kwargs = dict(meg=False)
    meg, fnirs, eyetrack = set(), set(), set()
    for pick in picks:
        if pick in _PICK_TYPES_KEYS:
            kwargs[pick] = True
        elif pick in _MEG_CH_TYPES_SPLIT:
            meg |= {pick}
        elif pick in _FNIRS_CH_TYPES_SPLIT:
            fnirs |= {pick}
        elif pick in _EYETRACK_CH_TYPES_SPLIT:
            eyetrack |= {pick}
        else:
            bad_type = pick
            break
    else:
        # bad_type is None but this could still be empty
        bad_type = list(picks)
        # triage MEG, FNIRS, and eyetrack, which are complicated due to non-bool entries
        extra_picks = set()
        if "ref_meg" not in picks and not with_ref_meg:
            kwargs["ref_meg"] = False
        if len(meg) > 0 and not kwargs.get("meg", False):
            # easiest just to iterate
            for use_meg in meg:
                extra_picks |= set(
                    pick_types(info, meg=use_meg, ref_meg=False, exclude=exclude)
                )
        if len(fnirs) and not kwargs.get("fnirs", False):
            idx = 0 if len(fnirs) == 1 else slice(None)
            kwargs["fnirs"] = list(fnirs)[idx]
        if len(eyetrack) and not kwargs.get("eyetrack", False):
            idx = 0 if len(eyetrack) == 1 else slice(None)
            kwargs["eyetrack"] = list(eyetrack)[idx]  # slice(None) is equivalent to all
        picks_type = pick_types(info, exclude=exclude, **kwargs)
        if len(extra_picks) > 0:
            picks_type = sorted(set(picks_type) | set(extra_picks))

    #
    # finally: ensure we have exactly one usable list
    #
    all_picks = (picks_generic, picks_name, picks_type)
    any_found = [len(p) > 0 for p in all_picks]
    if sum(any_found) == 0:
        if not allow_empty:
            raise ValueError(
                f"picks ({repr(orig_picks) + extra_repr}) could not be interpreted as "
                f'channel names (no channel "{bad_names}"), channel types (no type'
                f' "{bad_type}" present), or a generic type (just "all" or "data")'
            )
        picks = np.array([], int)
    elif sum(any_found) > 1:
        raise RuntimeError(
            "Some channel names are ambiguously equivalent to "
            "channel types, cannot use string-based "
            "picks for these"
        )
    else:
        picks = np.array(all_picks[np.where(any_found)[0][0]])

    picked_ch_type_or_generic = not len(picks_name)
    if len(bad_names) > 0 and not picked_ch_type_or_generic:
        raise ValueError(
            f"Channel(s) {bad_names} could not be picked, because "
            "they are not present in the info instance."
        )

    if return_kind:
        return picks, picked_ch_type_or_generic
    return picks
