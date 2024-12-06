# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..defaults import DEFAULTS
from ..utils import (
    _check_option,
    _check_preload,
    _on_missing,
    _validate_type,
    fill_doc,
    logger,
    pinv,
    verbose,
    warn,
)
from .constants import FIFF
from .meas_info import _check_ch_keys
from .pick import _ELECTRODE_CH_TYPES, pick_channels, pick_channels_forward, pick_types
from .proj import _has_eeg_average_ref_proj, make_eeg_average_ref_proj, setup_proj


def _check_before_reference(inst, ref_from, ref_to, ch_type):
    """Prepare instance for referencing."""
    # Check to see that data is preloaded
    _check_preload(inst, "Applying a reference")

    ch_type = _get_ch_type(inst, ch_type)
    ch_dict = {**{type_: True for type_ in ch_type}, "meg": False, "ref_meg": False}
    eeg_idx = pick_types(inst.info, **ch_dict)

    if ref_to is None:
        ref_to = [inst.ch_names[i] for i in eeg_idx]
        extra = "EEG channels found"
    else:
        extra = "channels supplied"
    if len(ref_to) == 0:
        raise ValueError(f"No {extra} to apply the reference to")

    _check_ssp(inst, ref_from + ref_to)

    # If the reference touches EEG/ECoG/sEEG/DBS electrodes, note in the
    # info that a non-CAR has been applied.
    ref_to_channels = pick_channels(inst.ch_names, ref_to, ordered=True)
    if len(np.intersect1d(ref_to_channels, eeg_idx)) > 0:
        with inst.info._unlock():
            inst.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

    return ref_to


def _check_ssp(inst, ref_items):
    """Check for SSPs that may block re-referencing."""
    projs_to_remove = []
    for i, proj in enumerate(inst.info["projs"]):
        # Remove any average reference projections
        if (
            proj["desc"] == "Average EEG reference"
            or proj["kind"] == FIFF.FIFFV_PROJ_ITEM_EEG_AVREF
        ):
            logger.info("Removing existing average EEG reference projection.")
            # Don't remove the projection right away, but do this at the end of
            # this loop.
            projs_to_remove.append(i)

        # Inactive SSPs may block re-referencing
        elif (
            not proj["active"]
            and len([ch for ch in ref_items if ch in proj["data"]["col_names"]]) > 0
        ):
            raise RuntimeError(
                "Inactive signal space projection (SSP) operators are "
                "present that operate on sensors involved in the desired "
                "referencing scheme. These projectors need to be applied "
                "using the apply_proj() method function before the desired "
                "reference can be set."
            )

    for i in projs_to_remove:
        del inst.info["projs"][i]

    # Need to call setup_proj after changing the projs:
    inst._projector, _ = setup_proj(inst.info, add_eeg_ref=False, activate=False)


def _check_before_dict_reference(inst, ref_dict):
    """Prepare instance for dict-based referencing."""
    # Check to see that data is preloaded
    _check_preload(inst, "Applying a reference")

    # Promote all values to list-like. This simplifies our logic and also helps catch
    # self-referencing cases like `{"Cz": ["Cz"]}`
    _refdict = {k: [v] if isinstance(v, str) else list(v) for k, v in ref_dict.items()}

    # Check that keys are strings and values are lists-of-strings
    key_types = {type(k) for k in _refdict}
    value_types = {type(v) for val in _refdict.values() for v in val}
    for elem_name, elem in dict(key=key_types, value=value_types).items():
        if bad_elem := elem - {str}:
            raise TypeError(
                f"{elem_name.capitalize()}s in the ref_channels dict must be strings. "
                f"Your dict has {elem_name}s of type "
                f'{", ".join(map(lambda x: x.__name__, bad_elem))}.'
            )

    # Check that keys are valid channels and values are lists-of-valid-channels
    ch_set = set(inst.ch_names)
    bad_ch_set = set(inst.info["bads"])
    keys = set(_refdict)
    values = set(sum(_refdict.values(), []))
    for elem_name, elem in dict(key=keys, value=values).items():
        if bad_elem := elem - ch_set:
            raise ValueError(
                f'ref_channels dict contains invalid {elem_name}(s) '
                f'({", ".join(bad_elem)}) '
                "that are not names of channels in the instance."
            )
        # Check that values are not bad channels
        if bad_elem := elem.intersection(bad_ch_set):
            warn(
                f"ref_channels dict contains {elem_name}(s) "
                f"({', '.join(bad_elem)}) "
                "that are marked as bad channels."
            )

    _check_ssp(inst, keys.union(values))

    # Check for self-referencing
    self_ref = [[k] == v for k, v in _refdict.items()]
    if any(self_ref):
        which = np.array(list(_refdict))[np.nonzero(self_ref)]
        for ch in which:
            warn(f"Channel {ch} is self-referenced, which will nullify the channel.")

    # Check that channel types match. First unpack list-like vals into separate items:
    pairs = [(k, v) for k in _refdict for v in _refdict[k]]
    ch_type_map = dict(zip(inst.ch_names, inst.get_channel_types()))
    mismatch = [ch_type_map[k] != ch_type_map[v] for k, v in pairs]
    if any(mismatch):
        mismatch_pairs = np.array(pairs)[mismatch]
        for k, v in mismatch_pairs:
            warn(
                f"Channel {k} ({ch_type_map[k]}) is referenced to channel {v} which is "
                f"a different channel type ({ch_type_map[v]})."
            )

    # convert channel names to indices
    keys_ix = pick_channels(inst.ch_names, list(_refdict), ordered=True)
    vals_ix = (pick_channels(inst.ch_names, v, ordered=True) for v in _refdict.values())
    return dict(zip(keys_ix, vals_ix))


def _apply_reference(inst, ref_from, ref_to=None, forward=None, ch_type="auto"):
    """Apply a custom EEG referencing scheme."""
    ref_to = _check_before_reference(inst, ref_from, ref_to, ch_type)

    # Compute reference
    if len(ref_from) > 0:
        # this is guaranteed below, but we should avoid the crazy pick_channels
        # behavior that [] gives all. Also use ordered=True just to make sure
        # that all supplied channels actually exist.
        assert len(ref_to) > 0
        ref_names = ref_from
        ref_from = pick_channels(inst.ch_names, ref_from, ordered=True)
        ref_to = pick_channels(inst.ch_names, ref_to, ordered=True)

        data = inst._data
        ref_data = data[..., ref_from, :].mean(-2, keepdims=True)
        data[..., ref_to, :] -= ref_data
        ref_data = ref_data[..., 0, :]

        # REST
        if forward is not None:
            # use ch_sel and the given forward
            forward = pick_channels_forward(forward, ref_names, ordered=True)
            # 1-3. Compute a forward (G) and avg-ref'ed data (done above)
            G = forward["sol"]["data"]
            assert G.shape[0] == len(ref_names)
            # 4. Compute the forward (G) and average-reference it (Ga):
            Ga = G - np.mean(G, axis=0, keepdims=True)
            # 5. Compute the Ga_inv by SVD
            Ga_inv = pinv(Ga, rtol=1e-6)
            # 6. Compute Ra = (G @ Ga_inv) in eq (8) from G and Ga_inv
            Ra = G @ Ga_inv
            # 7-8. Compute Vp = Ra @ Va; then Vpa=average(Vp)
            Vpa = np.mean(Ra @ data[..., ref_from, :], axis=-2, keepdims=True)
            data[..., ref_to, :] += Vpa
    else:
        ref_data = None

    return inst, ref_data


def _apply_dict_reference(inst, ref_dict):
    """Apply a dict-based custom EEG referencing scheme."""
    # this converts all keys to channel indices and all values to arrays of ch. indices:
    ref_dict = _check_before_dict_reference(inst, ref_dict)

    data = inst._data
    orig_data = data.copy()
    for ref_to, ref_from in ref_dict.items():
        ref_data = orig_data[..., ref_from, :].mean(-2, keepdims=True)
        data[..., [ref_to], :] -= ref_data

    with inst.info._unlock():
        inst.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    return inst, None


@fill_doc
def add_reference_channels(inst, ref_channels, copy=True):
    """Add reference channels to data that consists of all zeros.

    Adds reference channels to data that were not included during recording.
    This is useful when you need to re-reference your data to different
    channels. These added channels will consist of all zeros.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Instance of Raw or Epochs with EEG channels and reference channel(s).
    %(ref_channels)s
    copy : bool
        Specifies whether the data will be copied (True) or modified in-place
        (False). Defaults to True.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with added EEG reference channels.

    Notes
    -----
    .. warning::
        When :ref:`re-referencing <tut-set-eeg-ref>`,
        make sure to apply the montage using :meth:`mne.io.Raw.set_montage`
        only after calling this function. Applying a montage will only set
        locations of channels that exist at the time it is applied.
    """
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..io import BaseRaw

    # Check to see that data is preloaded
    _check_preload(inst, "add_reference_channels")
    _validate_type(ref_channels, (list, tuple, str), "ref_channels")
    if isinstance(ref_channels, str):
        ref_channels = [ref_channels]
    for ch in ref_channels:
        if ch in inst.info["ch_names"]:
            raise ValueError(f"Channel {ch} already specified in inst.")

    # Once CAR is applied (active), don't allow adding channels
    if _has_eeg_average_ref_proj(inst.info, check_active=True):
        raise RuntimeError("Average reference already applied to data.")

    if copy:
        inst = inst.copy()

    if isinstance(inst, BaseRaw | Evoked):
        data = inst._data
        refs = np.zeros((len(ref_channels), data.shape[1]))
        data = np.vstack((data, refs))
        inst._data = data
    elif isinstance(inst, BaseEpochs):
        data = inst._data
        x, y, z = data.shape
        refs = np.zeros((x * len(ref_channels), z))
        data = np.vstack((data.reshape((x * y, z), order="F"), refs))
        data = data.reshape(x, y + len(ref_channels), z, order="F")
        inst._data = data
    else:
        raise TypeError(
            f"inst should be Raw, Epochs, or Evoked instead of {type(inst)}."
        )
    nchan = len(inst.info["ch_names"])

    if inst.info.get("dig", None) is not None:
        # A montage has been set. Try to infer location of reference channels.
        # "zeroth" EEG electrode dig points is reference
        ref_dig_loc = [
            dl
            for dl in inst.info["dig"]
            if (dl["kind"] == FIFF.FIFFV_POINT_EEG and dl["ident"] == 0)
        ]
        if len(ref_channels) > 1 or len(ref_dig_loc) != len(ref_channels):
            ref_dig_array = np.full(12, np.nan)
            warn(
                "Location for this channel is unknown or ambiguous; consider calling "
                "set_montage() after adding new reference channels if needed. "
                "Applying a montage will only set locations of channels that "
                "exist at the time it is applied."
            )
        else:  # n_ref_channels == 1 and a single ref digitization exists
            ref_dig_array = np.concatenate(
                (ref_dig_loc[0]["r"], ref_dig_loc[0]["r"], np.zeros(6))
            )
            # Replace the (possibly new) Ref location for each channel
            for idx in pick_types(inst.info, meg=False, eeg=True, exclude=[]):
                inst.info["chs"][idx]["loc"][3:6] = ref_dig_loc[0]["r"]
    else:
        # If no montage has ever been set, we cannot even try to infer a location.
        ref_dig_array = np.full(12, np.nan)

    for ch in ref_channels:
        chan_info = {
            "ch_name": ch,
            "coil_type": FIFF.FIFFV_COIL_EEG,
            "kind": FIFF.FIFFV_EEG_CH,
            "logno": nchan + 1,
            "scanno": nchan + 1,
            "cal": 1,
            "range": 1.0,
            "unit_mul": FIFF.FIFF_UNITM_NONE,
            "unit": FIFF.FIFF_UNIT_V,
            "coord_frame": FIFF.FIFFV_COORD_HEAD,
            "loc": ref_dig_array,
        }
        inst.info["chs"].append(chan_info)
        inst.info._update_redundant()
    range_ = np.arange(1, len(ref_channels) + 1)
    if isinstance(inst, BaseRaw):
        inst._cals = np.hstack((inst._cals, [1] * len(ref_channels)))
        for pi, picks in enumerate(inst._read_picks):
            inst._read_picks[pi] = np.concatenate([picks, np.max(picks) + range_])
    elif isinstance(inst, BaseEpochs):
        picks = inst.picks
        inst.picks = np.concatenate([picks, np.max(picks) + range_])
    inst.info._check_consistency()
    set_eeg_reference(inst, ref_channels=ref_channels, copy=False, verbose=False)
    return inst


_ref_dict = {
    FIFF.FIFFV_MNE_CUSTOM_REF_ON: "on",
    FIFF.FIFFV_MNE_CUSTOM_REF_OFF: "off",
    FIFF.FIFFV_MNE_CUSTOM_REF_CSD: "CSD",
}


def _check_can_reref(inst):
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..io import BaseRaw

    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), "Instance")
    current_custom = inst.info["custom_ref_applied"]
    if current_custom not in (
        FIFF.FIFFV_MNE_CUSTOM_REF_ON,
        FIFF.FIFFV_MNE_CUSTOM_REF_OFF,
    ):
        raise RuntimeError(
            "Cannot set new reference on data with custom reference type "
            f"{_ref_dict[current_custom]!r}"
        )


@verbose
def set_eeg_reference(
    inst,
    ref_channels="average",
    copy=True,
    projection=False,
    ch_type="auto",
    forward=None,
    *,
    joint=False,
    verbose=None,
):
    """Specify which reference to use for EEG data.

    Use this function to explicitly specify the desired reference for EEG.
    This can be either an existing electrode or a new virtual channel.
    This function will re-reference the data according to the desired
    reference.

    Note that it is also possible to re-reference the signal using a
    Laplacian (LAP) "reference-free" transformation using the
    :func:`.compute_current_source_density` function.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Instance of Raw or Epochs with EEG channels and reference channel(s).
    %(ref_channels_set_eeg_reference)s
    copy : bool
        Specifies whether the data will be copied (True) or modified in-place
        (False). Defaults to True.
    %(projection_set_eeg_reference)s
    %(ch_type_set_eeg_reference)s
    %(forward_set_eeg_reference)s
    %(joint_set_eeg_reference)s
    %(verbose)s

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with EEG channels re-referenced. If ``ref_channels="average"`` and
        ``projection=True`` a projection will be added instead of directly
        re-referencing the data.
    ref_data : array
        Array of reference data subtracted from EEG channels. This will be
        ``None`` if ``projection=True``, or if ``ref_channels`` is ``"REST"`` or a
        :class:`dict`.
    %(set_eeg_reference_see_also_notes)s
    """
    from ..forward import Forward

    _check_can_reref(inst)

    if isinstance(ref_channels, dict):
        logger.info("Applying a custom dict-based reference.")
        return _apply_dict_reference(inst, ref_channels)

    ch_type = _get_ch_type(inst, ch_type)

    if projection:  # average reference projector
        if ref_channels != "average":
            raise ValueError(
                'Setting projection=True is only supported for ref_channels="average", '
                f"got {ref_channels!r}."
            )
        # We need verbose='error' here in case we add projs sequentially
        if _has_eeg_average_ref_proj(inst.info, ch_type=ch_type, verbose="error"):
            warn(
                "An average reference projection was already added. The data "
                "has been left untouched."
            )
        else:
            # Creating an average reference may fail. In this case, make
            # sure that the custom_ref_applied flag is left untouched.
            custom_ref_applied = inst.info["custom_ref_applied"]

            try:
                with inst.info._unlock():
                    inst.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
                if joint:
                    inst.add_proj(
                        make_eeg_average_ref_proj(
                            inst.info, ch_type=ch_type, activate=False
                        )
                    )
                else:
                    for this_ch_type in ch_type:
                        inst.add_proj(
                            make_eeg_average_ref_proj(
                                inst.info, ch_type=this_ch_type, activate=False
                            )
                        )
            except Exception:
                with inst.info._unlock():
                    inst.info["custom_ref_applied"] = custom_ref_applied
                raise
            # If the data has been preloaded, projections will no
            # longer be automatically applied.
            if inst.preload:
                logger.info(
                    "Average reference projection was added, "
                    "but has not been applied yet. Use the "
                    "apply_proj method to apply it."
                )
        return inst, None
    del projection  # not used anymore

    inst = inst.copy() if copy else inst
    ch_dict = {**{type_: True for type_ in ch_type}, "meg": False, "ref_meg": False}
    ch_sel = [inst.ch_names[i] for i in pick_types(inst.info, **ch_dict)]

    if ref_channels == "REST":
        _validate_type(forward, Forward, 'forward when ref_channels="REST"')
    else:
        forward = None  # signal to _apply_reference not to do REST

    if ref_channels in ("average", "REST"):
        logger.info(f"Applying {ref_channels} reference.")
        ref_channels = ch_sel

    if ref_channels == []:
        logger.info("EEG data marked as already having the desired reference.")
    else:
        logger.info(
            "Applying a custom "
            f"{tuple(DEFAULTS['titles'][type_] for type_ in ch_type)} "
            "reference."
        )

    return _apply_reference(inst, ref_channels, ch_sel, forward, ch_type=ch_type)


def _get_ch_type(inst, ch_type):
    _validate_type(ch_type, (str, list, tuple), "ch_type")
    valid_ch_types = ("auto",) + _ELECTRODE_CH_TYPES
    if isinstance(ch_type, str):
        _check_option("ch_type", ch_type, valid_ch_types)
        if ch_type != "auto":
            ch_type = [ch_type]
    elif isinstance(ch_type, list | tuple):
        for type_ in ch_type:
            _validate_type(type_, str, "ch_type")
            _check_option("ch_type", type_, valid_ch_types[1:])
        ch_type = list(ch_type)

    # if ch_type is 'auto', search through list to find first reasonable
    # reference-able channel type.
    if ch_type == "auto":
        for type_ in _ELECTRODE_CH_TYPES:
            if type_ in inst:
                ch_type = [type_]
                logger.info(
                    f"{DEFAULTS['titles'][type_]} channel type selected for "
                    "re-referencing"
                )
                break
        # if auto comes up empty, or the user specifies a bad ch_type.
        else:
            raise ValueError("No EEG, ECoG, sEEG or DBS channels found to rereference.")
    return ch_type


@verbose
def set_bipolar_reference(
    inst,
    anode,
    cathode,
    ch_name=None,
    ch_info=None,
    drop_refs=True,
    copy=True,
    on_bad="warn",
    verbose=None,
):
    """Re-reference selected channels using a bipolar referencing scheme.

    A bipolar reference takes the difference between two channels (the anode
    minus the cathode) and adds it as a new virtual channel. The original
    channels will be dropped by default.

    Multiple anodes and cathodes can be specified, in which case multiple
    virtual channels will be created. The 1st cathode will be subtracted
    from the 1st anode, the 2nd cathode from the 2nd anode, etc.

    By default, the virtual channels will be annotated with channel-info and
    -location of the anodes and coil types will be set to EEG_BIPOLAR.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Data containing the unreferenced channels.
    anode : str | list of str
        The name(s) of the channel(s) to use as anode in the bipolar reference.
    cathode : str | list of str
        The name(s) of the channel(s) to use as cathode in the bipolar
        reference.
    ch_name : str | list of str | None
        The channel name(s) for the virtual channel(s) containing the resulting
        signal. By default, bipolar channels are named after the anode and
        cathode, but it is recommended to supply a more meaningful name.
    ch_info : dict | list of dict | None
        This parameter can be used to supply a dictionary (or a dictionary for
        each bipolar channel) containing channel information to merge in,
        overwriting the default values. Defaults to None.
    drop_refs : bool
        Whether to drop the anode/cathode channels from the instance.
    copy : bool
        Whether to operate on a copy of the data (True) or modify it in-place
        (False). Defaults to True.
    on_bad : str
        If a bipolar channel is created from a bad anode or a bad cathode, mne
        warns if on_bad="warns", raises ValueError if on_bad="raise", and does
        nothing if on_bad="ignore". For "warn" and "ignore", the new bipolar
        channel will be marked as bad. Defaults to on_bad="warns".
    %(verbose)s

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with the specified channels re-referenced.

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.

    Notes
    -----
    1. If the anodes contain any EEG channels, this function removes
       any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. The data must be preloaded.

    .. versionadded:: 0.9.0
    """
    from ..epochs import BaseEpochs, EpochsArray
    from ..evoked import EvokedArray
    from ..io import BaseRaw, RawArray
    from .meas_info import create_info

    _check_can_reref(inst)
    if not isinstance(anode, list):
        anode = [anode]

    if not isinstance(cathode, list):
        cathode = [cathode]

    if len(anode) != len(cathode):
        raise ValueError(
            f"Number of anodes (got {len(anode)}) must equal the number "
            f"of cathodes (got {len(cathode)})."
        )

    if ch_name is None:
        ch_name = [f"{a}-{c}" for (a, c) in zip(anode, cathode)]
    elif not isinstance(ch_name, list):
        ch_name = [ch_name]
    if len(ch_name) != len(anode):
        raise ValueError(
            "Number of channel names must equal the number of "
            f"anodes/cathodes (got {len(ch_name)})."
        )

    # Check for duplicate channel names (it is allowed to give the name of the
    # anode or cathode channel, as they will be replaced).
    for ch, a, c in zip(ch_name, anode, cathode):
        if ch not in [a, c] and ch in inst.ch_names:
            raise ValueError(
                f'There is already a channel named "{ch}", please '
                "specify a different name for the bipolar "
                "channel using the ch_name parameter."
            )

    if ch_info is None:
        ch_info = [{} for _ in anode]
    elif not isinstance(ch_info, list):
        ch_info = [ch_info]
    if len(ch_info) != len(anode):
        raise ValueError(
            "Number of channel info dictionaries must equal the "
            "number of anodes/cathodes."
        )

    if copy:
        inst = inst.copy()

    anode = _check_before_reference(
        inst, ref_from=cathode, ref_to=anode, ch_type="auto"
    )

    # Create bipolar reference channels by multiplying the data
    # (channels x time) with a matrix (n_virtual_channels x channels)
    # and add them to the instance.
    multiplier = np.zeros((len(anode), len(inst.ch_names)))
    for idx, (a, c) in enumerate(zip(anode, cathode)):
        multiplier[idx, inst.ch_names.index(a)] = 1
        multiplier[idx, inst.ch_names.index(c)] = -1

    ref_info = create_info(
        ch_names=ch_name,
        sfreq=inst.info["sfreq"],
        ch_types=inst.get_channel_types(picks=anode),
    )

    # Update "chs" in Reference-Info.
    for ch_idx, (an, info) in enumerate(zip(anode, ch_info)):
        _check_ch_keys(info, ch_idx, name="ch_info", check_min=False)
        an_idx = inst.ch_names.index(an)
        # Copy everything from anode (except ch_name).
        an_chs = {k: v for k, v in inst.info["chs"][an_idx].items() if k != "ch_name"}
        ref_info["chs"][ch_idx].update(an_chs)
        # Set coil-type to bipolar.
        ref_info["chs"][ch_idx]["coil_type"] = FIFF.FIFFV_COIL_EEG_BIPOLAR
        # Update with info from ch_info-parameter.
        ref_info["chs"][ch_idx].update(info)

    # Set other info-keys from original instance.
    pick_info = {
        k: v
        for k, v in inst.info.items()
        if k not in ["chs", "ch_names", "bads", "nchan", "sfreq"]
    }

    with ref_info._unlock():
        ref_info.update(pick_info)

    # Rereferencing of data.
    ref_data = multiplier @ inst._data

    if isinstance(inst, BaseRaw):
        ref_inst = RawArray(ref_data, ref_info, first_samp=inst.first_samp, copy=None)
    elif isinstance(inst, BaseEpochs):
        ref_inst = EpochsArray(
            ref_data,
            ref_info,
            events=inst.events,
            tmin=inst.tmin,
            event_id=inst.event_id,
            metadata=inst.metadata,
        )
    else:
        ref_inst = EvokedArray(
            ref_data,
            ref_info,
            tmin=inst.tmin,
            comment=inst.comment,
            nave=inst.nave,
            kind="average",
        )

    # Add referenced instance to original instance.
    inst.add_channels([ref_inst], force_update_info=True)

    # Handle bad channels.
    bad_bipolar_chs = []
    for ch_idx, (a, c) in enumerate(zip(anode, cathode)):
        if a in inst.info["bads"] or c in inst.info["bads"]:
            bad_bipolar_chs.append(ch_name[ch_idx])

    # Add warnings if bad channels are present.
    if bad_bipolar_chs:
        msg = f"Bipolar channels are based on bad channels: {bad_bipolar_chs}."
        _on_missing(on_bad, msg)
        inst.info["bads"] += bad_bipolar_chs

    added_channels = ", ".join([name for name in ch_name])
    logger.info(f"Added the following bipolar channels:\n{added_channels}")

    for attr_name in ["picks", "_projector"]:
        setattr(inst, attr_name, None)

    # Drop remaining channels.
    if drop_refs:
        drop_channels = list((set(anode) | set(cathode)) & set(inst.ch_names))
        inst.drop_channels(drop_channels)

    return inst
