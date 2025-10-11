# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import contextlib
import datetime
import operator
import re
import string
from collections import Counter, OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from io import BytesIO
from textwrap import shorten

import numpy as np

from ..defaults import _handle_default
from ..html_templates import _get_html_template
from ..utils import (
    _check_fname,
    _check_on_missing,
    _check_option,
    _dt_to_stamp,
    _on_missing,
    _pl,
    _stamp_to_dt,
    _validate_type,
    check_fname,
    fill_doc,
    logger,
    object_diff,
    repr_html,
    verbose,
    warn,
)
from ._digitization import (
    DigPoint,
    _dig_kind_ints,
    _dig_kind_proper,
    _dig_kind_rev,
    _format_dig_points,
    _get_data_as_dict_from_dig,
    _read_dig_fif,
    write_dig,
)
from .compensator import get_current_comp
from .constants import FIFF, _ch_unit_mul_named
from .ctf_comp import _read_ctf_comp, write_ctf_comp
from .open import fiff_open
from .pick import (
    _DATA_CH_TYPES_SPLIT,
    _contains_ch_type,
    _picks_to_idx,
    channel_type,
    get_channel_type_constants,
    pick_types,
)
from .proc_history import _read_proc_history, _write_proc_history
from .proj import (
    Projection,
    _normalize_proj,
    _proj_equal,
    _read_proj,
    _uniquify_projs,
    _write_proj,
)
from .tag import (
    _ch_coord_dict,
    _float_item,
    _int_item,
    _rename_list,
    _update_ch_info_named,
    find_tag,
    read_tag,
)
from .tree import dir_tree_find
from .write import (
    DATE_NONE,
    _safe_name_list,
    end_block,
    start_and_end_file,
    start_block,
    write_ch_info,
    write_coord_trans,
    write_dig_points,
    write_float,
    write_float_matrix,
    write_id,
    write_int,
    write_julian,
    write_name_list_sanitized,
    write_string,
)

b = bytes  # alias

_SCALAR_CH_KEYS = (
    "scanno",
    "logno",
    "kind",
    "range",
    "cal",
    "coil_type",
    "unit",
    "unit_mul",
    "coord_frame",
)
_ALL_CH_KEYS_SET = set(_SCALAR_CH_KEYS + ("loc", "ch_name"))
# XXX we need to require these except when doing simplify_info
_MIN_CH_KEYS_SET = set(("kind", "cal", "unit", "loc", "ch_name"))


def _get_valid_units():
    """Get valid units according to the International System of Units (SI).

    The International System of Units (SI, :footcite:`WikipediaSI`) is the
    default system for describing units in the Brain Imaging Data Structure
    (BIDS). For more information, see the BIDS specification
    :footcite:`BIDSdocs` and the appendix "Units" therein.

    References
    ----------
    .. footbibliography::
    """
    valid_prefix_names = [
        "yocto",
        "zepto",
        "atto",
        "femto",
        "pico",
        "nano",
        "micro",
        "milli",
        "centi",
        "deci",
        "deca",
        "hecto",
        "kilo",
        "mega",
        "giga",
        "tera",
        "peta",
        "exa",
        "zetta",
        "yotta",
    ]
    valid_prefix_symbols = [
        "y",
        "z",
        "a",
        "f",
        "p",
        "n",
        "µ",
        "m",
        "c",
        "d",
        "da",
        "h",
        "k",
        "M",
        "G",
        "T",
        "P",
        "E",
        "Z",
        "Y",
    ]
    valid_unit_names = [
        "metre",
        "kilogram",
        "second",
        "ampere",
        "kelvin",
        "mole",
        "candela",
        "radian",
        "steradian",
        "hertz",
        "newton",
        "pascal",
        "joule",
        "watt",
        "coulomb",
        "volt",
        "farad",
        "ohm",
        "siemens",
        "weber",
        "tesla",
        "henry",
        "degree Celsius",
        "lumen",
        "lux",
        "becquerel",
        "gray",
        "sievert",
        "katal",
    ]
    valid_unit_symbols = [
        "m",
        "kg",
        "s",
        "A",
        "K",
        "mol",
        "cd",
        "rad",
        "sr",
        "Hz",
        "N",
        "Pa",
        "J",
        "W",
        "C",
        "V",
        "F",
        "Ω",
        "S",
        "Wb",
        "T",
        "H",
        "°C",
        "lm",
        "lx",
        "Bq",
        "Gy",
        "Sv",
        "kat",
    ]

    # Valid units are all possible combinations of either prefix name or prefix
    # symbol together with either unit name or unit symbol. E.g., nV for
    # nanovolt
    valid_units = []
    valid_units += [
        "".join([prefix, unit])
        for prefix in valid_prefix_names
        for unit in valid_unit_names
    ]
    valid_units += [
        "".join([prefix, unit])
        for prefix in valid_prefix_names
        for unit in valid_unit_symbols
    ]
    valid_units += [
        "".join([prefix, unit])
        for prefix in valid_prefix_symbols
        for unit in valid_unit_names
    ]
    valid_units += [
        "".join([prefix, unit])
        for prefix in valid_prefix_symbols
        for unit in valid_unit_symbols
    ]

    # units are also valid without a prefix
    valid_units += valid_unit_names
    valid_units += valid_unit_symbols

    # we also accept "n/a" as a unit, which is the default missing value in
    # BIDS
    valid_units += ["n/a"]

    return tuple(valid_units)


@verbose
def _unique_channel_names(ch_names, max_length=None, verbose=None):
    """Ensure unique channel names."""
    suffixes = tuple(string.ascii_lowercase)
    if max_length is not None:
        ch_names[:] = [name[:max_length] for name in ch_names]
    unique_ids = np.unique(ch_names, return_index=True)[1]
    if len(unique_ids) != len(ch_names):
        dups = {ch_names[x] for x in np.setdiff1d(range(len(ch_names)), unique_ids)}
        warn(
            "Channel names are not unique, found duplicates for: "
            f"{dups}. Applying running numbers for duplicates."
        )
        for ch_stem in dups:
            overlaps = np.where(np.array(ch_names) == ch_stem)[0]
            # We need an extra character since we append '-'.
            # np.ceil(...) is the maximum number of appended digits.
            if max_length is not None:
                n_keep = max_length - 1 - int(np.ceil(np.log10(len(overlaps))))
            else:
                n_keep = np.inf
            n_keep = min(len(ch_stem), n_keep)
            ch_stem = ch_stem[:n_keep]
            for idx, ch_idx in enumerate(overlaps):
                # try idx first, then loop through lower case chars
                for suffix in (idx,) + suffixes:
                    ch_name = ch_stem + f"-{suffix}"
                    if ch_name not in ch_names:
                        break
                if ch_name not in ch_names:
                    ch_names[ch_idx] = ch_name
                else:
                    raise ValueError(
                        "Adding a single alphanumeric for a "
                        "duplicate resulted in another "
                        f"duplicate name {ch_name}"
                    )
    return ch_names


# %% Mixin classes


class MontageMixin:
    """Mixin for Montage getting and setting."""

    @fill_doc
    def get_montage(self):
        """Get a DigMontage from instance.

        Returns
        -------
        montage : None | DigMontage
            A copy of the channel positions, if available, otherwise ``None``.
        """
        from ..channels.montage import make_dig_montage
        from ..transforms import _frame_to_str

        info = self if isinstance(self, Info) else self.info
        if info["dig"] is None:
            return None
        # obtain coord_frame, and landmark coords
        # (nasion, lpa, rpa, hsp, hpi) from DigPoints
        montage_bunch = _get_data_as_dict_from_dig(info["dig"])
        coord_frame = _frame_to_str.get(montage_bunch.coord_frame)

        # get the channel names and chs data structure
        ch_names, chs = info["ch_names"], info["chs"]
        picks = pick_types(
            info,
            meg=False,
            eeg=True,
            seeg=True,
            ecog=True,
            dbs=True,
            fnirs=True,
            exclude=[],
        )

        # channel positions from dig do not match ch_names one to one,
        # so use loc[:3] instead
        ch_pos = {ch_names[ii]: chs[ii]["loc"][:3] for ii in picks}

        # fNIRS uses multiple channels for the same sensors, we use
        # a private function to format these for dig montage.
        fnirs_picks = pick_types(info, fnirs=True, exclude=[])
        if len(ch_pos) == len(fnirs_picks):
            ch_pos = _get_fnirs_ch_pos(info)
        elif len(fnirs_picks) > 0:
            raise ValueError(
                "MNE does not support getting the montage "
                "for a mix of fNIRS and other data types. "
                "Please raise a GitHub issue if you "
                "require this feature."
            )

        # create montage
        montage = make_dig_montage(
            ch_pos=ch_pos,
            coord_frame=coord_frame,
            nasion=montage_bunch.nasion,
            lpa=montage_bunch.lpa,
            rpa=montage_bunch.rpa,
            hsp=montage_bunch.hsp,
            hpi=montage_bunch.hpi,
        )
        return montage

    @verbose
    def set_montage(
        self,
        montage,
        match_case=True,
        match_alias=False,
        on_missing="raise",
        verbose=None,
    ):
        """Set %(montage_types)s channel positions and digitization points.

        Parameters
        ----------
        %(montage)s
        %(match_case)s
        %(match_alias)s
        %(on_missing_montage)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The instance, modified in-place.

        See Also
        --------
        mne.channels.make_standard_montage
        mne.channels.make_dig_montage
        mne.channels.read_custom_montage

        Notes
        -----
        .. warning::
            Only %(montage_types)s channels can have their positions set using
            a montage. Other channel types (e.g., MEG channels) should have
            their positions defined properly using their data reading
            functions.
        .. warning::
            Applying a montage will only set locations of channels that exist
            at the time it is applied. This means when
            :ref:`re-referencing <tut-set-eeg-ref>`
            make sure to apply the montage only after calling
            :func:`mne.add_reference_channels`
        """
        # How to set up a montage to old named fif file (walk through example)
        # https://gist.github.com/massich/f6a9f4799f1fbeb8f5e8f8bc7b07d3df

        from ..channels.montage import _set_montage

        info = self if isinstance(self, Info) else self.info
        _set_montage(info, montage, match_case, match_alias, on_missing)
        return self


channel_type_constants = get_channel_type_constants(include_defaults=True)
_human2fiff = {
    k: v.get("kind", FIFF.FIFFV_COIL_NONE) for k, v in channel_type_constants.items()
}
_human2unit = {
    k: v.get("unit", FIFF.FIFF_UNIT_NONE) for k, v in channel_type_constants.items()
}
_unit2human = {
    FIFF.FIFF_UNIT_V: "V",
    FIFF.FIFF_UNIT_T: "T",
    FIFF.FIFF_UNIT_T_M: "T/m",
    FIFF.FIFF_UNIT_MOL: "M",
    FIFF.FIFF_UNIT_NONE: "NA",
    FIFF.FIFF_UNIT_CEL: "C",
    FIFF.FIFF_UNIT_S: "S",
    FIFF.FIFF_UNIT_PX: "px",
}


def _check_set(ch, projs, ch_type):
    """Ensure type change is compatible with projectors."""
    new_kind = _human2fiff[ch_type]
    if ch["kind"] != new_kind:
        for proj in projs:
            if ch["ch_name"] in proj["data"]["col_names"]:
                raise RuntimeError(
                    f"Cannot change channel type for channel {ch['ch_name']} in "
                    f'projector "{proj["desc"]}"'
                )
    ch["kind"] = new_kind


class SetChannelsMixin(MontageMixin):
    """Mixin class for Raw, Evoked, Epochs."""

    def _get_channel_positions(self, picks=None):
        """Get channel locations from info.

        Parameters
        ----------
        picks : str | list | slice | None
            None gets good data indices.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        info = self if isinstance(self, Info) else self.info
        picks = _picks_to_idx(info, picks)
        chs = info["chs"]
        pos = np.array([chs[k]["loc"][:3] for k in picks])
        n_zero = np.sum(np.sum(np.abs(pos), axis=1) == 0)
        if n_zero > 1:  # XXX some systems have origin (0, 0, 0)
            raise ValueError(
                f"Could not extract channel positions for {n_zero} channels"
            )
        return pos

    def _set_channel_positions(self, pos, names):
        """Update channel locations in info.

        Parameters
        ----------
        pos : array-like | np.ndarray, shape (n_points, 3)
            The channel positions to be set.
        names : list of str
            The names of the channels to be set.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        info = self if isinstance(self, Info) else self.info
        if len(pos) != len(names):
            raise ValueError(
                "Number of channel positions not equal to the number of names given."
            )
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[-1] != 3 or pos.ndim != 2:
            msg = (
                f"Channel positions must have the shape (n_points, 3) not {pos.shape}."
            )
            raise ValueError(msg)
        for name, p in zip(names, pos):
            if name in self.ch_names:
                idx = self.ch_names.index(name)
                info["chs"][idx]["loc"][:3] = p
            else:
                msg = f"{name} was not found in the info. Cannot be updated."
                raise ValueError(msg)

    @verbose
    def set_channel_types(self, mapping, *, on_unit_change="warn", verbose=None):
        """Specify the sensor types of channels.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping channel names to sensor types, e.g.,
            ``{'EEG061': 'eog'}``.
        on_unit_change : ``'raise'`` | ``'warn'`` | ``'ignore'``
            What to do if the measurement unit of a channel is changed
            automatically to match the new sensor type.

            .. versionadded:: 1.4
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The instance (modified in place).

            .. versionchanged:: 0.20
               Return the instance.

        Notes
        -----
        The following :term:`sensor types` are accepted:

            bio, chpi, csd, dbs, dipole, ecg, ecog, eeg, emg, eog, exci,
            eyegaze, fnirs_cw_amplitude, fnirs_fd_ac_amplitude, fnirs_fd_phase,
            fnirs_od, gof, gsr, hbo, hbr, ias, misc, pupil, ref_meg, resp,
            seeg, stim, syst, temperature.

        When working with eye-tracking data, see
        :func:`mne.preprocessing.eyetracking.set_channel_types_eyetrack`.

        .. versionadded:: 0.9.0
        """
        info = self if isinstance(self, Info) else self.info
        ch_names = info["ch_names"]

        # first check and assemble clean mappings of index and name
        unit_changes = dict()
        for ch_name, ch_type in mapping.items():
            if ch_name not in ch_names:
                raise ValueError(
                    f"This channel name ({ch_name}) doesn't exist in info."
                )

            c_ind = ch_names.index(ch_name)
            if ch_type not in _human2fiff:
                raise ValueError(
                    f"This function cannot change to this channel type: {ch_type}. "
                    "Accepted channel types are "
                    f"{', '.join(sorted(_human2unit.keys()))}."
                )
            # Set sensor type
            _check_set(info["chs"][c_ind], info["projs"], ch_type)
            unit_old = info["chs"][c_ind]["unit"]
            unit_new = _human2unit[ch_type]
            if unit_old not in _unit2human:
                raise ValueError(
                    f"Channel '{ch_name}' has unknown unit ({unit_old}). Please fix the"
                    " measurement info of your data."
                )
            if unit_old != _human2unit[ch_type]:
                this_change = (_unit2human[unit_old], _unit2human[unit_new])
                if this_change not in unit_changes:
                    unit_changes[this_change] = list()
                unit_changes[this_change].append(ch_name)
                # reset unit multiplication factor since the unit has now changed
                info["chs"][c_ind]["unit_mul"] = _ch_unit_mul_named[0]
            info["chs"][c_ind]["unit"] = _human2unit[ch_type]
            if ch_type in ["eeg", "seeg", "ecog", "dbs"]:
                coil_type = FIFF.FIFFV_COIL_EEG
            elif ch_type == "hbo":
                coil_type = FIFF.FIFFV_COIL_FNIRS_HBO
            elif ch_type == "hbr":
                coil_type = FIFF.FIFFV_COIL_FNIRS_HBR
            elif ch_type == "fnirs_cw_amplitude":
                coil_type = FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE
            elif ch_type == "fnirs_fd_ac_amplitude":
                coil_type = FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE
            elif ch_type == "fnirs_fd_phase":
                coil_type = FIFF.FIFFV_COIL_FNIRS_FD_PHASE
            elif ch_type == "fnirs_od":
                coil_type = FIFF.FIFFV_COIL_FNIRS_OD
            elif ch_type == "eyetrack_pos":
                coil_type = FIFF.FIFFV_COIL_EYETRACK_POS
            elif ch_type == "eyetrack_pupil":
                coil_type = FIFF.FIFFV_COIL_EYETRACK_PUPIL
            else:
                coil_type = FIFF.FIFFV_COIL_NONE
            info["chs"][c_ind]["coil_type"] = coil_type

        msg = "The unit for channel(s) {0} has changed from {1} to {2}."
        for this_change, names in unit_changes.items():
            _on_missing(
                on_missing=on_unit_change,
                msg=msg.format(", ".join(sorted(names)), *this_change),
                name="on_unit_change",
            )

        return self

    @verbose
    def rename_channels(self, mapping, allow_duplicates=False, *, verbose=None):
        """Rename channels.

        Parameters
        ----------
        %(mapping_rename_channels_duplicates)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The instance (modified in place).

            .. versionchanged:: 0.20
               Return the instance.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from ..channels.channels import rename_channels
        from ..io import BaseRaw

        info = self if isinstance(self, Info) else self.info

        ch_names_orig = list(info["ch_names"])
        rename_channels(info, mapping, allow_duplicates)

        # Update self._orig_units for Raw
        if isinstance(self, BaseRaw):
            # whatever mapping was provided, now we can just use a dict
            mapping = dict(zip(ch_names_orig, info["ch_names"]))
            for old_name, new_name in mapping.items():
                if old_name in self._orig_units:
                    self._orig_units[new_name] = self._orig_units.pop(old_name)
            ch_names = self.annotations.ch_names
            for ci, ch in enumerate(ch_names):
                ch_names[ci] = tuple(mapping.get(name, name) for name in ch)

        return self

    @verbose
    def plot_sensors(
        self,
        kind="topomap",
        ch_type=None,
        title=None,
        show_names=False,
        ch_groups=None,
        to_sphere=True,
        axes=None,
        block=False,
        show=True,
        sphere=None,
        *,
        verbose=None,
    ):
        """Plot sensor positions.

        Parameters
        ----------
        kind : str
            Whether to plot the sensors as 3d, topomap or as an interactive
            sensor selection dialog. Available options 'topomap', '3d',
            'select'. If 'select', a set of channels can be selected
            interactively by using lasso selector or clicking while holding
            control key. The selected channels are returned along with the
            figure instance. Defaults to 'topomap'.
        ch_type : None | str
            The channel type to plot. Available options ``'mag'``, ``'grad'``,
            ``'eeg'``, ``'seeg'``, ``'dbs'``, ``'ecog'``, ``'all'``. If ``'all'``, all
            the available mag, grad, eeg, seeg, dbs, and ecog channels are plotted. If
            None (default), then channels are chosen in the order given above.
        title : str | None
            Title for the figure. If None (default), equals to ``'Sensor
            positions (%%s)' %% ch_type``.
        show_names : bool | array of str
            Whether to display all channel names. If an array, only the channel
            names in the array are shown. Defaults to False.
        ch_groups : 'position' | array of shape (n_ch_groups, n_picks) | None
            Channel groups for coloring the sensors. If None (default), default
            coloring scheme is used. If 'position', the sensors are divided
            into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
            array, the channels are divided by picks given in the array.

            .. versionadded:: 0.13.0
        to_sphere : bool
            Whether to project the 3d locations to a sphere. When False, the
            sensor array appears similar as to looking downwards straight above
            the subject's head. Has no effect when kind='3d'. Defaults to True.

            .. versionadded:: 0.14.0
        axes : instance of Axes | instance of Axes3D | None
            Axes to draw the sensors to. If ``kind='3d'``, axes must be an
            instance of Axes3D. If None (default), a new axes will be created.

            .. versionadded:: 0.13.0
        block : bool
            Whether to halt program execution until the figure is closed.
            Defaults to False.

            .. versionadded:: 0.13.0
        show : bool
            Show figure if True. Defaults to True.
        %(sphere_topomap_auto)s
        %(verbose)s

        Returns
        -------
        fig : instance of Figure
            Figure containing the sensor topography.
        selection : list
            A list of selected channels. Only returned if ``kind=='select'``.

        See Also
        --------
        mne.viz.plot_layout

        Notes
        -----
        This function plots the sensor locations from the info structure using
        matplotlib. For drawing the sensors using PyVista see
        :func:`mne.viz.plot_alignment`.

        .. versionadded:: 0.12.0
        """
        from ..viz.utils import plot_sensors

        return plot_sensors(
            self if isinstance(self, Info) else self.info,
            kind=kind,
            ch_type=ch_type,
            title=title,
            show_names=show_names,
            ch_groups=ch_groups,
            to_sphere=to_sphere,
            axes=axes,
            block=block,
            show=show,
            sphere=sphere,
            verbose=verbose,
        )

    @verbose
    def anonymize(self, daysback=None, keep_his=False, verbose=None):
        """Anonymize measurement information in place.

        Parameters
        ----------
        %(daysback_anonymize_info)s
        %(keep_his_anonymize_info)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The modified instance.

        Notes
        -----
        %(anonymize_info_notes)s

        .. versionadded:: 0.13.0
        """
        info = self if isinstance(self, Info) else self.info
        anonymize_info(info, daysback=daysback, keep_his=keep_his, verbose=verbose)
        self.set_meas_date(info["meas_date"])  # unify annot update
        return self

    def set_meas_date(self, meas_date):
        """Set the measurement start date.

        Parameters
        ----------
        meas_date : datetime | float | tuple | None
            The new measurement date.
            If datetime object, it must be timezone-aware and in UTC.
            A tuple of (seconds, microseconds) or float (alias for
            ``(meas_date, 0)``) can also be passed and a datetime
            object will be automatically created. If None, will remove
            the time reference.

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The modified raw instance. Operates in place.

        See Also
        --------
        mne.io.Raw.anonymize

        Notes
        -----
        If you want to remove all time references in the file, call
        :func:`mne.io.anonymize_info(inst.info) <mne.io.anonymize_info>`
        after calling ``inst.set_meas_date(None)``.

        .. versionadded:: 0.20
        """
        from ..annotations import _handle_meas_date

        info = self if isinstance(self, Info) else self.info

        meas_date = _handle_meas_date(meas_date)
        with info._unlock():
            info["meas_date"] = meas_date

        # clear file_id and meas_id if needed
        if meas_date is None:
            for key in ("file_id", "meas_id"):
                value = info.get(key)
                if value is not None:
                    assert "msecs" not in value
                    value["secs"] = DATE_NONE[0]
                    value["usecs"] = DATE_NONE[1]
                    # The following copy is needed for a test CTF dataset
                    # otherwise value['machid'][:] = 0 would suffice
                    _tmp = value["machid"].copy()
                    _tmp[:] = 0
                    value["machid"] = _tmp

        if hasattr(self, "annotations"):
            self.annotations._orig_time = meas_date
        return self


class ContainsMixin:
    """Mixin class for Raw, Evoked, Epochs and Info."""

    def __contains__(self, ch_type):
        """Check channel type membership.

        Parameters
        ----------
        ch_type : str
            Channel type to check for. Can be e.g. ``'meg'``, ``'eeg'``,
            ``'stim'``, etc.

        Returns
        -------
        in : bool
            Whether or not the instance contains the given channel type.

        Examples
        --------
        Channel type membership can be tested as::

            >>> 'meg' in inst  # doctest: +SKIP
            True
            >>> 'seeg' in inst  # doctest: +SKIP
            False

        """
        # this method is not supported by Info object. An Info object inherits from a
        # dictionary and the 'key' in Info call is present all across MNE codebase, e.g.
        # to check for the presence of a key:
        # >>> 'bads' in info
        if ch_type == "meg":
            has_ch_type = _contains_ch_type(self.info, "mag") or _contains_ch_type(
                self.info, "grad"
            )
        else:
            has_ch_type = _contains_ch_type(self.info, ch_type)
        return has_ch_type

    @property
    def compensation_grade(self):
        """The current gradient compensation grade."""
        info = self if isinstance(self, Info) else self.info
        return get_current_comp(info)

    @fill_doc
    def get_channel_types(self, picks=None, unique=False, only_data_chs=False):
        """Get a list of channel type for each channel.

        Parameters
        ----------
        %(picks_all)s
        unique : bool
            Whether to return only unique channel types. Default is ``False``.
        only_data_chs : bool
            Whether to ignore non-data channels. Default is ``False``.

        Returns
        -------
        channel_types : list
            The channel types.
        """
        info = self if isinstance(self, Info) else self.info
        none = "data" if only_data_chs else "all"
        picks = _picks_to_idx(info, picks, none, (), allow_empty=False)
        ch_types = [channel_type(info, pick) for pick in picks]
        if only_data_chs:
            ch_types = [
                ch_type for ch_type in ch_types if ch_type in _DATA_CH_TYPES_SPLIT
            ]
        if unique:
            # set does not preserve order but dict does, so let's just use it
            ch_types = list({k: k for k in ch_types}.keys())
        return ch_types


# %% ValidatedDict class


class ValidatedDict(dict):
    _attributes = {}  # subclasses should set this to validated attributes

    def __init__(self, *args, **kwargs):
        self._unlocked = True
        super().__init__(*args, **kwargs)
        self._unlocked = False

    def __getstate__(self):
        """Get state (for pickling)."""
        return {"_unlocked": self._unlocked}

    def __setstate__(self, state):
        """Set state (for pickling)."""
        self._unlocked = state["_unlocked"]

    def __setitem__(self, key, val):
        """Attribute setter."""
        # During unpickling, the _unlocked attribute has not been set, so
        # let __setstate__ do it later and act unlocked now
        unlocked = getattr(self, "_unlocked", True)
        if key in self._attributes:
            if isinstance(self._attributes[key], str):
                if not unlocked:
                    raise RuntimeError(self._attributes[key])
            else:
                val = self._attributes[key](
                    val, info=self
                )  # attribute checker function
        else:
            class_name = self.__class__.__name__
            extra = ""
            if "temp" in self._attributes:
                var_name = _camel_to_snake(class_name)
                extra = (
                    f"You can set {var_name}['temp'] to store temporary objects in "
                    f"{class_name} instances, but these will not survive an I/O "
                    "round-trip."
                )
            raise RuntimeError(
                f"{class_name} does not support directly setting the key {repr(key)}. "
                + extra
            )
        super().__setitem__(key, val)

    def update(self, other=None, **kwargs):
        """Update method using __setitem__()."""
        iterable = other.items() if isinstance(other, Mapping) else other
        if other is not None:
            for key, val in iterable:
                self[key] = val
        for key, val in kwargs.items():
            self[key] = val

    def copy(self):
        """Copy the instance.

        Returns
        -------
        info : instance of Info
            The copied info.
        """
        return deepcopy(self)

    def __repr__(self):
        """Return a string representation."""
        mapping = ", ".join(f"{key}: {val}" for key, val in self.items())
        return f"<{_camel_to_snake(self.__class__.__name__)} | {mapping}>"


# %% Subject info


def _check_types(x, *, info, name, types, cast=None):
    _validate_type(x, types, name)
    if cast is not None and x is not None:
        x = cast(x)
    return x


def _check_bday(birthday_input, *, info):
    date = _check_types(
        birthday_input,
        info=info,
        name='subject_info["birthday"]',
        types=(datetime.date, None),
    )
    # test if we have a pd.Timestamp
    if hasattr(date, "date"):
        date = date.date()
    return date


class SubjectInfo(ValidatedDict):
    _attributes = {
        "id": partial(_check_types, name='subject_info["id"]', types=int),
        "his_id": partial(_check_types, name='subject_info["his_id"]', types=str),
        "last_name": partial(_check_types, name='subject_info["last_name"]', types=str),
        "first_name": partial(
            _check_types, name='subject_info["first_name"]', types=str
        ),
        "middle_name": partial(
            _check_types, name='subject_info["middle_name"]', types=str
        ),
        "birthday": partial(_check_bday),
        "sex": partial(_check_types, name='subject_info["sex"]', types=int),
        "hand": partial(_check_types, name='subject_info["hand"]', types=int),
        "weight": partial(
            _check_types, name='subject_info["weight"]', types="numeric", cast=float
        ),
        "height": partial(
            _check_types, name='subject_info["height"]', types="numeric", cast=float
        ),
    }

    def __init__(self, initial):
        _validate_type(initial, dict, "subject_info")
        super().__init__()
        for key, val in initial.items():
            self[key] = val


class HeliumInfo(ValidatedDict):
    _attributes = {
        "he_level_raw": partial(
            _check_types,
            name='helium_info["he_level_raw"]',
            types="numeric",
            cast=float,
        ),
        "helium_level": partial(
            _check_types,
            name='helium_info["helium_level"]',
            types="numeric",
            cast=float,
        ),
        "orig_file_guid": partial(
            _check_types, name='helium_info["orig_file_guid"]', types=str
        ),
        "meas_date": partial(
            _check_types,
            name='helium_info["meas_date"]',
            types=(datetime.datetime, None),
        ),
    }

    def __init__(self, initial):
        _validate_type(initial, dict, "helium_info")
        super().__init__()
        for key, val in initial.items():
            self[key] = val


# %% Info class and helpers


def _format_trans(obj, key):
    from ..transforms import Transform

    try:
        t = obj[key]
    except KeyError:
        pass
    else:
        if t is not None:
            obj[key] = Transform(t["from"], t["to"], t["trans"])


def _check_ch_keys(ch, ci, name='info["chs"]', check_min=True):
    ch_keys = set(ch)
    bad = sorted(ch_keys.difference(_ALL_CH_KEYS_SET))
    if bad:
        raise KeyError(f"key{_pl(bad)} errantly present for {name}[{ci}]: {bad}")
    if check_min:
        bad = sorted(_MIN_CH_KEYS_SET.difference(ch_keys))
        if bad:
            raise KeyError(
                f"key{_pl(bad)} missing for {name}[{ci}]: {bad}",
            )


def _check_bads_info_compat(bads, info):
    _validate_type(bads, list, "bads")
    if not len(bads):
        return  # e.g. in empty_info
    for bi, bad in enumerate(bads):
        _validate_type(bad, str, f"bads[{bi}]")
    if "ch_names" not in info:  # somewhere in init, or deepcopy, or _empty_info, etc.
        return
    missing = [bad for bad in bads if bad not in info["ch_names"]]
    if len(missing) > 0:
        raise ValueError(f"bad channel(s) {missing} marked do not exist in info")


class MNEBadsList(list):
    """Subclass of bads that checks inplace operations."""

    def __init__(self, *, bads, info):
        _check_bads_info_compat(bads, info)
        self._mne_info = info
        super().__init__(bads)

    def extend(self, iterable):
        if not isinstance(iterable, list):
            iterable = list(iterable)
        # can happen during pickling
        try:
            info = self._mne_info
        except AttributeError:
            pass  # can happen during pickling
        else:
            _check_bads_info_compat(iterable, info)
        return super().extend(iterable)

    def append(self, x):
        return self.extend([x])

    def __iadd__(self, x):
        self.extend(x)
        return self


# As options are added here, test_meas_info.py:test_info_bad should be updated
def _check_bads(bads, *, info):
    return MNEBadsList(bads=bads, info=info)


def _check_dev_head_t(dev_head_t, *, info):
    from ..transforms import Transform, _ensure_trans

    _validate_type(dev_head_t, (Transform, None), "info['dev_head_t']")
    if dev_head_t is not None:
        dev_head_t = _ensure_trans(dev_head_t, "meg", "head")
    return dev_head_t


# TODO: Add fNIRS convention to loc
class Info(ValidatedDict, SetChannelsMixin, MontageMixin, ContainsMixin):
    """Measurement information.

    This data structure behaves like a dictionary. It contains all metadata
    that is available for a recording. However, its keys are restricted to
    those provided by the
    `FIF format specification <https://github.com/mne-tools/fiff-constants>`__,
    so new entries should not be manually added.

    .. note::
        This class should not be instantiated directly via
        ``mne.Info(...)``. Instead, use :func:`mne.create_info` to create
        measurement information from scratch.

    .. warning::
        The only entries that should be manually changed by the user are:
        ``info['bads']``, ``info['description']``, ``info['device_info']``,
        ``info['proj_id']``, ``info['proj_name']``, ``info['dev_head_t']``,
        ``info['experimenter']``, ``info['helium_info']``,
        ``info['line_freq']``, ``info['temp']``, and ``info['subject_info']``.

        All other entries should be considered read-only, though they can be
        modified by various MNE-Python functions or methods (which have
        safeguards to ensure all fields remain in sync).

    Parameters
    ----------
    *args : list
        Arguments.
    **kwargs : dict
        Keyword arguments.

    Attributes
    ----------
    acq_pars : str | None
        MEG system acquisition parameters.
        See :class:`mne.AcqParserFIF` for details.
    acq_stim : str | None
        MEG system stimulus parameters.
    bads : list of str
        List of bad (noisy/broken) channels, by name. These channels will by
        default be ignored by many processing steps.
    ch_names : list of str
        The names of the channels.
    chs : list of dict
        A list of channel information dictionaries, one per channel.
        See Notes for more information.
    command_line : str
        Contains the command and arguments used to create the source space
        (used for source estimation).
    comps : list of dict
        CTF software gradient compensation data.
        See Notes for more information.
    ctf_head_t : Transform | None
        The transformation from 4D/CTF head coordinates to Neuromag head
        coordinates. This is only present in 4D/CTF data.
    custom_ref_applied : int
        Whether a custom (=other than an average projector) reference has been
        applied to the EEG data. This flag is checked by some algorithms that
        require an average reference to be set.
    description : str | None
        String description of the recording.
    dev_ctf_t : Transform | None
        The transformation from device coordinates to 4D/CTF head coordinates.
        This is only present in 4D/CTF data.
    dev_head_t : Transform | None
        The device to head transformation.
    device_info : dict | None
        Information about the acquisition device. See Notes for details.

        .. versionadded:: 0.19
    dig : list of dict | None
        The Polhemus digitization data in head coordinates.
        See Notes for more information.
    events : list of dict
        Event list, sometimes extracted from the stim channels by Neuromag
        systems. In general this should not be used and
        :func:`mne.find_events` should be used for event processing.
        See Notes for more information.
    experimenter : str | None
        Name of the person that ran the experiment.
    file_id : dict | None
        The FIF globally unique ID. See Notes for more information.
    gantry_angle : float | None
        Tilt angle of the gantry in degrees.
    helium_info : dict | None
        Information about the device helium. See Notes for details.

        .. versionadded:: 0.19
    highpass : float
        Highpass corner frequency in Hertz. Zero indicates a DC recording.
    hpi_meas : list of dict
        HPI measurements that were taken at the start of the recording
        (e.g. coil frequencies).
        See Notes for details.
    hpi_results : list of dict
        Head position indicator (HPI) digitization points and fit information
        (e.g., the resulting transform).
        See Notes for details.
    hpi_subsystem : dict | None
        Information about the HPI subsystem that was used (e.g., event
        channel used for cHPI measurements).
        See Notes for details.
    kit_system_id : int
        Identifies the KIT system.
    line_freq : float | None
        Frequency of the power line in Hertz.
    lowpass : float
        Lowpass corner frequency in Hertz.
        It is automatically set to half the sampling rate if there is
        otherwise no low-pass applied to the data.
    maxshield : bool
        True if active shielding (IAS) was active during recording.
    meas_date : datetime
        The time (UTC) of the recording.

        .. versionchanged:: 0.20
           This is stored as a :class:`~python:datetime.datetime` object
           instead of a tuple of seconds/microseconds.
    meas_file : str | None
        Raw measurement file (used for source estimation).
    meas_id : dict | None
        The ID assigned to this measurement by the acquisition system or
        during file conversion. Follows the same format as ``file_id``.
    mri_file : str | None
        File containing the MRI to head transformation (used for source
        estimation).
    mri_head_t : dict | None
        Transformation from MRI to head coordinates (used for source
        estimation).
    mri_id : dict | None
        MRI unique ID (used for source estimation).
    nchan : int
        Number of channels.
    proc_history : list of dict
        The MaxFilter processing history.
        See Notes for details.
    proj_id : int | None
        ID number of the project the experiment belongs to.
    proj_name : str | None
        Name of the project the experiment belongs to.
    projs : list of Projection
        List of SSP operators that operate on the data.
        See :class:`mne.Projection` for details.
    sfreq : float
        Sampling frequency in Hertz.
    subject_info : dict | None
        Information about the subject.
        See Notes for details.
    temp : object | None
        Can be used to store temporary objects in an Info instance. It will not
        survive an I/O roundtrip.

        .. versionadded:: 0.24
    utc_offset : str
        "UTC offset of related meas_date (sHH:MM).

        .. versionadded:: 0.19
    working_dir : str
        Working directory used when the source space was created (used for
        source estimation).
    xplotter_layout : str
        Layout of the Xplotter (Neuromag system only).

    See Also
    --------
    mne.create_info
    mne.pick_info

    Notes
    -----
    The following parameters have a nested structure.

    * ``chs`` list of dict:

        cal : float
            The calibration factor to bring the channels to physical
            units. Used in product with ``range`` to scale the data read
            from disk.
        ch_name : str
            The channel name.
        coil_type : int
            Coil type, e.g. ``FIFFV_COIL_MEG``.
        coord_frame : int
            The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
        kind : int
            The kind of channel, e.g. ``FIFFV_EEG_CH``.
        loc : array, shape (12,)
            Channel location information. The first three elements ``[:3]`` always store
            the nominal channel position. The remaining 9 elements store different
            information based on the channel type:

            MEG
                Remaining 9 elements ``[3:]``, contain the EX, EY, and EZ normal
                triplets (columns) of the coil rotation/orientation matrix.
            EEG
                Elements ``[3:6]`` contain the reference channel position.
            Eyetrack
                Element ``[3]`` contains information about which eye was tracked
                (-1 for left, 1 for right), and element ``[4]`` contains information
                about the the axis of coordinate data (-1 for x-coordinate data, 1 for
                y-coordinate data).
            Dipole
                Elements ``[3:6]`` contain dipole orientation information.
        logno : int
            Logical channel number, conventions in the usage of this
            number vary.
        range : float
            The hardware-oriented part of the calibration factor.
            This should be only applied to the continuous raw data.
            Used in product with ``cal`` to scale data read from disk.
        scanno : int
            Scanning order number, starting from 1.
        unit : int
            The unit to use, e.g. ``FIFF_UNIT_T_M``.
        unit_mul : int
            Unit multipliers, most commonly ``FIFF_UNITM_NONE``.

    * ``comps`` list of dict:

        ctfkind : int
            CTF compensation grade.
        colcals : ndarray
            Column calibrations.
        mat : dict
            A named matrix dictionary (with entries "data", "col_names", etc.)
            containing the compensation matrix.
        rowcals : ndarray
            Row calibrations.
        save_calibrated : bool
            Were the compensation data saved in calibrated form.

    * ``device_info`` dict:

        type : str
            Device type.
        model : str
            Device model.
        serial : str
            Device serial.
        site : str
            Device site.

    * ``dig`` list of dict:

        kind : int
            The kind of channel,
            e.g. ``FIFFV_POINT_EEG``, ``FIFFV_POINT_CARDINAL``.
        r : array, shape (3,)
            3D position in m. and coord_frame.
        ident : int
            Number specifying the identity of the point.
            e.g. ``FIFFV_POINT_NASION`` if kind is ``FIFFV_POINT_CARDINAL``, or
            42 if kind is ``FIFFV_POINT_EEG``.
        coord_frame : int
            The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.

    * ``events`` list of dict:

        channels : list of int
            Channel indices for the events.
        list : ndarray, shape (n_events * 3,)
            Events in triplets as number of samples, before, after.

    * ``file_id`` dict:

        version : int
            FIF format version, i.e. ``FIFFC_VERSION``.
        machid : ndarray, shape (2,)
            Unique machine ID, usually derived from the MAC address.
        secs : int
            Time in seconds.
        usecs : int
            Time in microseconds.

    * ``helium_info`` dict:

        he_level_raw : float
            Helium level (%) before position correction.
        helium_level : float
            Helium level (%) after position correction.
        orig_file_guid : str
            Original file GUID.
        meas_date : datetime.datetime | None
            The helium level meas date.

            .. versionchanged:: 1.8
               This is stored as a :class:`~python:datetime.datetime` object
               instead of a tuple of seconds/microseconds.

    * ``hpi_meas`` list of dict:

        creator : str
            Program that did the measurement.
        sfreq : float
            Sample rate.
        nchan : int
            Number of channels used.
        nave : int
            Number of averages used.
        ncoil : int
            Number of coils used.
        first_samp : int
            First sample used.
        last_samp : int
            Last sample used.
        hpi_coils : list of dict
            Coils, containing:

                number: int
                    Coil number
                epoch : ndarray
                    Buffer containing one epoch and channel.
                slopes : ndarray, shape (n_channels,)
                    HPI data.
                corr_coeff : ndarray, shape (n_channels,)
                    HPI curve fit correlations.
                coil_freq : float
                    HPI coil excitation frequency

    * ``hpi_results`` list of dict:

        dig_points : list
            Digitization points (see ``dig`` definition) for the HPI coils.
        order : ndarray, shape (ncoil,)
            The determined digitization order.
        used : ndarray, shape (nused,)
            The indices of the used coils.
        moments : ndarray, shape (ncoil, 3)
            The coil moments.
        goodness : ndarray, shape (ncoil,)
            The goodness of fits.
        good_limit : float
            The goodness of fit limit.
        dist_limit : float
            The distance limit.
        accept : int
            Whether or not the fit was accepted.
        coord_trans : instance of Transform
            The resulting MEG<->head transformation.

    * ``hpi_subsystem`` dict:

        ncoil : int
            The number of coils.
        event_channel : str
            The event channel used to encode cHPI status (e.g., STI201).
        hpi_coils : list of ndarray
            List of length ``ncoil``, each 4-element ndarray contains the
            event bits used on the event channel to indicate cHPI status
            (using the first element of these arrays is typically
            sufficient).

    * ``mri_id`` dict:

        version : int
            FIF format version, i.e. ``FIFFC_VERSION``.
        machid : ndarray, shape (2,)
            Unique machine ID, usually derived from the MAC address.
        secs : int
            Time in seconds.
        usecs : int
            Time in microseconds.

    * ``proc_history`` list of dict:

        block_id : dict
            See ``id`` above.
        date : ndarray, shape (2,)
            2-element tuple of seconds and microseconds.
        experimenter : str
            Name of the person who ran the program.
        creator : str
            Program that did the processing.
        max_info : dict
            Maxwel filtering info, can contain:

                sss_info : dict
                    SSS processing information.
                max_st
                    tSSS processing information.
                sss_ctc : dict
                    Cross-talk processing information.
                sss_cal : dict
                    Fine-calibration information.
        smartshield : dict
            MaxShield information. This dictionary is (always?) empty,
            but its presence implies that MaxShield was used during
            acquisition.

    * ``subject_info`` dict:

        id : int
            Integer subject identifier.
        his_id : str
            String subject identifier.
        last_name : str
            Last name.
        first_name : str
            First name.
        middle_name : str
            Middle name.
        birthday : datetime.date
            The subject birthday.

            .. versionchanged:: 1.8
               This is stored as a :class:`~python:datetime.date` object
               instead of a tuple of seconds/microseconds.
        sex : int
            Subject sex (0=unknown, 1=male, 2=female).
        hand : int
            Handedness (1=right, 2=left, 3=ambidextrous).
        weight : float
            Weight in kilograms.
        height : float
            Height in meters.
    """

    _attributes = {
        "acq_pars": "acq_pars cannot be set directly. "
        "See mne.AcqParserFIF() for details.",
        "acq_stim": "acq_stim cannot be set directly.",
        "bads": _check_bads,
        "ch_names": "ch_names cannot be set directly. "
        "Please use methods inst.add_channels(), "
        "inst.drop_channels(), inst.pick(), "
        "inst.rename_channels(), inst.reorder_channels() "
        "and inst.set_channel_types() instead.",
        "chs": "chs cannot be set directly. "
        "Please use methods inst.add_channels(), "
        "inst.drop_channels(), inst.pick(), "
        "inst.rename_channels(), inst.reorder_channels() "
        "and inst.set_channel_types() instead.",
        "command_line": "command_line cannot be set directly.",
        "comps": "comps cannot be set directly. "
        "Please use method Raw.apply_gradient_compensation() "
        "instead.",
        "ctf_head_t": "ctf_head_t cannot be set directly.",
        "custom_ref_applied": "custom_ref_applied cannot be set directly. "
        "Please use method inst.set_eeg_reference() "
        "instead.",
        "description": partial(_check_types, name="description", types=(str, None)),
        "dev_ctf_t": "dev_ctf_t cannot be set directly.",
        "dev_head_t": _check_dev_head_t,
        "device_info": partial(_check_types, name="device_info", types=(dict, None)),
        "dig": "dig cannot be set directly. "
        "Please use method inst.set_montage() instead.",
        "events": "events cannot be set directly.",
        "experimenter": partial(_check_types, name="experimenter", types=(str, None)),
        "file_id": "file_id cannot be set directly.",
        "gantry_angle": "gantry_angle cannot be set directly.",
        "helium_info": partial(
            _check_types, name="helium_info", types=(dict, None), cast=HeliumInfo
        ),
        "highpass": "highpass cannot be set directly. "
        "Please use method inst.filter() instead.",
        "hpi_meas": "hpi_meas can not be set directly.",
        "hpi_results": "hpi_results cannot be set directly.",
        "hpi_subsystem": "hpi_subsystem cannot be set directly.",
        "kit_system_id": "kit_system_id cannot be set directly.",
        "line_freq": partial(
            _check_types, name="line_freq", types=("numeric", None), cast=float
        ),
        "lowpass": "lowpass cannot be set directly. "
        "Please use method inst.filter() instead.",
        "maxshield": "maxshield cannot be set directly.",
        "meas_date": "meas_date cannot be set directly. "
        "Please use method inst.set_meas_date() instead.",
        "meas_file": "meas_file cannot be set directly.",
        "meas_id": "meas_id cannot be set directly.",
        "mri_file": "mri_file cannot be set directly.",
        "mri_head_t": "mri_head_t cannot be set directly.",
        "mri_id": "mri_id cannot be set directly.",
        "nchan": "nchan cannot be set directly. "
        "Please use methods inst.add_channels(), "
        "inst.drop_channels(), and inst.pick() instead.",
        "proc_history": "proc_history cannot be set directly.",
        "proj_id": partial(_check_types, name="proj_id", types=(int, None), cast=int),
        "proj_name": partial(_check_types, name="proj_name", types=(str, None)),
        "projs": "projs cannot be set directly. "
        "Please use methods inst.add_proj() and inst.del_proj() "
        "instead.",
        "sfreq": "sfreq cannot be set directly. "
        "Please use method inst.resample() instead.",
        "subject_info": partial(
            _check_types, name="subject_info", types=(dict, None), cast=SubjectInfo
        ),
        "temp": lambda x, info=None: x,
        "utc_offset": "utc_offset cannot be set directly.",
        "working_dir": "working_dir cannot be set directly.",
        "xplotter_layout": "xplotter_layout cannot be set directly.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unlocked = True
        # Deal with h5io writing things as dict
        if "bads" in self:
            self["bads"] = MNEBadsList(bads=self["bads"], info=self)
        for key in ("dev_head_t", "ctf_head_t", "dev_ctf_t"):
            _format_trans(self, key)
        for res in self.get("hpi_results", []):
            _format_trans(res, "coord_trans")
        if self.get("dig", None) is not None and len(self["dig"]):
            if isinstance(self["dig"], dict):  # needs to be unpacked
                self["dig"] = _dict_unpack(self["dig"], _DIG_CAST)
            if not isinstance(self["dig"][0], DigPoint):
                self["dig"] = _format_dig_points(self["dig"])
        if isinstance(self.get("chs", None), dict):
            self["chs"]["ch_name"] = [
                str(x) for x in np.char.decode(self["chs"]["ch_name"], encoding="utf8")
            ]
            self["chs"] = _dict_unpack(self["chs"], _CH_CAST)
        for pi, proj in enumerate(self.get("projs", [])):
            if not isinstance(proj, Projection):
                self["projs"][pi] = Projection(**proj)
        # Old files could have meas_date as tuple instead of datetime
        try:
            meas_date = self["meas_date"]
        except KeyError:
            pass
        else:
            self["meas_date"] = _ensure_meas_date_none_or_dt(meas_date)
        self._unlocked = False
        # with validation and casting
        for key in ("helium_info", "subject_info"):
            if key in self:
                self[key] = self[key]

    def __setstate__(self, state):
        """Set state (for pickling)."""
        super().__setstate__(state)
        self["bads"] = MNEBadsList(bads=self["bads"], info=self)

    @contextlib.contextmanager
    def _unlock(self, *, update_redundant=False, check_after=False):
        """Context manager unlocking access to attributes."""
        # needed for nested _unlock()
        state = self._unlocked if hasattr(self, "_unlocked") else False

        self._unlocked = True
        try:
            yield
        except Exception:
            raise
        else:
            if update_redundant:
                self._update_redundant()
            if check_after:
                self._check_consistency()
        finally:
            self._unlocked = state

    def normalize_proj(self):
        """(Re-)Normalize projection vectors after subselection.

        Applying projection after sub-selecting a set of channels that
        were originally used to compute the original projection vectors
        can be dangerous (e.g., if few channels remain, most power was
        in channels that are no longer picked, etc.). By default, mne
        will emit a warning when this is done.

        This function will re-normalize projectors to use only the
        remaining channels, thus avoiding that warning. Only use this
        function if you're confident that the projection vectors still
        adequately capture the original signal of interest.
        """
        _normalize_proj(self)

    def __repr__(self):
        """Summarize info instead of printing all."""
        from ..io.kit.constants import KIT_SYSNAMES
        from ..transforms import Transform, _coord_frame_name

        MAX_WIDTH = 68
        strs = ["<Info | %s non-empty values"]
        non_empty = 0
        titles = _handle_default("titles")
        for k, v in self.items():
            if k == "ch_names":
                if v:
                    entr = shorten(", ".join(v), MAX_WIDTH, placeholder=" ...")
                else:
                    entr = "[]"  # always show
                    non_empty -= 1  # don't count as non-empty
            elif k == "bads":
                if v:
                    entr = f"{len(v)} items ("
                    entr += ", ".join(v)
                    entr = shorten(entr, MAX_WIDTH, placeholder=" ...") + ")"
                else:
                    entr = "[]"  # always show
                    non_empty -= 1  # don't count as non-empty
            elif k == "projs":
                if v:
                    entr = ", ".join(
                        p["desc"] + ": o" + ("n" if p["active"] else "ff") for p in v
                    )
                    entr = shorten(entr, MAX_WIDTH, placeholder=" ...")
                else:
                    entr = "[]"  # always show projs
                    non_empty -= 1  # don't count as non-empty
            elif k == "meas_date":
                if v is None:
                    entr = "unspecified"
                else:
                    entr = v.strftime("%Y-%m-%d %H:%M:%S %Z")
            elif k == "kit_system_id" and v is not None:
                entr = f"{v} ({KIT_SYSNAMES.get(v, 'unknown')})"
            elif k == "dig" and v is not None:
                counts = Counter(d["kind"] for d in v)
                counts = [
                    f"{counts[ii]} {_dig_kind_proper[_dig_kind_rev[ii]]}"
                    for ii in _dig_kind_ints
                    if ii in counts
                ]
                counts = f" ({', '.join(counts)})" if len(counts) else ""
                entr = f"{len(v)} item{_pl(v)}{counts}"
            elif isinstance(v, Transform):
                # show entry only for non-identity transform
                if not np.allclose(v["trans"], np.eye(v["trans"].shape[0])):
                    frame1 = _coord_frame_name(v["from"])
                    frame2 = _coord_frame_name(v["to"])
                    entr = f"{frame1} -> {frame2} transform"
                else:
                    entr = ""
            elif k in ["sfreq", "lowpass", "highpass"]:
                entr = f"{v:.1f} Hz"
            elif isinstance(v, str):
                entr = shorten(v, MAX_WIDTH, placeholder=" ...")
            elif k == "chs":
                # TODO someday we should refactor with _repr_html_ with
                # bad vs good
                ch_types = [channel_type(self, idx) for idx in range(len(v))]
                ch_counts = Counter(ch_types)
                entr = ", ".join(
                    f"{count} {titles.get(ch_type, ch_type.upper())}"
                    for ch_type, count in ch_counts.items()
                )
            elif k == "custom_ref_applied":
                entr = str(bool(v))
                if not v:
                    non_empty -= 1  # don't count if 0
            elif isinstance(v, ValidatedDict):
                entr = repr(v)
            else:
                try:
                    this_len = len(v)
                except TypeError:
                    entr = f"{v}" if v is not None else ""
                else:
                    if this_len > 0:
                        entr = f"{this_len} item{_pl(this_len)} ({type(v).__name__})"
                    else:
                        entr = ""
            if entr != "":
                non_empty += 1
                strs.append(f"{k}: {entr}")
        st = "\n ".join(sorted(strs))
        st += "\n>"
        st %= non_empty
        return st

    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        result = Info.__new__(Info)
        result._unlocked = True
        for k, v in self.items():
            # chs is roughly half the time but most are immutable
            if k == "chs":
                # dict shallow copy is fast, so use it then overwrite
                result[k] = list()
                for ch in v:
                    ch = ch.copy()  # shallow
                    ch["loc"] = ch["loc"].copy()
                    result[k].append(ch)
            elif k == "ch_names":
                # we know it's list of str, shallow okay and saves ~100 µs
                result[k] = v.copy()
            elif k == "hpi_meas":
                hms = list()
                for hm in v:
                    hm = hm.copy()
                    # the only mutable thing here is some entries in coils
                    hm["hpi_coils"] = [coil.copy() for coil in hm["hpi_coils"]]
                    # There is a *tiny* risk here that someone could write
                    # raw.info['hpi_meas'][0]['hpi_coils'][1]['epoch'] = ...
                    # and assume that info.copy() will make an actual copy,
                    # but copying these entries has a 2x slowdown penalty so
                    # probably not worth it for such a deep corner case:
                    # for coil in hpi_coils:
                    #     for key in ('epoch', 'slopes', 'corr_coeff'):
                    #         coil[key] = coil[key].copy()
                    hms.append(hm)
                result[k] = hms
            else:
                result[k] = deepcopy(v, memodict)
        result._unlocked = False
        return result

    def _check_consistency(self, prepend_error=""):
        """Do some self-consistency checks and datatype tweaks."""
        meas_date = self.get("meas_date")
        if meas_date is not None:
            if (
                not isinstance(self["meas_date"], datetime.datetime)
                or self["meas_date"].tzinfo is None
                or self["meas_date"].tzinfo is not datetime.timezone.utc
            ):
                raise RuntimeError(
                    f'{prepend_error}info["meas_date"] must be a datetime object in UTC'
                    f" or None, got {repr(self['meas_date'])!r}"
                )

        chs = [ch["ch_name"] for ch in self["chs"]]
        if (
            len(self["ch_names"]) != len(chs)
            or any(ch_1 != ch_2 for ch_1, ch_2 in zip(self["ch_names"], chs))
            or self["nchan"] != len(chs)
        ):
            raise RuntimeError(
                f"{prepend_error}info channel name inconsistency detected, please "
                "notify MNE-Python developers"
            )

        # make sure we have the proper datatypes
        with self._unlock():
            for key in ("sfreq", "highpass", "lowpass"):
                if self.get(key) is not None:
                    self[key] = float(self[key])

        for pi, proj in enumerate(self.get("projs", [])):
            _validate_type(proj, Projection, f'info["projs"][{pi}]')
            for key in ("kind", "active", "desc", "data", "explained_var"):
                if key not in proj:
                    raise RuntimeError(f"Projection incomplete, missing {key}")

        # Ensure info['chs'] has immutable entries (copies much faster)
        for ci, ch in enumerate(self["chs"]):
            _check_ch_keys(ch, ci)
            ch_name = ch["ch_name"]
            _validate_type(ch_name, str, f'info["chs"][{ci}]["ch_name"]')
            for key in _SCALAR_CH_KEYS:
                val = ch.get(key, 1)
                _validate_type(val, "numeric", f'info["chs"][{ci}][{key}]')
            loc = ch["loc"]
            if not (isinstance(loc, np.ndarray) and loc.shape == (12,)):
                raise TypeError(
                    f'Bad info: info["chs"][{ci}]["loc"] must be ndarray with '
                    f"12 elements, got {repr(loc)}"
                )

        # make sure channel names are unique
        with self._unlock():
            self["ch_names"] = _unique_channel_names(self["ch_names"])
            for idx, ch_name in enumerate(self["ch_names"]):
                self["chs"][idx]["ch_name"] = ch_name

    def _update_redundant(self):
        """Update the redundant entries."""
        with self._unlock():
            self["ch_names"] = [ch["ch_name"] for ch in self["chs"]]
            self["nchan"] = len(self["chs"])

    @property
    def ch_names(self):
        try:
            ch_names = self["ch_names"]
        except KeyError:
            ch_names = []

        return ch_names

    @repr_html
    def _repr_html_(self):
        """Summarize info for HTML representation."""
        info_template = _get_html_template("repr", "info.html.jinja")
        return info_template.render(info=self)

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Write measurement info in fif file.

        Parameters
        ----------
        fname : path-like
            The name of the file. Should end by ``'-info.fif'``.
        %(overwrite)s

            .. versionadded:: 1.10
        %(verbose)s

        See Also
        --------
        mne.io.write_info
        """
        write_info(fname, self, overwrite=overwrite)


def _simplify_info(info, *, keep=()):
    """Return a simplified info structure to speed up picking."""
    chs = [
        {key: ch[key] for key in ("ch_name", "kind", "unit", "coil_type", "loc", "cal")}
        for ch in info["chs"]
    ]
    keys = ("bads", "comps", "projs", "custom_ref_applied") + keep
    sub_info = Info((key, info[key]) for key in keys if key in info)
    with sub_info._unlock():
        sub_info["chs"] = chs
    sub_info._update_redundant()
    return sub_info


@verbose
def read_fiducials(fname, *, verbose=None):
    """Read fiducials from a fiff file.

    Parameters
    ----------
    fname : path-like
        The filename to read.
    %(verbose)s

    Returns
    -------
    pts : list of dict
        List of digitizer points (each point in a dict).
    coord_frame : int
        The coordinate frame of the points (one of
        ``mne.io.constants.FIFF.FIFFV_COORD_...``).
    """
    fname = _check_fname(fname=fname, overwrite="read", must_exist=True)
    fid, tree, _ = fiff_open(fname)
    with fid:
        pts = _read_dig_fif(fid, tree)
    return pts, pts[0]["coord_frame"]


@verbose
def write_fiducials(
    fname, pts, coord_frame="unknown", *, overwrite=False, verbose=None
):
    """Write fiducials to a fiff file.

    Parameters
    ----------
    fname : path-like
        Destination file name.
    pts : iterator of dict
        Iterator through digitizer points. Each point is a dictionary with
        the keys 'kind', 'ident' and 'r'.
    coord_frame : str | int
        The coordinate frame of the points. If a string, must be one of
        ``'meg'``, ``'mri'``, ``'mri_voxel'``, ``'head'``,
        ``'mri_tal'``, ``'ras'``, ``'fs_tal'``, ``'ctf_head'``,
        ``'ctf_meg'``, and ``'unknown'``
        If an integer, must be one of the constants defined as
        ``mne.io.constants.FIFF.FIFFV_COORD_...``.
    %(overwrite)s

        .. versionadded:: 1.0
    %(verbose)s
    """
    write_dig(fname, pts, coord_frame, overwrite=overwrite)


@verbose
def read_info(fname, verbose=None):
    """Read measurement info from a file.

    Parameters
    ----------
    fname : path-like
        File name.
    %(verbose)s

    Returns
    -------
    %(info_not_none)s
    """
    check_fname(fname, "Info", (".fif", ".fif.gz"))
    fname = _check_fname(fname, must_exist=True, overwrite="read")
    f, tree, _ = fiff_open(fname)
    with f as fid:
        info = read_meas_info(fid, tree)[0]
    return info


def read_bad_channels(fid, node):
    """Read bad channels.

    Parameters
    ----------
    fid : file
        The file descriptor.
    node : dict
        The node of the FIF tree that contains info on the bad channels.

    Returns
    -------
    bads : list
        A list of bad channel's names.
    """
    return _read_bad_channels(fid, node)


def _read_bad_channels(fid, node, ch_names_mapping):
    ch_names_mapping = {} if ch_names_mapping is None else ch_names_mapping
    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = []
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None and tag.data is not None:
                bads = _safe_name_list(tag.data, "read", "bads")
        bads[:] = _rename_list(bads, ch_names_mapping)
    return bads


def _write_bad_channels(fid, bads, ch_names_mapping):
    if bads is not None and len(bads) > 0:
        ch_names_mapping = {} if ch_names_mapping is None else ch_names_mapping
        bads = _rename_list(bads, ch_names_mapping)
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list_sanitized(fid, FIFF.FIFF_MNE_CH_NAME_LIST, bads, "bads")
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)


@verbose
def read_meas_info(fid, tree, clean_bads=False, verbose=None):
    """Read the measurement info.

    Parameters
    ----------
    fid : file
        Open file descriptor.
    tree : tree
        FIF tree structure.
    clean_bads : bool
        If True, clean info['bads'] before running consistency check.
        Should only be needed for old files where we did not check bads
        before saving.
    %(verbose)s

    Returns
    -------
    %(info_not_none)s
    meas : dict
        Node in tree that contains the info.
    """
    from ..transforms import Transform, invert_transform

    #   Find the desired blocks
    meas = dir_tree_find(tree, FIFF.FIFFB_MEAS)
    if len(meas) == 0:
        raise ValueError("Could not find measurement data")
    if len(meas) > 1:
        raise ValueError("Cannot read more that 1 measurement data")
    meas = meas[0]

    meas_info = dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    if len(meas_info) == 0:
        raise ValueError("Could not find measurement info")
    if len(meas_info) > 1:
        raise ValueError("Cannot read more that 1 measurement info")
    meas_info = meas_info[0]

    #   Read measurement info
    dev_head_t = None
    ctf_head_t = None
    dev_ctf_t = None
    meas_date = None
    utc_offset = None
    highpass = None
    lowpass = None
    nchan = None
    sfreq = None
    chs = []
    experimenter = None
    description = None
    proj_id = None
    proj_name = None
    line_freq = None
    gantry_angle = None
    custom_ref_applied = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    xplotter_layout = None
    kit_system_id = None
    for k in range(meas_info["nent"]):
        kind = meas_info["directory"][k].kind
        pos = meas_info["directory"][k].pos
        if kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = int(tag.data.item())
        elif kind == FIFF.FIFF_SFREQ:
            tag = read_tag(fid, pos)
            sfreq = float(tag.data.item())
        elif kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
        elif kind == FIFF.FIFF_LOWPASS:
            tag = read_tag(fid, pos)
            if not np.isnan(tag.data.item()):
                lowpass = float(tag.data.item())
        elif kind == FIFF.FIFF_HIGHPASS:
            tag = read_tag(fid, pos)
            if not np.isnan(tag.data):
                highpass = float(tag.data.item())
        elif kind == FIFF.FIFF_MEAS_DATE:
            tag = read_tag(fid, pos)
            meas_date = tuple(tag.data)
            if len(meas_date) == 1:  # can happen from old C conversions
                meas_date = (meas_date[0], 0)
        elif kind == FIFF.FIFF_UTC_OFFSET:
            tag = read_tag(fid, pos)
            utc_offset = str(tag.data)
        elif kind == FIFF.FIFF_COORD_TRANS:
            tag = read_tag(fid, pos)
            cand = tag.data

            if (
                cand["from"] == FIFF.FIFFV_COORD_DEVICE
                and cand["to"] == FIFF.FIFFV_COORD_HEAD
            ):
                dev_head_t = cand
            elif (
                cand["from"] == FIFF.FIFFV_COORD_HEAD
                and cand["to"] == FIFF.FIFFV_COORD_DEVICE
            ):
                # this reversal can happen with BabyMEG data
                dev_head_t = invert_transform(cand)
            elif (
                cand["from"] == FIFF.FIFFV_MNE_COORD_CTF_HEAD
                and cand["to"] == FIFF.FIFFV_COORD_HEAD
            ):
                ctf_head_t = cand
            elif (
                cand["from"] == FIFF.FIFFV_MNE_COORD_CTF_DEVICE
                and cand["to"] == FIFF.FIFFV_MNE_COORD_CTF_HEAD
            ):
                dev_ctf_t = cand
        elif kind == FIFF.FIFF_EXPERIMENTER:
            tag = read_tag(fid, pos)
            experimenter = tag.data
        elif kind == FIFF.FIFF_DESCRIPTION:
            tag = read_tag(fid, pos)
            description = tag.data
        elif kind == FIFF.FIFF_PROJ_ID:
            tag = read_tag(fid, pos)
            proj_id = int(tag.data.item())
        elif kind == FIFF.FIFF_PROJ_NAME:
            tag = read_tag(fid, pos)
            proj_name = tag.data
        elif kind == FIFF.FIFF_LINE_FREQ:
            tag = read_tag(fid, pos)
            line_freq = float(tag.data.item())
        elif kind == FIFF.FIFF_GANTRY_ANGLE:
            tag = read_tag(fid, pos)
            gantry_angle = float(tag.data.item())
        elif kind in [FIFF.FIFF_MNE_CUSTOM_REF, 236]:  # 236 used before v0.11
            tag = read_tag(fid, pos)
            custom_ref_applied = int(tag.data.item())
        elif kind == FIFF.FIFF_XPLOTTER_LAYOUT:
            tag = read_tag(fid, pos)
            xplotter_layout = str(tag.data)
        elif kind == FIFF.FIFF_MNE_KIT_SYSTEM_ID:
            tag = read_tag(fid, pos)
            kit_system_id = int(tag.data.item())
    ch_names_mapping = _read_extended_ch_info(chs, meas_info, fid)

    # Check that we have everything we need
    if nchan is None:
        raise ValueError("Number of channels is not defined")

    if sfreq is None:
        raise ValueError("Sampling frequency is not defined")

    if len(chs) == 0:
        raise ValueError("Channel information not defined")

    if len(chs) != nchan:
        raise ValueError("Incorrect number of channel definitions found")

    if dev_head_t is None or ctf_head_t is None:
        hpi_result = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
        if len(hpi_result) == 1:
            hpi_result = hpi_result[0]
            for k in range(hpi_result["nent"]):
                kind = hpi_result["directory"][k].kind
                pos = hpi_result["directory"][k].pos
                if kind == FIFF.FIFF_COORD_TRANS:
                    tag = read_tag(fid, pos)
                    cand = tag.data
                    if (
                        cand["from"] == FIFF.FIFFV_COORD_DEVICE
                        and cand["to"] == FIFF.FIFFV_COORD_HEAD
                        and dev_head_t is None
                    ):
                        dev_head_t = cand
                    elif (
                        cand["from"] == FIFF.FIFFV_MNE_COORD_CTF_HEAD
                        and cand["to"] == FIFF.FIFFV_COORD_HEAD
                        and ctf_head_t is None
                    ):
                        ctf_head_t = cand

    #   Locate the Polhemus data
    dig = _read_dig_fif(fid, meas_info)

    #   Locate the acquisition information
    acqpars = dir_tree_find(meas_info, FIFF.FIFFB_DACQ_PARS)
    acq_pars = None
    acq_stim = None
    if len(acqpars) == 1:
        acqpars = acqpars[0]
        for k in range(acqpars["nent"]):
            kind = acqpars["directory"][k].kind
            pos = acqpars["directory"][k].pos
            if kind == FIFF.FIFF_DACQ_PARS:
                tag = read_tag(fid, pos)
                acq_pars = tag.data
            elif kind == FIFF.FIFF_DACQ_STIM:
                tag = read_tag(fid, pos)
                acq_stim = tag.data

    #   Load the SSP data
    projs = _read_proj(fid, meas_info, ch_names_mapping=ch_names_mapping)

    #   Load the CTF compensation data
    comps = _read_ctf_comp(fid, meas_info, chs, ch_names_mapping=ch_names_mapping)

    #   Load the bad channel list
    bads = _read_bad_channels(fid, meas_info, ch_names_mapping=ch_names_mapping)

    #
    #   Put the data together
    #
    info = Info(file_id=tree["id"])
    info._unlocked = True

    #   Locate events list
    events = dir_tree_find(meas_info, FIFF.FIFFB_EVENTS)
    evs = list()
    for event in events:
        ev = dict()
        for k in range(event["nent"]):
            kind = event["directory"][k].kind
            pos = event["directory"][k].pos
            if kind == FIFF.FIFF_EVENT_CHANNELS:
                ev["channels"] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_EVENT_LIST:
                ev["list"] = read_tag(fid, pos).data
        evs.append(ev)
    info["events"] = evs

    #   Locate HPI result
    hpi_results = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
    hrs = list()
    for hpi_result in hpi_results:
        hr = dict()
        hr["dig_points"] = []
        for k in range(hpi_result["nent"]):
            kind = hpi_result["directory"][k].kind
            pos = hpi_result["directory"][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                hr["dig_points"].append(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_DIGITIZATION_ORDER:
                hr["order"] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_COILS_USED:
                hr["used"] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_COIL_MOMENTS:
                hr["moments"] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_FIT_GOODNESS:
                hr["goodness"] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_FIT_GOOD_LIMIT:
                hr["good_limit"] = float(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_HPI_FIT_DIST_LIMIT:
                hr["dist_limit"] = float(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_HPI_FIT_ACCEPT:
                hr["accept"] = int(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_COORD_TRANS:
                hr["coord_trans"] = read_tag(fid, pos).data
        hrs.append(hr)
    info["hpi_results"] = hrs

    #   Locate HPI Measurement
    hpi_meass = dir_tree_find(meas_info, FIFF.FIFFB_HPI_MEAS)
    hms = list()
    for hpi_meas in hpi_meass:
        hm = dict()
        for k in range(hpi_meas["nent"]):
            kind = hpi_meas["directory"][k].kind
            pos = hpi_meas["directory"][k].pos
            if kind == FIFF.FIFF_CREATOR:
                hm["creator"] = str(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_SFREQ:
                hm["sfreq"] = float(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_NCHAN:
                hm["nchan"] = int(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_NAVE:
                hm["nave"] = int(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_HPI_NCOIL:
                hm["ncoil"] = int(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_FIRST_SAMPLE:
                hm["first_samp"] = int(read_tag(fid, pos).data.item())
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                hm["last_samp"] = int(read_tag(fid, pos).data.item())
        hpi_coils = dir_tree_find(hpi_meas, FIFF.FIFFB_HPI_COIL)
        hcs = []
        for hpi_coil in hpi_coils:
            hc = dict()
            for k in range(hpi_coil["nent"]):
                kind = hpi_coil["directory"][k].kind
                pos = hpi_coil["directory"][k].pos
                if kind == FIFF.FIFF_HPI_COIL_NO:
                    hc["number"] = int(read_tag(fid, pos).data.item())
                elif kind == FIFF.FIFF_EPOCH:
                    hc["epoch"] = read_tag(fid, pos).data
                    hc["epoch"].flags.writeable = False
                elif kind == FIFF.FIFF_HPI_SLOPES:
                    hc["slopes"] = read_tag(fid, pos).data
                    hc["slopes"].flags.writeable = False
                elif kind == FIFF.FIFF_HPI_CORR_COEFF:
                    hc["corr_coeff"] = read_tag(fid, pos).data
                    hc["corr_coeff"].flags.writeable = False
                elif kind == FIFF.FIFF_HPI_COIL_FREQ:
                    hc["coil_freq"] = float(read_tag(fid, pos).data.item())
            hcs.append(hc)
        hm["hpi_coils"] = hcs
        hms.append(hm)
    info["hpi_meas"] = hms
    del hms

    subject_info = dir_tree_find(meas_info, FIFF.FIFFB_SUBJECT)
    si = None
    if len(subject_info) == 1:
        subject_info = subject_info[0]
        si = dict()
        for k in range(subject_info["nent"]):
            kind = subject_info["directory"][k].kind
            pos = subject_info["directory"][k].pos
            if kind == FIFF.FIFF_SUBJ_ID:
                tag = read_tag(fid, pos)
                si["id"] = int(tag.data.item())
            elif kind == FIFF.FIFF_SUBJ_HIS_ID:
                tag = read_tag(fid, pos)
                si["his_id"] = str(tag.data)
            elif kind == FIFF.FIFF_SUBJ_LAST_NAME:
                tag = read_tag(fid, pos)
                si["last_name"] = str(tag.data)
            elif kind == FIFF.FIFF_SUBJ_FIRST_NAME:
                tag = read_tag(fid, pos)
                si["first_name"] = str(tag.data)
            elif kind == FIFF.FIFF_SUBJ_MIDDLE_NAME:
                tag = read_tag(fid, pos)
                si["middle_name"] = str(tag.data)
            elif kind == FIFF.FIFF_SUBJ_BIRTH_DAY:
                try:
                    tag = read_tag(fid, pos)
                except OverflowError:
                    warn(
                        "Encountered an error while trying to read the "
                        "birthday from the input data. No birthday will be "
                        "set. Please check the integrity of the birthday "
                        "information in the input data."
                    )
                    continue
                si["birthday"] = tag.data
            elif kind == FIFF.FIFF_SUBJ_SEX:
                tag = read_tag(fid, pos)
                si["sex"] = int(tag.data.item())
            elif kind == FIFF.FIFF_SUBJ_HAND:
                tag = read_tag(fid, pos)
                si["hand"] = int(tag.data.item())
            elif kind == FIFF.FIFF_SUBJ_WEIGHT:
                tag = read_tag(fid, pos)
                si["weight"] = float(tag.data.item())
            elif kind == FIFF.FIFF_SUBJ_HEIGHT:
                tag = read_tag(fid, pos)
                si["height"] = float(tag.data.item())
    info["subject_info"] = si
    del si

    device_info = dir_tree_find(meas_info, FIFF.FIFFB_DEVICE)
    di = None
    if len(device_info) == 1:
        device_info = device_info[0]
        di = dict()
        for k in range(device_info["nent"]):
            kind = device_info["directory"][k].kind
            pos = device_info["directory"][k].pos
            if kind == FIFF.FIFF_DEVICE_TYPE:
                tag = read_tag(fid, pos)
                di["type"] = str(tag.data)
            elif kind == FIFF.FIFF_DEVICE_MODEL:
                tag = read_tag(fid, pos)
                di["model"] = str(tag.data)
            elif kind == FIFF.FIFF_DEVICE_SERIAL:
                tag = read_tag(fid, pos)
                di["serial"] = str(tag.data)
            elif kind == FIFF.FIFF_DEVICE_SITE:
                tag = read_tag(fid, pos)
                di["site"] = str(tag.data)
    info["device_info"] = di
    del di

    helium_info = dir_tree_find(meas_info, FIFF.FIFFB_HELIUM)
    hi = None
    if len(helium_info) == 1:
        helium_info = helium_info[0]
        hi = dict()
        for k in range(helium_info["nent"]):
            kind = helium_info["directory"][k].kind
            pos = helium_info["directory"][k].pos
            if kind == FIFF.FIFF_HE_LEVEL_RAW:
                tag = read_tag(fid, pos)
                hi["he_level_raw"] = float(tag.data.item())
            elif kind == FIFF.FIFF_HELIUM_LEVEL:
                tag = read_tag(fid, pos)
                hi["helium_level"] = float(tag.data.item())
            elif kind == FIFF.FIFF_ORIG_FILE_GUID:
                tag = read_tag(fid, pos)
                hi["orig_file_guid"] = str(tag.data)
            elif kind == FIFF.FIFF_MEAS_DATE:
                tag = read_tag(fid, pos)
                hi["meas_date"] = _ensure_meas_date_none_or_dt(
                    tuple(int(t) for t in tag.data),
                )
        if "meas_date" not in hi:
            hi["meas_date"] = None
    info["helium_info"] = hi
    del hi

    hpi_subsystem = dir_tree_find(meas_info, FIFF.FIFFB_HPI_SUBSYSTEM)
    hs = None
    if len(hpi_subsystem) == 1:
        hpi_subsystem = hpi_subsystem[0]
        hs = dict()
        for k in range(hpi_subsystem["nent"]):
            kind = hpi_subsystem["directory"][k].kind
            pos = hpi_subsystem["directory"][k].pos
            if kind == FIFF.FIFF_HPI_NCOIL:
                tag = read_tag(fid, pos)
                hs["ncoil"] = int(tag.data.item())
            elif kind == FIFF.FIFF_EVENT_CHANNEL:
                tag = read_tag(fid, pos)
                hs["event_channel"] = str(tag.data)
            hpi_coils = dir_tree_find(hpi_subsystem, FIFF.FIFFB_HPI_COIL)
            hc = []
            for coil in hpi_coils:
                this_coil = dict()
                for j in range(coil["nent"]):
                    kind = coil["directory"][j].kind
                    pos = coil["directory"][j].pos
                    if kind == FIFF.FIFF_EVENT_BITS:
                        tag = read_tag(fid, pos)
                        this_coil["event_bits"] = np.array(tag.data)
                hc.append(this_coil)
            hs["hpi_coils"] = hc
    info["hpi_subsystem"] = hs

    #   Read processing history
    info["proc_history"] = _read_proc_history(fid, tree)

    #  Make the most appropriate selection for the measurement id
    if meas_info["parent_id"] is None:
        if meas_info["id"] is None:
            if meas["id"] is None:
                if meas["parent_id"] is None:
                    info["meas_id"] = info["file_id"]
                else:
                    info["meas_id"] = meas["parent_id"]
            else:
                info["meas_id"] = meas["id"]
        else:
            info["meas_id"] = meas_info["id"]
    else:
        info["meas_id"] = meas_info["parent_id"]
    info["experimenter"] = experimenter
    info["description"] = description
    info["proj_id"] = proj_id
    info["proj_name"] = proj_name
    if meas_date is None:
        meas_date = (info["meas_id"]["secs"], info["meas_id"]["usecs"])
    info["meas_date"] = _ensure_meas_date_none_or_dt(meas_date)
    info["utc_offset"] = utc_offset

    info["sfreq"] = sfreq
    info["highpass"] = highpass if highpass is not None else 0.0
    info["lowpass"] = lowpass if lowpass is not None else info["sfreq"] / 2.0
    info["line_freq"] = line_freq
    info["gantry_angle"] = gantry_angle

    #   Add the channel information and make a list of channel names
    #   for convenience
    info["chs"] = chs

    #
    #   Add the coordinate transformations
    #
    info["dev_head_t"] = dev_head_t
    info["ctf_head_t"] = ctf_head_t
    info["dev_ctf_t"] = dev_ctf_t
    if dev_head_t is not None and ctf_head_t is not None and dev_ctf_t is None:
        head_ctf_trans = np.linalg.inv(ctf_head_t["trans"])
        dev_ctf_trans = np.dot(head_ctf_trans, info["dev_head_t"]["trans"])
        info["dev_ctf_t"] = Transform("meg", "ctf_head", dev_ctf_trans)

    #   All kinds of auxliary stuff
    info["dig"] = _format_dig_points(dig)
    info["bads"] = bads
    info._update_redundant()
    if clean_bads:
        info["bads"] = [b for b in bads if b in info["ch_names"]]
    info["projs"] = projs
    info["comps"] = comps
    info["acq_pars"] = acq_pars
    info["acq_stim"] = acq_stim
    info["custom_ref_applied"] = custom_ref_applied
    info["xplotter_layout"] = xplotter_layout
    info["kit_system_id"] = kit_system_id
    info._check_consistency()
    info._unlocked = False
    return info, meas


def _read_extended_ch_info(chs, parent, fid):
    ch_infos = dir_tree_find(parent, FIFF.FIFFB_CH_INFO)
    if len(ch_infos) == 0:
        return
    _check_option("length of channel infos", len(ch_infos), [len(chs)])
    logger.info("    Reading extended channel information")

    # Here we assume that ``remap`` is in the same order as the channels
    # themselves, which is hopefully safe enough.
    ch_names_mapping = dict()
    for new, ch in zip(ch_infos, chs):
        for k in range(new["nent"]):
            kind = new["directory"][k].kind
            try:
                key, cast = _CH_READ_MAP[kind]
            except KeyError:
                # This shouldn't happen if we're up to date with the FIFF
                # spec
                warn(f"Discarding extra channel information kind {kind}")
                continue
            assert key in ch
            data = read_tag(fid, new["directory"][k].pos).data
            if data is not None:
                data = cast(data)
                if key == "ch_name":
                    ch_names_mapping[ch[key]] = data
                ch[key] = data
        _update_ch_info_named(ch)
    # we need to return ch_names_mapping so that we can also rename the
    # bad channels
    return ch_names_mapping


def _rename_comps(comps, ch_names_mapping):
    if not (comps and ch_names_mapping):
        return
    for comp in comps:
        data = comp["data"]
        for key in ("row_names", "col_names"):
            data[key][:] = _rename_list(data[key], ch_names_mapping)


def _ensure_meas_date_none_or_dt(meas_date):
    if meas_date is None or np.array_equal(meas_date, DATE_NONE):
        meas_date = None
    elif not isinstance(meas_date, datetime.datetime):
        meas_date = _stamp_to_dt(meas_date)
    return meas_date


def _check_dates(info, prepend_error=""):
    """Check dates before writing as fif files.

    It's needed because of the limited integer precision
    of the fix standard.
    """
    for key in ("file_id", "meas_id"):
        value = info.get(key)
        if value is not None:
            assert "msecs" not in value
            for key_2 in ("secs", "usecs"):
                if (
                    value[key_2] < np.iinfo(">i4").min
                    or value[key_2] > np.iinfo(">i4").max
                ):
                    raise RuntimeError(
                        f"{prepend_error}info[{key}][{key_2}] must be between "
                        f'"{np.iinfo(">i4").min!r}" and "{np.iinfo(">i4").max!r}", got '
                        f'"{value[key_2]!r}"'
                    )

    meas_date = info.get("meas_date")
    if meas_date is None:
        return

    meas_date_stamp = _dt_to_stamp(meas_date)
    if (
        meas_date_stamp[0] < np.iinfo(">i4").min
        or meas_date_stamp[0] > np.iinfo(">i4").max
    ):
        raise RuntimeError(
            f'{prepend_error}info["meas_date"] seconds must be between '
            f'"{(np.iinfo(">i4").min, 0)!r}" and "{(np.iinfo(">i4").max, 0)!r}", got '
            f'"{meas_date_stamp[0]!r}"'
        )


@fill_doc
def write_meas_info(fid, info, data_type=None, reset_range=True):
    """Write measurement info into a file id (from a fif file).

    Parameters
    ----------
    fid : file
        Open file descriptor.
    %(info_not_none)s
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.

    Notes
    -----
    Tags are written in a particular order for compatibility with maxfilter.
    """
    info._check_consistency()
    _check_dates(info)

    # Measurement info
    start_block(fid, FIFF.FIFFB_MEAS_INFO)

    # Add measurement id
    if info["meas_id"] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info["meas_id"])

    for event in info["events"]:
        start_block(fid, FIFF.FIFFB_EVENTS)
        if event.get("channels") is not None:
            write_int(fid, FIFF.FIFF_EVENT_CHANNELS, event["channels"])
        if event.get("list") is not None:
            write_int(fid, FIFF.FIFF_EVENT_LIST, event["list"])
        end_block(fid, FIFF.FIFFB_EVENTS)

    #   HPI Result
    for hpi_result in info["hpi_results"]:
        start_block(fid, FIFF.FIFFB_HPI_RESULT)
        write_dig_points(fid, hpi_result["dig_points"])
        if "order" in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_DIGITIZATION_ORDER, hpi_result["order"])
        if "used" in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_COILS_USED, hpi_result["used"])
        if "moments" in hpi_result:
            write_float_matrix(fid, FIFF.FIFF_HPI_COIL_MOMENTS, hpi_result["moments"])
        if "goodness" in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_GOODNESS, hpi_result["goodness"])
        if "good_limit" in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_GOOD_LIMIT, hpi_result["good_limit"])
        if "dist_limit" in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_DIST_LIMIT, hpi_result["dist_limit"])
        if "accept" in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_FIT_ACCEPT, hpi_result["accept"])
        if "coord_trans" in hpi_result:
            write_coord_trans(fid, hpi_result["coord_trans"])
        end_block(fid, FIFF.FIFFB_HPI_RESULT)

    #   HPI Measurement
    for hpi_meas in info["hpi_meas"]:
        start_block(fid, FIFF.FIFFB_HPI_MEAS)
        if hpi_meas.get("creator") is not None:
            write_string(fid, FIFF.FIFF_CREATOR, hpi_meas["creator"])
        if hpi_meas.get("sfreq") is not None:
            write_float(fid, FIFF.FIFF_SFREQ, hpi_meas["sfreq"])
        if hpi_meas.get("nchan") is not None:
            write_int(fid, FIFF.FIFF_NCHAN, hpi_meas["nchan"])
        if hpi_meas.get("nave") is not None:
            write_int(fid, FIFF.FIFF_NAVE, hpi_meas["nave"])
        if hpi_meas.get("ncoil") is not None:
            write_int(fid, FIFF.FIFF_HPI_NCOIL, hpi_meas["ncoil"])
        if hpi_meas.get("first_samp") is not None:
            write_int(fid, FIFF.FIFF_FIRST_SAMPLE, hpi_meas["first_samp"])
        if hpi_meas.get("last_samp") is not None:
            write_int(fid, FIFF.FIFF_LAST_SAMPLE, hpi_meas["last_samp"])
        for hpi_coil in hpi_meas["hpi_coils"]:
            start_block(fid, FIFF.FIFFB_HPI_COIL)
            if hpi_coil.get("number") is not None:
                write_int(fid, FIFF.FIFF_HPI_COIL_NO, hpi_coil["number"])
            if hpi_coil.get("epoch") is not None:
                write_float_matrix(fid, FIFF.FIFF_EPOCH, hpi_coil["epoch"])
            if hpi_coil.get("slopes") is not None:
                write_float(fid, FIFF.FIFF_HPI_SLOPES, hpi_coil["slopes"])
            if hpi_coil.get("corr_coeff") is not None:
                write_float(fid, FIFF.FIFF_HPI_CORR_COEFF, hpi_coil["corr_coeff"])
            if hpi_coil.get("coil_freq") is not None:
                write_float(fid, FIFF.FIFF_HPI_COIL_FREQ, hpi_coil["coil_freq"])
            end_block(fid, FIFF.FIFFB_HPI_COIL)
        end_block(fid, FIFF.FIFFB_HPI_MEAS)

    #   Polhemus data
    write_dig_points(fid, info["dig"], block=True)

    #   megacq parameters
    if info["acq_pars"] is not None or info["acq_stim"] is not None:
        start_block(fid, FIFF.FIFFB_DACQ_PARS)
        if info["acq_pars"] is not None:
            write_string(fid, FIFF.FIFF_DACQ_PARS, info["acq_pars"])

        if info["acq_stim"] is not None:
            write_string(fid, FIFF.FIFF_DACQ_STIM, info["acq_stim"])

        end_block(fid, FIFF.FIFFB_DACQ_PARS)

    #   Coordinate transformations if the HPI result block was not there
    if info["dev_head_t"] is not None:
        write_coord_trans(fid, info["dev_head_t"])

    if info["ctf_head_t"] is not None:
        write_coord_trans(fid, info["ctf_head_t"])

    if info["dev_ctf_t"] is not None:
        write_coord_trans(fid, info["dev_ctf_t"])

    #   Projectors
    ch_names_mapping = _make_ch_names_mapping(info["chs"])
    _write_proj(fid, info["projs"], ch_names_mapping=ch_names_mapping)

    #   Bad channels
    _write_bad_channels(fid, info["bads"], ch_names_mapping=ch_names_mapping)

    #   General
    if info.get("experimenter") is not None:
        write_string(fid, FIFF.FIFF_EXPERIMENTER, info["experimenter"])
    if info.get("description") is not None:
        write_string(fid, FIFF.FIFF_DESCRIPTION, info["description"])
    if info.get("proj_id") is not None:
        write_int(fid, FIFF.FIFF_PROJ_ID, info["proj_id"])
    if info.get("proj_name") is not None:
        write_string(fid, FIFF.FIFF_PROJ_NAME, info["proj_name"])
    if info.get("meas_date") is not None:
        write_int(fid, FIFF.FIFF_MEAS_DATE, _dt_to_stamp(info["meas_date"]))
    if info.get("utc_offset") is not None:
        write_string(fid, FIFF.FIFF_UTC_OFFSET, info["utc_offset"])
    write_int(fid, FIFF.FIFF_NCHAN, info["nchan"])
    write_float(fid, FIFF.FIFF_SFREQ, info["sfreq"])
    if info["lowpass"] is not None:
        write_float(fid, FIFF.FIFF_LOWPASS, info["lowpass"])
    if info["highpass"] is not None:
        write_float(fid, FIFF.FIFF_HIGHPASS, info["highpass"])
    if info.get("line_freq") is not None:
        write_float(fid, FIFF.FIFF_LINE_FREQ, info["line_freq"])
    if info.get("gantry_angle") is not None:
        write_float(fid, FIFF.FIFF_GANTRY_ANGLE, info["gantry_angle"])
    if data_type is not None:
        write_int(fid, FIFF.FIFF_DATA_PACK, data_type)
    if info.get("custom_ref_applied"):
        write_int(fid, FIFF.FIFF_MNE_CUSTOM_REF, info["custom_ref_applied"])
    if info.get("xplotter_layout"):
        write_string(fid, FIFF.FIFF_XPLOTTER_LAYOUT, info["xplotter_layout"])

    #  Channel information
    _write_ch_infos(fid, info["chs"], reset_range, ch_names_mapping)

    # Subject information
    if info.get("subject_info") is not None:
        start_block(fid, FIFF.FIFFB_SUBJECT)
        si = info["subject_info"]
        if si.get("id") is not None:
            write_int(fid, FIFF.FIFF_SUBJ_ID, si["id"])
        if si.get("his_id") is not None:
            write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, si["his_id"])
        if si.get("last_name") is not None:
            write_string(fid, FIFF.FIFF_SUBJ_LAST_NAME, si["last_name"])
        if si.get("first_name") is not None:
            write_string(fid, FIFF.FIFF_SUBJ_FIRST_NAME, si["first_name"])
        if si.get("middle_name") is not None:
            write_string(fid, FIFF.FIFF_SUBJ_MIDDLE_NAME, si["middle_name"])
        if si.get("birthday") is not None:
            write_julian(fid, FIFF.FIFF_SUBJ_BIRTH_DAY, si["birthday"])
        if si.get("sex") is not None:
            write_int(fid, FIFF.FIFF_SUBJ_SEX, si["sex"])
        if si.get("hand") is not None:
            write_int(fid, FIFF.FIFF_SUBJ_HAND, si["hand"])
        if si.get("weight") is not None:
            write_float(fid, FIFF.FIFF_SUBJ_WEIGHT, si["weight"])
        if si.get("height") is not None:
            write_float(fid, FIFF.FIFF_SUBJ_HEIGHT, si["height"])
        end_block(fid, FIFF.FIFFB_SUBJECT)
        del si

    if info.get("device_info") is not None:
        start_block(fid, FIFF.FIFFB_DEVICE)
        di = info["device_info"]
        if di.get("type") is not None:
            write_string(fid, FIFF.FIFF_DEVICE_TYPE, di["type"])
        for key in ("model", "serial", "site"):
            if di.get(key) is not None:
                write_string(fid, getattr(FIFF, "FIFF_DEVICE_" + key.upper()), di[key])
        end_block(fid, FIFF.FIFFB_DEVICE)
        del di

    if info.get("helium_info") is not None:
        start_block(fid, FIFF.FIFFB_HELIUM)
        hi = info["helium_info"]
        if hi.get("he_level_raw") is not None:
            write_float(fid, FIFF.FIFF_HE_LEVEL_RAW, hi["he_level_raw"])
        if hi.get("helium_level") is not None:
            write_float(fid, FIFF.FIFF_HELIUM_LEVEL, hi["helium_level"])
        if hi.get("orig_file_guid") is not None:
            write_string(fid, FIFF.FIFF_ORIG_FILE_GUID, hi["orig_file_guid"])
        if hi.get("meas_date", None) is not None:
            write_int(fid, FIFF.FIFF_MEAS_DATE, _dt_to_stamp(hi["meas_date"]))
        end_block(fid, FIFF.FIFFB_HELIUM)
        del hi

    if info.get("hpi_subsystem") is not None:
        hs = info["hpi_subsystem"]
        start_block(fid, FIFF.FIFFB_HPI_SUBSYSTEM)
        if hs.get("ncoil") is not None:
            write_int(fid, FIFF.FIFF_HPI_NCOIL, hs["ncoil"])
        if hs.get("event_channel") is not None:
            write_string(fid, FIFF.FIFF_EVENT_CHANNEL, hs["event_channel"])
        if hs.get("hpi_coils") is not None:
            for coil in hs["hpi_coils"]:
                start_block(fid, FIFF.FIFFB_HPI_COIL)
                if coil.get("event_bits") is not None:
                    write_int(fid, FIFF.FIFF_EVENT_BITS, coil["event_bits"])
                end_block(fid, FIFF.FIFFB_HPI_COIL)
        end_block(fid, FIFF.FIFFB_HPI_SUBSYSTEM)
        del hs

    #   CTF compensation info
    comps = info["comps"]
    if ch_names_mapping:
        comps = deepcopy(comps)
        _rename_comps(comps, ch_names_mapping)
    write_ctf_comp(fid, comps)

    #   KIT system ID
    if info.get("kit_system_id") is not None:
        write_int(fid, FIFF.FIFF_MNE_KIT_SYSTEM_ID, info["kit_system_id"])

    end_block(fid, FIFF.FIFFB_MEAS_INFO)

    #   Processing history
    _write_proc_history(fid, info)


@verbose
def write_info(
    fname, info, *, data_type=None, reset_range=True, overwrite=False, verbose=None
):
    """Write measurement info in fif file.

    Parameters
    ----------
    fname : path-like
        The name of the file. Should end by ``-info.fif``.
    %(info_not_none)s
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.
    %(overwrite)s
    %(verbose)s
    """
    with start_and_end_file(fname, overwrite=overwrite) as fid:
        start_block(fid, FIFF.FIFFB_MEAS)
        write_meas_info(fid, info, data_type, reset_range)
        end_block(fid, FIFF.FIFFB_MEAS)


@verbose
def _merge_info_values(infos, key, verbose=None):
    """Merge things together.

    Fork for {'dict', 'list', 'array', 'other'}
    and consider cases where one or all are of the same type.

    Does special things for "projs", "bads", and "meas_date".
    """
    values = [d[key] for d in infos]
    msg = (
        f"Don't know how to merge '{key}'. Make sure values are compatible, got types:"
        f"\n    {[type(v) for v in values]}"
    )

    def _flatten(lists):
        return [item for sublist in lists for item in sublist]

    def _check_isinstance(values, kind, func):
        return func([isinstance(v, kind) for v in values])

    def _where_isinstance(values, kind):
        """Get indices of instances."""
        return np.where([isinstance(v, type) for v in values])[0]

    # list
    if _check_isinstance(values, list, all):
        lists = (d[key] for d in infos)
        if key == "projs":
            return _uniquify_projs(_flatten(lists))
        elif key == "bads":
            return sorted(set(_flatten(lists)))
        else:
            return _flatten(lists)
    elif _check_isinstance(values, list, any):
        idx = _where_isinstance(values, list)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            lists = (d[key] for d in infos if isinstance(d[key], list))
            return _flatten(lists)
    # dict
    elif _check_isinstance(values, dict, all):
        is_qual = all(object_diff(values[0], v) == "" for v in values[1:])
        if is_qual:
            return values[0]
        else:
            RuntimeError(msg)
    elif _check_isinstance(values, dict, any):
        idx = _where_isinstance(values, dict)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            raise RuntimeError(msg)
    # ndarray
    elif _check_isinstance(values, np.ndarray, all) or _check_isinstance(
        values, tuple, all
    ):
        is_qual = all(np.array_equal(values[0], x) for x in values[1:])
        if is_qual:
            return values[0]
        elif key == "meas_date":
            logger.info(f"Found multiple entries for {key}. Setting value to `None`")
            return None
        else:
            raise RuntimeError(msg)
    elif _check_isinstance(values, (np.ndarray, tuple), any):
        idx = _where_isinstance(values, np.ndarray)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            raise RuntimeError(msg)
    # proj_id
    elif _check_isinstance(values, (int, type(None)), all) and key == "proj_id":
        unique_values = set(values)
        if len(unique_values) != 1:
            logger.info("Found multiple proj_ids, using the first one.")
        return list(unique_values)[0]

    elif key == "experimenter" or key == "proj_name":
        if _check_isinstance(values, (str, type(None)), all):
            unique_values = set(values)
            unique_values.discard(None)
            if len(unique_values) == 1:
                return list(unique_values)[0]
            else:
                return None
    # other
    else:
        unique_values = set(values)
        if len(unique_values) == 1:
            return list(values)[0]
        elif isinstance(list(unique_values)[0], BytesIO):
            logger.info("Found multiple StringIO instances. Setting value to `None`")
            return None
        elif isinstance(list(unique_values)[0], str):
            logger.info(f"Found multiple {key}. Setting value to `None`")
            return None
        else:
            raise RuntimeError(msg)


@verbose
def _merge_info(infos, force_update_to_first=False, verbose=None):
    """Merge multiple measurement info dictionaries.

     - Fields that are present in only one info object will be used in the
       merged info.
     - Fields that are present in multiple info objects and are the same
       will be used in the merged info.
     - Fields that are present in multiple info objects and are different
       will result in a None value in the merged info.
     - Channels will be concatenated. If multiple info objects contain
       channels with the same name, an exception is raised.

    Parameters
    ----------
    infos | list of instance of Info
        Info objects to merge into one info object.
    force_update_to_first : bool
        If True, force the fields for objects in `info` will be updated
        to match those in the first item. Use at your own risk, as this
        may overwrite important metadata.
    %(verbose)s

    Returns
    -------
    info : instance of Info
        The merged info object.
    """
    for info in infos:
        info._check_consistency()
    if force_update_to_first is True:
        infos = deepcopy(infos)
        _force_update_info(infos[0], infos[1:])
    info = Info()
    info._unlocked = True
    info["chs"] = []
    for this_info in infos:
        info["chs"].extend(this_info["chs"])
    info._update_redundant()
    duplicates = {ch for ch in info["ch_names"] if info["ch_names"].count(ch) > 1}
    if len(duplicates) > 0:
        msg = (
            "The following channels are present in more than one input "
            f"measurement info objects: {list(duplicates)}"
        )
        raise ValueError(msg)

    transforms = ["ctf_head_t", "dev_head_t", "dev_ctf_t"]
    for trans_name in transforms:
        trans = [i[trans_name] for i in infos if i[trans_name]]
        if len(trans) == 0:
            info[trans_name] = None
        elif len(trans) == 1:
            info[trans_name] = trans[0]
        elif all(
            np.all(trans[0]["trans"] == x["trans"])
            and trans[0]["from"] == x["from"]
            and trans[0]["to"] == x["to"]
            for x in trans[1:]
        ):
            info[trans_name] = trans[0]
        else:
            msg = f"Measurement infos provide mutually inconsistent {trans_name}"
            raise ValueError(msg)

    # KIT system-IDs
    kit_sys_ids = [i["kit_system_id"] for i in infos if i["kit_system_id"]]
    if len(kit_sys_ids) == 0:
        info["kit_system_id"] = None
    elif len(set(kit_sys_ids)) == 1:
        info["kit_system_id"] = kit_sys_ids[0]
    else:
        raise ValueError("Trying to merge channels from different KIT systems")

    # hpi infos and digitization data:
    fields = ["hpi_results", "hpi_meas", "dig"]
    for k in fields:
        values = [i[k] for i in infos if i[k]]
        if len(values) == 0:
            info[k] = []
        elif len(values) == 1:
            info[k] = values[0]
        elif all(object_diff(values[0], v) == "" for v in values[1:]):
            info[k] = values[0]
        else:
            msg = f"Measurement infos are inconsistent for {k}"
            raise ValueError(msg)

    # other fields
    other_fields = [
        "acq_pars",
        "acq_stim",
        "bads",
        "comps",
        "custom_ref_applied",
        "description",
        "experimenter",
        "file_id",
        "highpass",
        "utc_offset",
        "hpi_subsystem",
        "events",
        "device_info",
        "helium_info",
        "line_freq",
        "lowpass",
        "meas_id",
        "proj_id",
        "proj_name",
        "projs",
        "sfreq",
        "gantry_angle",
        "subject_info",
        "sfreq",
        "xplotter_layout",
        "proc_history",
    ]

    for k in other_fields:
        info[k] = _merge_info_values(infos, k)

    info["meas_date"] = infos[0]["meas_date"]
    info._unlocked = False

    return info


@verbose
def create_info(ch_names, sfreq, ch_types="misc", verbose=None):
    """Create a basic Info instance suitable for use with create_raw.

    Parameters
    ----------
    ch_names : list of str | int
        Channel names. If an int, a list of channel names will be created
        from ``range(ch_names)``.
    sfreq : float
        Sample rate of the data.
    ch_types : list of str | str
        Channel types, default is ``'misc'`` which is a
        :term:`non-data channel <non-data channels>`.
        Currently supported fields are 'bio', 'chpi', 'csd', 'dbs', 'dipole',
        'ecg', 'ecog', 'eeg', 'emg', 'eog', 'exci', 'eyegaze',
        'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude', 'fnirs_fd_phase',
        'fnirs_od', 'gof', 'gsr', 'hbo', 'hbr', 'ias', 'misc', 'pupil',
        'ref_meg', 'resp', 'seeg', 'stim', 'syst', 'temperature' (see also
        :term:`sensor types`).
        If str, then all channels are assumed to be of the same type.
    %(verbose)s

    Returns
    -------
    %(info_not_none)s

    Notes
    -----
    The info dictionary will be sparsely populated to enable functionality
    within the rest of the package. Advanced functionality such as source
    localization can only be obtained through substantial, proper
    modifications of the info structure (not recommended).

    Note that the MEG device-to-head transform ``info['dev_head_t']`` will
    be initialized to the identity transform.

    Proper units of measure:

    * V: eeg, eog, seeg, dbs, emg, ecg, bio, ecog, resp, fnirs_fd_ac_amplitude,
      fnirs_cw_amplitude, fnirs_od
    * T: mag, chpi, ref_meg
    * T/m: grad
    * M: hbo, hbr
    * rad: fnirs_fd_phase
    * Am: dipole
    * S: gsr
    * C: temperature
    * V/m²: csd
    * GOF: gof
    * AU: misc, stim, eyegaze, pupil
    """
    try:
        ch_names = operator.index(ch_names)  # int-like
    except TypeError:
        pass
    else:
        ch_names = list(np.arange(ch_names).astype(str))
    _validate_type(ch_names, (list, tuple), "ch_names", ("list, tuple, or int"))
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    nchan = len(ch_names)
    if isinstance(ch_types, str):
        ch_types = [ch_types] * nchan
    ch_types = np.atleast_1d(np.array(ch_types, np.str_))
    if ch_types.ndim != 1 or len(ch_types) != nchan:
        raise ValueError(
            f"ch_types and ch_names must be the same length ({len(ch_types)} != "
            f"{nchan}) for ch_types={ch_types}"
        )
    info = _empty_info(sfreq)
    ch_types_dict = get_channel_type_constants(include_defaults=True)
    for ci, (ch_name, ch_type) in enumerate(zip(ch_names, ch_types)):
        _validate_type(ch_name, "str", "each entry in ch_names")
        _validate_type(ch_type, "str", "each entry in ch_types")
        if ch_type not in ch_types_dict:
            raise KeyError(f"kind must be one of {list(ch_types_dict)}, not {ch_type}")
        this_ch_dict = ch_types_dict[ch_type]
        kind = this_ch_dict["kind"]
        # handle chpi, where kind is a *list* of FIFF constants:
        kind = kind[0] if isinstance(kind, list | tuple) else kind
        # mirror what tag.py does here
        coord_frame = _ch_coord_dict.get(kind, FIFF.FIFFV_COORD_UNKNOWN)
        coil_type = this_ch_dict.get("coil_type", FIFF.FIFFV_COIL_NONE)
        unit = this_ch_dict.get("unit", FIFF.FIFF_UNIT_NONE)
        chan_info = dict(
            loc=np.full(12, np.nan),
            unit_mul=FIFF.FIFF_UNITM_NONE,
            range=1.0,
            cal=1.0,
            kind=kind,
            coil_type=coil_type,
            unit=unit,
            coord_frame=coord_frame,
            ch_name=str(ch_name),
            scanno=ci + 1,
            logno=ci + 1,
        )
        info["chs"].append(chan_info)

    info._update_redundant()
    info._check_consistency()
    info._unlocked = False
    return info


RAW_INFO_FIELDS = (
    "acq_pars",
    "acq_stim",
    "bads",
    "ch_names",
    "chs",
    "comps",
    "ctf_head_t",
    "custom_ref_applied",
    "description",
    "dev_ctf_t",
    "dev_head_t",
    "dig",
    "experimenter",
    "events",
    "utc_offset",
    "device_info",
    "file_id",
    "highpass",
    "hpi_meas",
    "hpi_results",
    "helium_info",
    "hpi_subsystem",
    "kit_system_id",
    "line_freq",
    "lowpass",
    "meas_date",
    "meas_id",
    "nchan",
    "proj_id",
    "proj_name",
    "projs",
    "sfreq",
    "subject_info",
    "xplotter_layout",
    "proc_history",
    "gantry_angle",
)


def _empty_info(sfreq):
    """Create an empty info dictionary."""
    from ..transforms import Transform

    _none_keys = (
        "acq_pars",
        "acq_stim",
        "ctf_head_t",
        "description",
        "dev_ctf_t",
        "dig",
        "experimenter",
        "utc_offset",
        "device_info",
        "file_id",
        "highpass",
        "hpi_subsystem",
        "kit_system_id",
        "helium_info",
        "line_freq",
        "lowpass",
        "meas_date",
        "meas_id",
        "proj_id",
        "proj_name",
        "subject_info",
        "xplotter_layout",
        "gantry_angle",
    )
    _list_keys = (
        "bads",
        "chs",
        "comps",
        "events",
        "hpi_meas",
        "hpi_results",
        "projs",
        "proc_history",
    )
    info = Info()
    info._unlocked = True
    for k in _none_keys:
        info[k] = None
    for k in _list_keys:
        info[k] = list()
    info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
    info["highpass"] = 0.0
    info["sfreq"] = float(sfreq)
    info["lowpass"] = info["sfreq"] / 2.0
    info["dev_head_t"] = Transform("meg", "head")
    info._update_redundant()
    info._check_consistency()
    return info


def _force_update_info(info_base, info_target):
    """Update target info objects with values from info base.

    Note that values in info_target will be overwritten by those in info_base.
    This will overwrite all fields except for: 'chs', 'ch_names', 'nchan'.

    Parameters
    ----------
    info_base : mne.Info
        The Info object you want to use for overwriting values
        in target Info objects.
    info_target : mne.Info | list of mne.Info
        The Info object(s) you wish to overwrite using info_base. These objects
        will be modified in-place.
    """
    exclude_keys = ["chs", "ch_names", "nchan", "bads"]
    info_target = np.atleast_1d(info_target).ravel()
    all_infos = np.hstack([info_base, info_target])
    for ii in all_infos:
        if not isinstance(ii, Info):
            raise ValueError(f"Inputs must be of type Info. Found type {type(ii)}")
    for key, val in info_base.items():
        if key in exclude_keys:
            continue
        for i_targ in info_target:
            with i_targ._unlock():
                i_targ[key] = val


def _add_timedelta_to_stamp(meas_date_stamp, delta_t):
    """Add a timedelta to a meas_date tuple."""
    if meas_date_stamp is not None:
        meas_date_stamp = _dt_to_stamp(_stamp_to_dt(meas_date_stamp) + delta_t)
    return meas_date_stamp


@verbose
def anonymize_info(info, daysback=None, keep_his=False, verbose=None):
    """Anonymize measurement information in place.

    .. warning:: If ``info`` is part of an object like
                 :class:`raw.info <mne.io.Raw>`, you should directly use
                 the method :meth:`raw.anonymize() <mne.io.Raw.anonymize>`
                 to ensure that all parts of the data are anonymized and
                 stay synchronized (e.g.,
                 :class:`raw.annotations <mne.Annotations>`).

    Parameters
    ----------
    %(info_not_none)s
    %(daysback_anonymize_info)s
    %(keep_his_anonymize_info)s
    %(verbose)s

    Returns
    -------
    info : instance of Info
        The anonymized measurement information.

    Notes
    -----
    %(anonymize_info_notes)s
    """
    _validate_type(info, "info", "self")

    default_anon_dos = datetime.datetime(
        2000, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
    )
    default_str = "mne_anonymize"
    default_subject_id = 0
    default_sex = 0
    default_desc = "Anonymized using a time shift to preserve age at acquisition"

    none_meas_date = info["meas_date"] is None

    if none_meas_date:
        if daysback is not None:
            warn(
                'Input info has "meas_date" set to None. '
                "Removing all information from time/date structures, "
                "*NOT* performing any time shifts!"
            )
    else:
        # compute timeshift delta
        if daysback is None:
            delta_t = info["meas_date"] - default_anon_dos
        else:
            delta_t = datetime.timedelta(days=daysback)
        with info._unlock():
            info["meas_date"] = info["meas_date"] - delta_t

    # file_id and meas_id
    for key in ("file_id", "meas_id"):
        value = info.get(key)
        if value is not None:
            assert "msecs" not in value
            if none_meas_date or ((value["secs"], value["usecs"]) == DATE_NONE):
                # Don't try to shift backwards in time when no measurement
                # date is available or when file_id is already a place holder
                tmp = DATE_NONE
            else:
                tmp = _add_timedelta_to_stamp((value["secs"], value["usecs"]), -delta_t)
            value["secs"] = tmp[0]
            value["usecs"] = tmp[1]
            # The following copy is needed for a test CTF dataset
            # otherwise value['machid'][:] = 0 would suffice
            _tmp = value["machid"].copy()
            _tmp[:] = 0
            value["machid"] = _tmp

    # subject info
    subject_info = info.get("subject_info")
    if subject_info is not None:
        if subject_info.get("id") is not None:
            subject_info["id"] = default_subject_id
        if keep_his:
            logger.info(
                "Not fully anonymizing info - keeping his_id, sex, and hand info"
            )
        else:
            if subject_info.get("his_id") is not None:
                subject_info["his_id"] = str(default_subject_id)
            if subject_info.get("sex") is not None:
                subject_info["sex"] = default_sex
            if subject_info.get("hand") is not None:
                del subject_info["hand"]  # there's no "unknown" setting

        for key in ("last_name", "first_name", "middle_name"):
            if subject_info.get(key) is not None:
                subject_info[key] = default_str

        # anonymize the subject birthday
        if none_meas_date:
            subject_info.pop("birthday", None)
        elif subject_info.get("birthday") is not None:
            subject_info["birthday"] = subject_info["birthday"] - delta_t

        for key in ("weight", "height"):
            if subject_info.get(key) is not None:
                subject_info[key] = 0

    info["experimenter"] = default_str
    info["description"] = default_desc
    with info._unlock():
        if info["proj_id"] is not None:
            info["proj_id"] = 0
        if info["proj_name"] is not None:
            info["proj_name"] = default_str
        if info["utc_offset"] is not None:
            info["utc_offset"] = None

    proc_hist = info.get("proc_history")
    if proc_hist is not None:
        for record in proc_hist:
            record["block_id"]["machid"][:] = 0
            record["experimenter"] = default_str
            if none_meas_date:
                record["block_id"]["secs"] = DATE_NONE[0]
                record["block_id"]["usecs"] = DATE_NONE[1]
                record["date"] = DATE_NONE
            else:
                this_t0 = (record["block_id"]["secs"], record["block_id"]["usecs"])
                this_t1 = _add_timedelta_to_stamp(this_t0, -delta_t)
                record["block_id"]["secs"] = this_t1[0]
                record["block_id"]["usecs"] = this_t1[1]
                record["date"] = _add_timedelta_to_stamp(record["date"], -delta_t)

    hi = info.get("helium_info")
    if hi is not None:
        if hi.get("orig_file_guid") is not None:
            hi["orig_file_guid"] = default_str
        if none_meas_date and hi.get("meas_date") is not None:
            hi["meas_date"] = _ensure_meas_date_none_or_dt(DATE_NONE)
        elif hi.get("meas_date") is not None:
            hi["meas_date"] = hi["meas_date"] - delta_t

    di = info.get("device_info")
    if di is not None:
        for k in ("serial", "site"):
            if di.get(k) is not None:
                di[k] = default_str

    err_mesg = (
        "anonymize_info generated an inconsistent info object. Underlying Error:\n"
    )
    info._check_consistency(prepend_error=err_mesg)
    err_mesg = (
        "anonymize_info generated an inconsistent info object. "
        "daysback parameter was too large. "
        "Underlying Error:\n"
    )
    _check_dates(info, prepend_error=err_mesg)

    return info


@fill_doc
def _bad_chans_comp(info, ch_names):
    """Check if channel names are consistent with current compensation status.

    Parameters
    ----------
    %(info_not_none)s

    ch_names : list of str
        The channel names to check.

    Returns
    -------
    status : bool
        True if compensation is *currently* in use but some compensation
            channels are not included in picks

        False if compensation is *currently* not being used
            or if compensation is being used and all compensation channels
            in info and included in picks.

    missing_ch_names: array-like of str, shape (n_missing,)
        The names of compensation channels not included in picks.
        Returns [] if no channels are missing.

    """
    if "comps" not in info:
        # should this be thought of as a bug?
        return False, []

    # only include compensation channels that would affect selected channels
    ch_names_s = set(ch_names)
    comp_names = []
    for comp in info["comps"]:
        if len(ch_names_s.intersection(comp["data"]["row_names"])) > 0:
            comp_names.extend(comp["data"]["col_names"])
    comp_names = sorted(set(comp_names))

    missing_ch_names = sorted(set(comp_names).difference(ch_names))

    if get_current_comp(info) != 0 and len(missing_ch_names) > 0:
        return True, missing_ch_names

    return False, missing_ch_names


_DIG_CAST = dict(kind=int, ident=int, r=lambda x: x, coord_frame=int)
# key -> const, cast, write
_CH_INFO_MAP = OrderedDict(
    scanno=(FIFF.FIFF_CH_SCAN_NO, _int_item, write_int),
    logno=(FIFF.FIFF_CH_LOGICAL_NO, _int_item, write_int),
    kind=(FIFF.FIFF_CH_KIND, _int_item, write_int),
    range=(FIFF.FIFF_CH_RANGE, _float_item, write_float),
    cal=(FIFF.FIFF_CH_CAL, _float_item, write_float),
    coil_type=(FIFF.FIFF_CH_COIL_TYPE, _int_item, write_int),
    loc=(FIFF.FIFF_CH_LOC, lambda x: x, write_float),
    unit=(FIFF.FIFF_CH_UNIT, _int_item, write_int),
    unit_mul=(FIFF.FIFF_CH_UNIT_MUL, _int_item, write_int),
    ch_name=(FIFF.FIFF_CH_DACQ_NAME, str, write_string),
    coord_frame=(FIFF.FIFF_CH_COORD_FRAME, _int_item, write_int),
)
# key -> cast
_CH_CAST = OrderedDict((key, val[1]) for key, val in _CH_INFO_MAP.items())
# const -> key, cast
_CH_READ_MAP = OrderedDict((val[0], (key, val[1])) for key, val in _CH_INFO_MAP.items())


@contextlib.contextmanager
def _writing_info_hdf5(info):
    # Make info writing faster by packing chs and dig into numpy arrays
    orig_dig = info.get("dig", None)
    orig_chs = info["chs"]
    with info._unlock():
        try:
            if orig_dig is not None and len(orig_dig) > 0:
                info["dig"] = _dict_pack(info["dig"], _DIG_CAST)
            info["chs"] = _dict_pack(info["chs"], _CH_CAST)
            info["chs"]["ch_name"] = np.char.encode(
                info["chs"]["ch_name"], encoding="utf8"
            )
            yield
        finally:
            if orig_dig is not None:
                info["dig"] = orig_dig
            info["chs"] = orig_chs


def _dict_pack(obj, casts):
    # pack a list of dict into dict of array
    return {key: np.array([o[key] for o in obj]) for key in casts}


def _dict_unpack(obj, casts):
    # unpack a dict of array into a list of dict
    n = len(obj[list(casts)[0]])
    return [{key: cast(obj[key][ii]) for key, cast in casts.items()} for ii in range(n)]


def _make_ch_names_mapping(chs):
    orig_ch_names = [c["ch_name"] for c in chs]
    ch_names = orig_ch_names.copy()
    _unique_channel_names(ch_names, max_length=15, verbose="error")
    ch_names_mapping = dict()
    if orig_ch_names != ch_names:
        ch_names_mapping.update(zip(orig_ch_names, ch_names))
    return ch_names_mapping


def _write_ch_infos(fid, chs, reset_range, ch_names_mapping):
    ch_names_mapping = dict() if ch_names_mapping is None else ch_names_mapping
    for k, c in enumerate(chs):
        #   Scan numbers may have been messed up
        c = c.copy()
        c["ch_name"] = ch_names_mapping.get(c["ch_name"], c["ch_name"])
        assert len(c["ch_name"]) <= 15
        c["scanno"] = k + 1
        # for float/double, the "range" param is unnecessary
        if reset_range:
            c["range"] = 1.0
        write_ch_info(fid, c)
    # only write new-style channel information if necessary
    if len(ch_names_mapping):
        logger.info(
            "    Writing channel names to FIF truncated to 15 characters with remapping"
        )
        for ch in chs:
            start_block(fid, FIFF.FIFFB_CH_INFO)
            assert set(ch) == set(_CH_INFO_MAP)
            for key, (const, _, write) in _CH_INFO_MAP.items():
                write(fid, const, ch[key])
            end_block(fid, FIFF.FIFFB_CH_INFO)


def _ensure_infos_match(info1, info2, name, *, on_mismatch="raise"):
    """Check if infos match.

    Parameters
    ----------
    info1, info2 : instance of Info
        The infos to compare.
    name : str
        The name of the object appearing in the error message of the comparison
        fails.
    on_mismatch : 'raise' | 'warn' | 'ignore'
        What to do in case of a mismatch of ``dev_head_t`` between ``info1``
        and ``info2``.
    """
    _check_on_missing(on_missing=on_mismatch, name="on_mismatch")

    info1._check_consistency()
    info2._check_consistency()

    if info1["nchan"] != info2["nchan"]:
        raise ValueError(f"{name}.info['nchan'] must match")
    if set(info1["bads"]) != set(info2["bads"]):
        raise ValueError(f"{name}.info['bads'] must match")
    if info1["sfreq"] != info2["sfreq"]:
        raise ValueError(f"{name}.info['sfreq'] must match")
    if set(info1["ch_names"]) != set(info2["ch_names"]):
        raise ValueError(f"{name}.info['ch_names'] must match")
    if info1["ch_names"] != info2["ch_names"]:
        msg = (
            f"{name}.info['ch_names']: Channel order must match. Use "
            '"mne.match_channel_orders()" to sort channels.'
        )
        raise ValueError(msg)
    if len(info2["projs"]) != len(info1["projs"]):
        raise ValueError(f"SSP projectors in {name} must be the same")
    if any(not _proj_equal(p1, p2) for p1, p2 in zip(info2["projs"], info1["projs"])):
        raise ValueError(f"SSP projectors in {name} must be the same")
    if (info1["dev_head_t"] is None) ^ (info2["dev_head_t"] is None) or (
        info1["dev_head_t"] is not None
        and not np.allclose(
            info1["dev_head_t"]["trans"],
            info2["dev_head_t"]["trans"],
            rtol=1e-6,
            equal_nan=True,
        )
    ):
        msg = (
            f"{name}.info['dev_head_t'] differs. The "
            f"instances probably come from different runs, and "
            f"are therefore associated with different head "
            f"positions. Manually change info['dev_head_t'] to "
            f"avoid this message but beware that this means the "
            f"MEG sensors will not be properly spatially aligned. "
            f"See mne.preprocessing.maxwell_filter to realign the "
            f"runs to a common head position."
        )
        _on_missing(on_missing=on_mismatch, msg=msg, name="on_mismatch")


def _get_fnirs_ch_pos(info):
    """Return positions of each fNIRS optode.

    fNIRS uses two types of optodes, sources and detectors.
    There can be multiple connections between each source
    and detector at different wavelengths. This function
    returns the location of each source and detector.
    """
    from ..preprocessing.nirs import _fnirs_optode_names, _optode_position

    srcs, dets = _fnirs_optode_names(info)
    ch_pos = {}
    for optode in [*srcs, *dets]:
        ch_pos[optode] = _optode_position(info, optode)
    return ch_pos


def _camel_to_snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
