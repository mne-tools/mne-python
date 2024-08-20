# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations

import importlib
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ...annotations import Annotations
from ...utils import (
    _check_fname,
    _validate_type,
    check_version,
    copy_doc,
    fill_doc,
    logger,
    verbose,
    warn,
)
from ..base import BaseRaw

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

units = {"uv": 1e-6}


@fill_doc
class RawANT(BaseRaw):
    r"""Reader for Raw ANT files in .cnt format.

    Parameters
    ----------
    fname : file-like
        Path to the ANT raw file to load. The file should have the extension ``.cnt``.
    eog : str | None
        Regex pattern to find EOG channel labels. If None, no EOG channels are
        automatically detected.
    misc : str | None
        Regex pattern to find miscellaneous channels. If None, no miscellaneous channels
        are automatically detected. The default pattern ``"BIP\d+"`` will mark all
        bipolar channels as ``misc``.

        .. note::

            A bipolar channel might actually contain ECG, EOG or other signal types
            which might have a dedicated channel type in MNE-Python. In this case, use
            :meth:`mne.io.Raw.set_channel_types` to change the channel type of the
            channel.
    bipolars : list of str | tuple of str | None
        The list of channels to treat as bipolar EEG channels. Each element should be
        a string of the form ``'anode-cathode'`` or in ANT terminology as ``'label-
        reference'``. If None, all channels are interpreted as ``'eeg'`` channels
        referenced to the same reference electrode. Bipolar channels are treated
        as EEG channels with a special coil type in MNE-Python, see also
        :func:`mne.set_bipolar_reference`

        .. warning::

            Do not provide auxiliary channels in this argument, provide them in the
            ``eog`` and ``misc`` arguments.
    impedance_annotation : str
        The string to use for impedance annotations. Defaults to ``"impedance"``,
        however, the impedance measurement might mark the end of a segment and the
        beginning of a new segment, in which case a discontinuity similar to what
        :func:`mne.concatenate_raws` produces is present. In this case, it's better to
        include a ``BAD_xxx`` annotation to mark the discontinuity.

        .. note::

            Note that the impedance annotation will likely have a duration of ``0``.
            If the measurement marks a discontinuity, the duration should be modified to
            cover the discontinuity in its entirety.
    %(verbose)s
    """

    _extra_attributes = ("impedances",)

    @verbose
    def __init__(
        self,
        fname: str | Path,
        eog: str | None,
        misc: str | None,
        bipolars: list[str] | tuple[str, ...] | None,
        impedance_annotation: str,
        *,
        verbose=None,
    ) -> None:
        logger.info("Reading ANT file %s", fname)
        if importlib.util.find_spec("antio") is None:
            raise ImportError(
                "Missing optional dependency 'antio'. Use pip or conda to install "
                "'antio'."
            )
        check_version("antio", "0.2.0")

        from antio import read_cnt
        from antio.parser import read_data, read_info, read_triggers

        fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")
        _validate_type(eog, (str, None), "eog")
        _validate_type(misc, (str, None), "misc")
        _validate_type(bipolars, (list, tuple, None), "bipolar")
        _validate_type(impedance_annotation, (str,), "impedance_annotation")
        if len(impedance_annotation) == 0:
            raise ValueError("The impedance annotation cannot be an empty string.")
        cnt = read_cnt(str(fname))
        # parse channels, sampling frequency, and create info
        ch_info = read_info(cnt)  # load in 2 lines for compat with antio 0.2 and 0.3
        ch_names, ch_units, ch_refs = ch_info[0], ch_info[1], ch_info[2]
        ch_types = _parse_ch_types(ch_names, eog, misc, ch_refs)
        if bipolars is not None:  # handle bipolar channels
            bipolars_idx = _handle_bipolar_channels(ch_names, ch_refs, bipolars)
            for idx, ch in zip(bipolars_idx, bipolars):
                if ch_types[idx] != "eeg":
                    warn(
                        f"Channel {ch} was not parsed as an EEG channel, changing to "
                        "EEG channel type since bipolar EEG was requested."
                    )
                ch_names[idx] = ch
                ch_types[idx] = "eeg"
        info = create_info(
            ch_names, sfreq=cnt.get_sample_frequency(), ch_types=ch_types
        )
        if bipolars is not None:
            with info._unlock():
                for idx in bipolars_idx:
                    info["chs"][idx]["coil_type"] = FIFF.FIFFV_COIL_EEG_BIPOLAR
        # read and scale data array
        data = read_data(cnt)
        _scale_data(data, ch_units)
        super().__init__(info, preload=data, filenames=[fname], verbose=verbose)
        # look for annotations (called trigger by ant)
        onsets, durations, descriptions, impedances, disconnect = read_triggers(cnt)
        onsets, durations, descriptions = _prepare_annotations(
            onsets, durations, descriptions, disconnect, impedance_annotation
        )
        onsets = np.array(onsets) / self.info["sfreq"]
        durations = np.array(durations) / self.info["sfreq"]
        annotations = Annotations(onsets, duration=durations, description=descriptions)
        self.set_annotations(annotations)
        # set impedance similarly as for brainvision files
        self._impedances = [
            {ch: imp[k] for k, ch in enumerate(ch_names)} for imp in impedances
        ]

    @property
    def impedances(self) -> list[dict[str, float]]:
        """List of impedance measurements."""
        return self._impedances


def _handle_bipolar_channels(
    ch_names: list[str], ch_refs: list[str], bipolars: list[str] | tuple[str, ...]
) -> list[int]:
    """Handle bipolar channels."""
    bipolars_idx = []
    for ch in bipolars:
        _validate_type(ch, (str,), "bipolar_channel")
        if "-" not in ch:
            raise ValueError(
                "Bipolar channels should be provided as 'anode-cathode' or "
                f"'label-reference'. '{ch}' is not valid."
            )
        anode, cathode = ch.split("-")
        if anode not in ch_names:
            raise ValueError(f"Anode channel {anode} not found in the channels.")
        idx = ch_names.index(anode)
        if cathode != ch_refs[idx]:
            raise ValueError(
                f"Reference electrode for {anode} is {ch_refs[idx]}, not {cathode}."
            )
        # store idx for later FIFF coil type change
        bipolars_idx.append(idx)
    return bipolars_idx


def _parse_ch_types(
    ch_names: list[str], eog: str | None, misc: str | None, ch_refs: list[str]
) -> list[str]:
    """Parse the channel types."""
    eog = re.compile(eog) if eog is not None else None
    misc = re.compile(misc) if misc is not None else None
    ch_types = []
    for ch in ch_names:
        if eog is not None and re.fullmatch(eog, ch):
            ch_types.append("eog")
        elif misc is not None and re.fullmatch(misc, ch):
            ch_types.append("misc")
        else:
            ch_types.append("eeg")
    eeg_refs = [ch_refs[k] for k, elt in enumerate(ch_types) if elt == "eeg"]
    if len(set(eeg_refs)) == 1:
        logger.info(
            "All %i EEG channels are referenced to %s.", len(eeg_refs), eeg_refs[0]
        )
    else:
        warn("All EEG channels are not referenced to the same electrode.")
    return ch_types


def _prepare_annotations(
    onsets: list[int],
    durations: list[int],
    descriptions: list[str],
    disconnect: dict[str, list[int]],
    impedance_annotation: str,
) -> tuple[list[int], list[int], list[str]]:
    """Parse the ANT triggers into better Annotations."""
    # first, let's replace the description 'impedance' with impedance_annotation
    for k, desc in enumerate(descriptions):
        if desc.lower() == "impedance":
            descriptions[k] = impedance_annotation
    # next, let's look for amplifier connection/disconnection and let's try to create
    # BAD_disconnection annotations from them.
    if (
        len(disconnect["start"]) == len(disconnect["stop"])
        and len(disconnect["start"]) != 0
        and all(
            0 <= stop - start
            for start, stop in zip(disconnect["start"], disconnect["stop"])
        )
    ):
        for start, stop in zip(disconnect["start"], disconnect["stop"]):
            onsets.append(start)
            durations.append(stop - start)
            descriptions.append("BAD_disconnection")
    else:
        for elt in disconnect["start"]:
            onsets.append(elt)
            durations.append(0)
            descriptions.append("Amplifier disconnected")
        for elt in disconnect["stop"]:
            onsets.append(elt)
            durations.append(0)
            descriptions.append("Amplifier reconnected")
    return onsets, durations, descriptions


def _scale_data(data: NDArray[np.float64], ch_units: list[str]) -> None:
    """Scale the data array based on the human-readable units reported by ANT.

    Operates in-place.
    """
    units_index = defaultdict(list)
    for idx, unit in enumerate(ch_units):
        units_index[unit].append(idx)
    for unit, value in units_index.items():
        if unit in units:
            data[np.array(value, dtype=np.int16), :] *= units[unit]
        else:
            warn(
                f"Unit {unit} not recognized, not scaling. Please report the unit on "
                "a github issue on https://github.com/mne-tools/mne-python."
            )


@copy_doc(RawANT)
def read_raw_ant(
    fname,
    eog=None,
    misc=r"BIP\d+",
    bipolars=None,
    impedance_annotation="impedance",
    *,
    verbose=None,
) -> RawANT:
    """
    Returns
    -------
    raw : instance of RawANT
        A Raw object containing ANT data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.
        The impedance measurements are stored in the extra attribute ``raw.impedances``.
    """
    return RawANT(
        fname,
        eog=eog,
        misc=misc,
        bipolars=bipolars,
        impedance_annotation=impedance_annotation,
        verbose=verbose,
    )
