# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
from datetime import datetime, timezone

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ..._fiff.utils import _file_size, _read_segments_file
from ...annotations import Annotations
from ...utils import _check_fname, fill_doc, logger, warn
from ..base import BaseRaw, _get_scaling

CH_TYPE_MAPPING = {
    "CC": "SEEG",
}


# See https://blackrockneurotech.com/wp-content/uploads/LB-0023-7.00_NEV_File_Format.pdf
DATA_BYTE_SIZE = 2
ORIG_FORMAT = "short"


nsx_header_dict = {
    "basic": [
        ("file_id", "S8"),  # achFileType
        # file specification split into major and minor version number
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        # bytes of basic & extended header
        ("bytes_in_headers", "uint32"),
        # label of the sampling group (e.g., "1 kS/s" or "LFP low")
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        # time origin: 2byte uint16 values for ...
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        # number of channel_count match number of extended headers
        ("channel_count", "uint32"),
    ],
    "extended": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        # used front-end amplifier bank (e.g., A, B, C, D)
        ("physical_connector", "uint8"),
        # used connector pin (e.g., 1-37 on bank A, B, C or D)
        ("connector_pin", "uint8"),
        # digital and analog value ranges of the signal
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        # units of the analog range values ("mV" or "uV")
        ("units", "S16"),
        # filter settings used to create nsx from source signal
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),  # 0=None, 1=Butterworth
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],  # 0=None, 1=Butterworth,
    "data>2.1<3": [
        ("header", "uint8"),
        ("timestamp", "uint32"),
        ("nb_data_points", "uint32"),
    ],
    "data>=3": [
        ("header", "uint8"),
        ("timestamp", "uint64"),
        ("nb_data_points", "uint32"),
    ],
}


@fill_doc
def read_raw_nsx(
    input_fname, stim_channel=True, eog=None, misc=None, preload=False, *, verbose=None
) -> "RawNSX":
    """Reader function for NSx (Blackrock Microsystems) files.

    Parameters
    ----------
    input_fname : str
        Path to the NSx file.
    stim_channel : ``'auto'`` | str | list of str | int | list of int
        Defaults to ``'auto'``, which means that channels named ``'status'`` or
        ``'trigger'`` (case insensitive) are set to STIM. If str (or list of
        str), all channels matching the name(s) are set to STIM. If int (or
        list of ints), channels corresponding to the indices are set to STIM.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawEDF
        The raw instance.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    Notes
    -----
    NSx files with id (= NEURALSG), i.e., version 2.1 is currently not
    supported.

    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """
    input_fname = _check_fname(
        input_fname, overwrite="read", must_exist=True, name="input_fname"
    )
    if not input_fname.suffix.lower().startswith(".ns"):
        raise NotImplementedError(
            f"Only NSx files are supported, got {input_fname.suffix}."
        )
    return RawNSX(
        input_fname, stim_channel, eog, misc, preload=preload, verbose=verbose
    )


@fill_doc
class RawNSX(BaseRaw):
    """Raw object from NSx file from Blackrock Microsystems.

    Parameters
    ----------
    input_fname : str
        Path to the NSx file.
    stim_channel : ``'auto'`` | str | list of str | int | list of int
        Defaults to ``'auto'``, which means that channels named ``'status'`` or
        ``'trigger'`` (case insensitive) are set to STIM. If str (or list of
        str), all channels matching the name(s) are set to STIM. If int (or
        list of ints), channels corresponding to the indices are set to STIM.
    eog : list or tuple
        Names of channels or list of indices that should be designated EOG
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated MISC
        channels. Values should correspond to the electrodes in the file.
        Default is None.
    %(preload)s
    %(verbose)s

    Notes
    -----
    NSx files with id (= NEURALSG), i.e., version 2.1 is currently not
    supported.

    If channels named 'status' or 'trigger' are present, they are considered as
    STIM channels by default. Use func:`mne.find_events` to parse events
    encoded in such analog stim channels.
    """

    def __init__(
        self,
        input_fname,
        stim_channel="auto",
        eog=None,
        misc=None,
        preload=False,
        verbose=None,
    ):
        logger.info(f"Extracting NSX parameters from {input_fname}...")
        input_fname = os.path.abspath(input_fname)
        (
            info,
            data_fname,
            fmt,
            n_samples,
            orig_format,
            raw_extras,
            orig_units,
        ) = _get_hdr_info(input_fname, stim_channel=stim_channel, eog=eog, misc=misc)
        raw_extras["orig_format"] = orig_format
        first_samps = (raw_extras["timestamp"][0],)
        super().__init__(
            info,
            first_samps=first_samps,
            last_samps=[first_samps[0] + n_samples - 1],
            filenames=[data_fname],
            orig_format=orig_format,
            preload=preload,
            verbose=verbose,
            raw_extras=[raw_extras],
            orig_units=orig_units,
        )

        # Add annotations for in-data skips
        if len(self._raw_extras[0]["timestamp"]) > 1:
            starts = (
                self._raw_extras[0]["timestamp"] + self._raw_extras[0]["nb_data_points"]
            )[:-1] + 1
            stops = self._raw_extras[0]["timestamp"][1:] - 1
            durations = (stops - starts + 1) / self.info["sfreq"]
            annot = Annotations(
                onset=(starts / self.info["sfreq"]),
                duration=durations,
                description="BAD_ACQ_SKIP",
                orig_time=self.info["meas_date"],
            )
            self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        dtype = self._raw_extras[fi]["orig_format"]
        first_samps = self._raw_extras[fi]["timestamp"]
        recording_extents = self._raw_extras[fi]["nb_data_points"]
        offsets = self._raw_extras[fi]["offset_to_data_block"]
        for first_samp, recording_extent, offset in zip(
            first_samps, recording_extents, offsets
        ):
            if start > first_samp + recording_extent or stop < first_samp:
                # There is nothing to read in this chunk
                continue
            i_start = max(start, first_samp)
            i_stop = min(stop, first_samp + recording_extent)
            _read_segments_file(
                self,
                data[:, i_start - start : i_stop - start],
                idx,
                fi,
                i_start - first_samp,
                i_stop - first_samp,
                cals,
                mult,
                dtype,
                n_channels=None,
                offset=offset,
                trigger_ch=None,
            )


def _read_header(fname):
    nsx_file_id = np.fromfile(fname, count=1, dtype=[("file_id", "S8")])[0][
        "file_id"
    ].decode()

    if nsx_file_id in ["NEURALCD", "BRSMPGRP"]:
        basic_header = _read_header_22_and_above(fname)
    elif nsx_file_id == "NEURALSG":
        raise NotImplementedError(
            "NSx file id (= NEURALSG), i.e., file"
            " version 2.1 is currently not supported."
        )
    else:
        raise ValueError(
            f"NSx file id (={nsx_file_id}) does not match"
            " with supported file ids:"
            " ('NEURALCD', 'BRSMPGRP')"
        )

    time_origin = datetime(
        *[
            basic_header.pop(xx)
            for xx in (
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "millisecond",
            )
        ],
        tzinfo=timezone.utc,
    )
    basic_header["meas_date"] = time_origin
    return basic_header


def _read_header_22_and_above(fname):
    basic_header = {}
    dtype0 = nsx_header_dict["basic"]
    dtype1 = nsx_header_dict["extended"]

    nsx_file_header = np.fromfile(fname, count=1, dtype=dtype0)[0]
    basic_header.update(
        {name: nsx_file_header[name] for name in nsx_file_header.dtype.names}
    )

    offset_dtype0 = np.dtype(dtype0).itemsize
    shape = nsx_file_header["channel_count"]
    basic_header["extended"] = np.memmap(
        fname, shape=shape, offset=offset_dtype0, dtype=dtype1, mode="r"
    )

    # The following values are stored in mHz
    # See:
    # https://blackrockneurotech.com/wp-content/uploads/LB-0023-7.00_NEV_File_Format.pdf
    basic_header["highpass"] = basic_header["extended"]["hi_freq_corner"]
    basic_header["lowpass"] = basic_header["extended"]["lo_freq_corner"]
    for x in ["highpass", "lowpass"]:
        basic_header[x] = basic_header[x] * 1e-3

    ver_major, ver_minor = basic_header.pop("ver_major"), basic_header.pop("ver_minor")
    basic_header["spec"] = f"{ver_major}.{ver_minor}"

    data_header = list()
    index = 0
    offset = basic_header["bytes_in_headers"]
    filesize = _file_size(fname)
    if float(basic_header["spec"]) < 3.0:
        dtype2 = nsx_header_dict["data>2.1<3"]
    else:
        dtype2 = nsx_header_dict["data>=3"]
    while offset < filesize:
        dh = np.memmap(fname, dtype=dtype2, shape=1, offset=offset, mode="r")[0]
        data_header.append(
            {
                "header": dh["header"],
                "timestamp": dh["timestamp"],
                "nb_data_points": dh["nb_data_points"],
                "offset_to_data_block": offset + dh.dtype.itemsize,
            }
        )
        # data size = number of data points * (data_bytes * number of channels)
        # use of `int` avoids overflow problem
        data_size = (
            int(dh["nb_data_points"])
            * int(basic_header["channel_count"])
            * DATA_BYTE_SIZE
        )
        # define new offset (to possible next data block)
        offset = data_header[index]["offset_to_data_block"] + data_size
        index += 1

    basic_header["data_header"] = data_header
    return basic_header


def _get_hdr_info(fname, stim_channel=True, eog=None, misc=None):
    """Read header information NSx file."""
    eog = eog if eog is not None else []
    misc = misc if misc is not None else []

    nsx_info = _read_header(fname)
    ch_names = list(nsx_info["extended"]["electrode_label"])
    ch_types = list(nsx_info["extended"]["type"])
    ch_units = list(nsx_info["extended"]["units"])
    ch_names, ch_types, ch_units = (
        list(map(bytes.decode, xx)) for xx in (ch_names, ch_types, ch_units)
    )
    max_analog_val = nsx_info["extended"]["max_analog_val"].astype("double")
    min_analog_val = nsx_info["extended"]["min_analog_val"].astype("double")
    max_digital_val = nsx_info["extended"]["max_digital_val"].astype("double")
    min_digital_val = nsx_info["extended"]["min_digital_val"].astype("double")
    cals = (max_analog_val - min_analog_val) / (max_digital_val - min_digital_val)

    stim_channel_idxs, _ = _check_stim_channel(stim_channel, ch_names)

    nchan = int(nsx_info["channel_count"])
    logger.info("Setting channel info structure...")
    chs = list()
    pick_mask = np.ones(len(ch_names))

    orig_units = {}
    for idx, ch_name in enumerate(ch_names):
        chan_info = {}
        chan_info["logno"] = int(nsx_info["extended"]["electrode_id"][idx])
        chan_info["scanno"] = int(nsx_info["extended"]["electrode_id"][idx])
        chan_info["ch_name"] = ch_name
        chan_info["unit_mul"] = FIFF.FIFF_UNITM_NONE
        ch_unit = ch_units[idx]
        chan_info["unit"] = FIFF.FIFF_UNIT_V
        # chan_info["range"] = _unit_range_dict[ch_units[idx]]
        chan_info["range"] = 1 / _get_scaling("eeg", ch_units[idx])
        chan_info["cal"] = cals[idx]
        chan_info["coord_frame"] = FIFF.FIFFV_COORD_HEAD
        chan_info["coil_type"] = FIFF.FIFFV_COIL_EEG
        chan_info["kind"] = FIFF.FIFFV_SEEG_CH
        # montage can't be stored in NSx so channel locs are unknown:
        chan_info["loc"] = np.full(12, np.nan)
        orig_units[ch_name] = ch_unit

        # if the NSx info contained channel type information
        # set it now. They are always set to 'CC'.
        # If not inferable, set it to 'SEEG' with a warning.
        ch_type = ch_types[idx]
        ch_const = getattr(FIFF, f"FIFFV_{CH_TYPE_MAPPING.get(ch_type, 'SEEG')}_CH")
        chan_info["kind"] = ch_const
        # if user passes in explicit mapping for eog, misc and stim
        # channels set them here.
        if ch_name in eog or idx in eog or idx - nchan in eog:
            chan_info["coil_type"] = FIFF.FIFFV_COIL_NONE
            chan_info["kind"] = FIFF.FIFFV_EOG_CH
            pick_mask[idx] = False
        elif ch_name in misc or idx in misc or idx - nchan in misc:
            chan_info["coil_type"] = FIFF.FIFFV_COIL_NONE
            chan_info["kind"] = FIFF.FIFFV_MISC_CH
            pick_mask[idx] = False
        elif idx in stim_channel_idxs:
            chan_info["coil_type"] = FIFF.FIFFV_COIL_NONE
            chan_info["unit"] = FIFF.FIFF_UNIT_NONE
            chan_info["kind"] = FIFF.FIFFV_STIM_CH
            pick_mask[idx] = False
            chan_info["ch_name"] = ch_name
            ch_names[idx] = chan_info["ch_name"]
        chs.append(chan_info)

    sfreq = nsx_info["timestamp_resolution"] / nsx_info["period"]
    info = _empty_info(sfreq)
    info["meas_date"] = nsx_info["meas_date"]
    info["chs"] = chs
    info["ch_names"] = ch_names

    highpass = nsx_info["highpass"][:128]
    lowpass = nsx_info["lowpass"][:128]
    _decode_online_filters(info, highpass, lowpass)

    # Some keys to be consistent with FIF measurement info
    info["description"] = None

    info._unlocked = False
    info._update_redundant()

    orig_format = ORIG_FORMAT

    raw_extras = {
        key: [r[key] for r in nsx_info["data_header"]]
        for key in nsx_info["data_header"][0]
    }
    for key in raw_extras:
        raw_extras[key] = np.array(raw_extras[key], int)
    good_data_packets = raw_extras.pop("header") == 1
    if not good_data_packets.any():
        raise RuntimeError("NSx file appears to be broken")
    raw_extras = {key: raw_extras[key][good_data_packets] for key in raw_extras.keys()}
    raw_extras["timestamp"] = raw_extras["timestamp"] // nsx_info["period"]
    first_samp = raw_extras["timestamp"][0]
    last_samp = raw_extras["timestamp"][-1] + raw_extras["nb_data_points"][-1]
    n_samples = last_samp - first_samp

    return (
        info,
        fname,
        nsx_info["spec"],
        n_samples,
        orig_format,
        raw_extras,
        orig_units,
    )


def _decode_online_filters(info, highpass, lowpass):
    """Decode low/high-pass filters that are applied online."""
    if np.all(highpass == highpass[0]):
        if highpass[0] == "NaN":
            # Placeholder for future use. Highpass set in _empty_info.
            pass
        else:
            hp = float(highpass[0])
            info["highpass"] = hp
    else:
        info["highpass"] = float(np.max(highpass))
        warn(
            "Channels contain different highpass filters. Highest filter "
            "setting will be stored."
        )

    if np.all(lowpass == lowpass[0]):
        if lowpass[0] in ("NaN", "0", "0.0"):
            # Placeholder for future use. Lowpass set in _empty_info.
            pass
        else:
            info["lowpass"] = float(lowpass[0])
    else:
        info["lowpass"] = float(np.min(lowpass))
        warn(
            "Channels contain different lowpass filters. Lowest filter "
            "setting will be stored."
        )


def _check_stim_channel(stim_channel, ch_names):
    """Check that the stimulus channel exists in the current datafile."""
    DEFAULT_STIM_CH_NAMES = ["status", "trigger"]

    if stim_channel is None or stim_channel is False:
        return [], []

    if stim_channel is True:  # convenient aliases
        stim_channel = "auto"

    if isinstance(stim_channel, str):
        if stim_channel == "auto":
            if "auto" in ch_names:
                warn(
                    RuntimeWarning,
                    "Using `stim_channel='auto'` when auto"
                    " also corresponds to a channel name is ambiguous."
                    " Please use `stim_channel=['auto']`.",
                )
            else:
                valid_stim_ch_names = DEFAULT_STIM_CH_NAMES
        else:
            valid_stim_ch_names = [stim_channel.lower()]

    elif isinstance(stim_channel, int):
        valid_stim_ch_names = [ch_names[stim_channel].lower()]

    elif isinstance(stim_channel, list):
        if all([isinstance(s, str) for s in stim_channel]):
            valid_stim_ch_names = [s.lower() for s in stim_channel]
        elif all([isinstance(s, int) for s in stim_channel]):
            valid_stim_ch_names = [ch_names[s].lower() for s in stim_channel]
        else:
            raise ValueError("Invalid stim_channel")
    else:
        raise ValueError("Invalid stim_channel")

    ch_names_low = [ch.lower() for ch in ch_names]
    found = list(set(valid_stim_ch_names) & set(ch_names_low))

    stim_channel_idxs = [ch_names_low.index(f) for f in found]
    names = [ch_names[idx] for idx in stim_channel_idxs]
    return stim_channel_idxs, names
