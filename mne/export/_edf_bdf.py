# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime as dt
from collections.abc import Callable

import numpy as np

from mne.annotations import _sync_onset
from mne.utils import _check_edfio_installed, warn

_check_edfio_installed()
from edfio import (  # noqa: E402
    Bdf,
    BdfSignal,
    Edf,
    EdfAnnotation,
    EdfSignal,
    Patient,
    Recording,
)


# copied from edfio (Apache license)
def _round_float_to_8_characters(
    value: float,
    round_func: Callable[[float], int],
) -> float:
    if isinstance(value, int) or value.is_integer():
        return value
    length = 8
    integer_part_length = str(value).find(".")
    if integer_part_length == length:
        return round_func(value)
    factor = 10 ** (length - 1 - integer_part_length)
    return round_func(value * factor) / factor


def _export_raw_edf_bdf(fname, raw, physical_range, add_ch_type, file_format):
    """Export Raw objects to EDF/BDF files.

    Parameters
    ----------
    fname : str
        Output file name.
    raw : instance of Raw
        The raw instance to export.
    physical_range : str or tuple
        Physical range setting.
    add_ch_type : bool
        Whether to add channel type to signal label.
    file_format : str
        File format ("EDF" or "BDF").

    Notes
    -----
    TODO: if in future the Info object supports transducer or technician information,
    allow writing those here.
    """
    units = dict(
        eeg="uV", ecog="uV", seeg="uV", eog="uV", ecg="uV", emg="uV", bio="uV", dbs="uV"
    )

    if file_format == "EDF":
        digital_min, digital_max = -32767, 32767  # 16-bit
        signal_class = EdfSignal
        writer_class = Edf
    else:  # BDF
        digital_min, digital_max = -8388607, 8388607  # 24-bit
        signal_class = BdfSignal
        writer_class = Bdf

    ch_types = np.array(raw.get_channel_types())

    # load and prepare data
    raw.load_data()
    data = raw.get_data(units=units)
    sfreq = raw.info["sfreq"]
    pad_annotations = []

    # Sampling frequency in EDF/BDF only supports integers, so to allow for float
    # sampling rates from Raw, we adjust the output sampling rate for all channels and
    # the data record duration.
    if float(sfreq).is_integer():
        out_sfreq = int(sfreq)
        data_record_duration = None
        # make non-integer second durations work
        if (pad_width := int(np.ceil(raw.n_times / sfreq) * sfreq - raw.n_times)) > 0:
            warn(
                f"{file_format} format requires equal-length data blocks, so "
                f"{pad_width / sfreq:.3g} seconds of edge values were appended to all "
                "channels when writing the final block."
            )
            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, int(pad_width)),
                ),
                "edge",
            )

            pad_annotations.append(
                EdfAnnotation(
                    raw.times[-1] + 1 / sfreq, pad_width / sfreq, "BAD_ACQ_SKIP"
                )
            )
    else:
        data_record_duration = _round_float_to_8_characters(
            np.floor(sfreq) / sfreq, round
        )
        out_sfreq = np.floor(sfreq) / data_record_duration
        warn(
            f"Data has a non-integer sampling rate of {sfreq}; writing to "
            f"{file_format} format may cause a small change to sample times."
        )

    # extract filter information
    lowpass = raw.info["lowpass"]
    highpass = raw.info["highpass"]
    linefreq = raw.info["line_freq"]
    filter_str_info = f"HP:{highpass}Hz LP:{lowpass}Hz"
    if linefreq is not None:
        filter_str_info += f" N:{linefreq}Hz"

    # compute physical range
    if physical_range == "auto":
        # get max and min for each channel type data
        ch_types_phys_max = dict()
        ch_types_phys_min = dict()

        for _type in np.unique(ch_types):
            _picks = [n for n, t in zip(raw.ch_names, ch_types) if t == _type]
            _data = raw.get_data(units=units, picks=_picks)
            ch_types_phys_max[_type] = _data.max()
            ch_types_phys_min[_type] = _data.min()
    elif physical_range == "channelwise":
        prange = None
    else:
        # get the physical min and max of the data in uV
        # Physical ranges of the data in uV are usually set by the manufacturer and
        # electrode properties. In general, physical min and max should be the clipping
        # levels of the ADC input, and they should be the same for all channels. For
        # example, Nihon Kohden uses ±3200 uV for all EEG channels (corresponding to the
        # actual clipping levels of their input amplifiers & ADC). For a discussion,
        # see https://github.com/sccn/eeglab/issues/246
        pmin, pmax = physical_range[0], physical_range[1]

        # check that physical min and max is not exceeded
        if data.max() > pmax:
            warn(
                f"The maximum μV of the data {data.max()} is more than the physical max"
                f" passed in {pmax}."
            )
        if data.min() < pmin:
            warn(
                f"The minimum μV of the data {data.min()} is less than the physical min"
                f" passed in {pmin}."
            )
        data = np.clip(data, pmin, pmax)
        prange = pmin, pmax

    # create signals
    signals = []
    for idx, ch in enumerate(raw.ch_names):
        ch_type = ch_types[idx]
        signal_label = f"{ch_type.upper()} {ch}" if add_ch_type else ch
        if len(signal_label) > 16:
            raise RuntimeError(
                f"Signal label for {ch} ({ch_type}) is longer than 16 characters, which"
                f" is not supported by the {file_format} standard. Please shorten the "
                f"channel name before exporting to {file_format}."
            )

        if physical_range == "auto":  # per channel type
            pmin = ch_types_phys_min[ch_type]
            pmax = ch_types_phys_max[ch_type]
            if pmax == pmin:
                pmax = pmin + 1
            prange = pmin, pmax

        signals.append(
            signal_class(
                data[idx],
                out_sfreq,
                label=signal_label,
                transducer_type="",
                physical_dimension="" if ch_type == "stim" else "uV",
                physical_range=prange,
                digital_range=(digital_min, digital_max),
                prefiltering=filter_str_info,
            )
        )

    # create patient info
    subj_info = raw.info.get("subject_info")
    if subj_info is not None:
        # get the full name of subject if available
        first_name = subj_info.get("first_name", "")
        middle_name = subj_info.get("middle_name", "")
        last_name = subj_info.get("last_name", "")
        name = "_".join(filter(None, [first_name, middle_name, last_name]))

        birthday = subj_info.get("birthday")
        hand = subj_info.get("hand")
        weight = subj_info.get("weight")
        height = subj_info.get("height")
        sex = subj_info.get("sex")

        additional_patient_info = []
        for key, value in [("height", height), ("weight", weight), ("hand", hand)]:
            if value:
                additional_patient_info.append(f"{key}={value}")

        patient = Patient(
            code=subj_info.get("his_id") or "X",
            sex={0: "X", 1: "M", 2: "F", None: "X"}[sex],
            birthdate=birthday,
            name=name or "X",
            additional=additional_patient_info,
        )
    else:
        patient = None

    # create recording info
    if (meas_date := raw.info["meas_date"]) is not None:
        startdate = dt.date(meas_date.year, meas_date.month, meas_date.day)
        starttime = dt.time(
            meas_date.hour, meas_date.minute, meas_date.second, meas_date.microsecond
        )
    else:
        startdate = None
        starttime = None

    device_info = raw.info.get("device_info")
    if device_info is not None:
        device_type = device_info.get("type") or "X"
        recording = Recording(startdate=startdate, equipment_code=device_type)
    else:
        recording = Recording(startdate=startdate)

    # create annotations
    annotations = []
    for desc, onset, duration, ch_names in zip(
        raw.annotations.description,
        # subtract raw.first_time because EDF/BDF marks events starting from the first
        # available data point and ignores raw.first_time
        _sync_onset(raw, raw.annotations.onset, inverse=False),
        raw.annotations.duration,
        raw.annotations.ch_names,
    ):
        if ch_names:
            for ch_name in ch_names:
                annotations.append(
                    EdfAnnotation(onset, duration, desc + f"@@{ch_name}")
                )
        else:
            annotations.append(EdfAnnotation(onset, duration, desc))

    annotations.extend(pad_annotations)

    # write to file
    writer_class(
        signals=signals,
        patient=patient,
        recording=recording,
        starttime=starttime,
        data_record_duration=data_record_duration,
        annotations=annotations,
    ).write(fname)


def _export_raw_edf(fname, raw, physical_range, add_ch_type):
    """Export Raw object to EDF."""
    _export_raw_edf_bdf(fname, raw, physical_range, add_ch_type, file_format="EDF")


def _export_raw_bdf(fname, raw, physical_range, add_ch_type):
    """Export Raw object to BDF."""
    _export_raw_edf_bdf(fname, raw, physical_range, add_ch_type, file_format="BDF")
