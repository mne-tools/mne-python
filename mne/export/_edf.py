# Authors: MNE Developers
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime as dt

import numpy as np

from ..utils import _check_edfio_installed, warn

_check_edfio_installed()
from edfio import Edf, EdfAnnotation, EdfSignal, Patient, Recording  # noqa: E402
from edfio._utils import round_float_to_8_characters  # noqa: E402


def _export_raw(fname, raw, physical_range, add_ch_type):
    """Export Raw objects to EDF files.

    TODO: if in future the Info object supports transducer or
    technician information, allow writing those here.
    """
    # scale to save data in EDF
    phys_dims = "uV"

    # get EEG-related data in uV
    units = dict(
        eeg="uV", ecog="uV", seeg="uV", eog="uV", ecg="uV", emg="uV", bio="uV", dbs="uV"
    )

    digital_min = -32767
    digital_max = 32767

    # load data first
    raw.load_data()

    # remove extra STI channels
    orig_ch_types = raw.get_channel_types()
    drop_chs = []
    if "stim" in orig_ch_types:
        stim_index = np.argwhere(np.array(orig_ch_types) == "stim")
        stim_index = np.atleast_1d(stim_index.squeeze()).tolist()
        drop_chs.extend([raw.ch_names[idx] for idx in stim_index])

    # Add warning if any channel types are not voltage based.
    # Users are expected to only export data that is voltage based,
    # such as EEG, ECoG, sEEG, etc.
    # Non-voltage channels are dropped by the export function.
    # Note: we can write these other channels, such as 'misc'
    # but these are simply a "catch all" for unknown or undesired
    # channels.
    voltage_types = list(units) + ["stim", "misc"]
    non_voltage_ch = [ch not in voltage_types for ch in orig_ch_types]
    if any(non_voltage_ch):
        warn(
            f"Non-voltage channels detected: {non_voltage_ch}. MNE-Python's "
            "EDF exporter only supports voltage-based channels, because the "
            "EDF format cannot accommodate much of the accompanying data "
            "necessary for channel types like MEG and fNIRS (channel "
            "orientations, coordinate frame transforms, etc). You can "
            "override this restriction by setting those channel types to "
            '"misc" but no guarantees are made of the fidelity of that '
            "approach."
        )

    ch_names = [ch for ch in raw.ch_names if ch not in drop_chs]
    ch_types = np.array(raw.get_channel_types(picks=ch_names))
    n_times = raw.n_times

    # get the entire dataset in uV
    data = raw.get_data(units=units, picks=ch_names)

    # Sampling frequency in EDF only supports integers, so to allow for
    # float sampling rates from Raw, we adjust the output sampling rate
    # for all channels and the data record duration.
    sfreq = raw.info["sfreq"]
    if float(sfreq).is_integer():
        out_sfreq = int(sfreq)
        data_record_duration = None
        # make non-integer second durations work
        if pad_width := int(np.ceil(n_times / sfreq) * sfreq - n_times):
            warn(
                f"EDF format requires equal-length data blocks, "
                f"so {pad_width / sfreq} seconds of "
                "zeros were appended to all channels when writing the "
                "final block."
            )
            data = np.pad(data, (0, int(pad_width)))
    else:
        data_record_duration = round_float_to_8_characters(
            np.floor(sfreq) / sfreq, round
        )
        out_sfreq = np.floor(sfreq) / data_record_duration
        warn(
            f"Data has a non-integer sampling rate of {sfreq}; writing to "
            "EDF format may cause a small change to sample times."
        )

    # get any filter information applied to the data
    lowpass = raw.info["lowpass"]
    highpass = raw.info["highpass"]
    linefreq = raw.info["line_freq"]
    filter_str_info = f"HP:{highpass}Hz LP:{lowpass}Hz N:{linefreq}Hz"

    if physical_range == "auto":
        # get max and min for each channel type data
        ch_types_phys_max = dict()
        ch_types_phys_min = dict()

        for _type in np.unique(ch_types):
            _picks = [n for n, t in zip(ch_names, ch_types) if t == _type]
            _data = raw.get_data(units=units, picks=_picks)
            ch_types_phys_max[_type] = _data.max()
            ch_types_phys_min[_type] = _data.min()
    else:
        # get the physical min and max of the data in uV
        # Physical ranges of the data in uV is usually set by the manufacturer
        # and properties of the electrode. In general, physical max and min
        # should be the clipping levels of the ADC input and they should be
        # the same for all channels. For example, Nihon Kohden uses +3200 uV
        # and -3200 uV for all EEG channels (which are the actual clipping
        # levels of their input amplifiers & ADC).
        # For full discussion, see: https://github.com/sccn/eeglab/issues/246
        pmin, pmax = physical_range[0], physical_range[1]

        # check that physical min and max is not exceeded
        if data.max() > pmax:
            warn(
                f"The maximum μV of the data {data.max()} is "
                f"more than the physical max passed in {pmax}.",
            )
        if data.min() < pmin:
            warn(
                f"The minimum μV of the data {data.min()} is "
                f"less than the physical min passed in {pmin}.",
            )
        data = np.clip(data, pmin, pmax)
    signals: list[EdfSignal] = []
    for idx, ch in enumerate(ch_names):
        ch_type = ch_types[idx]
        signal_label = f"{ch_type.upper()} {ch}" if add_ch_type else ch
        if len(signal_label) > 16:
            raise RuntimeError(
                f"Signal label for {ch} ({ch_type}) is "
                f"longer than 16 characters, which is not "
                f"supported in EDF. Please shorten the "
                f"channel name before exporting to EDF."
            )

        if physical_range == "auto":
            # take the channel type minimum and maximum
            pmin = ch_types_phys_min[ch_type]
            pmax = ch_types_phys_max[ch_type]

        signals.append(
            EdfSignal(
                data[idx],
                out_sfreq,
                label=signal_label,
                transducer_type="",
                physical_dimension=phys_dims,
                physical_range=(pmin, pmax),
                digital_range=(digital_min, digital_max),
                prefiltering=filter_str_info,
            )
        )

    # set patient info
    subj_info = raw.info.get("subject_info")
    if subj_info is not None:
        # get the full name of subject if available
        first_name = subj_info.get("first_name", "")
        middle_name = subj_info.get("middle_name", "")
        last_name = subj_info.get("last_name", "")
        name = " ".join(filter(None, [first_name, middle_name, last_name]))

        birthday = subj_info.get("birthday")
        hand = subj_info.get("hand")
        weight = subj_info.get("weight")
        height = subj_info.get("height")
        sex = subj_info.get("sex")

        additional_patient_info = []
        for key, value in [("height", height), ("weight", weight), ("hand", hand)]:
            if value:
                additional_patient_info.append(f"{key}={value}")
        if len(additional_patient_info) == 0:
            additional_patient_info = None
        else:
            additional_patient_info = " ".join(additional_patient_info)

        patient = Patient(
            code=subj_info.get("his_id", ""),
            sex={0: "X", 1: "M", 2: "F", None: "X"}[sex],
            birthdate=dt.date(*birthday),
            name=name.replace(" ", "_"),
            additional=additional_patient_info.split(),
        )
    else:
        patient = None

    # set measurement date
    if meas_date := raw.info["meas_date"]:
        startdate = dt.date(meas_date.year, meas_date.month, meas_date.day)
        starttime = dt.time(
            meas_date.hour, meas_date.minute, meas_date.second, meas_date.microsecond
        )
    else:
        startdate = None
        starttime = None

    device_info = raw.info.get("device_info")
    if device_info is not None:
        device_type = device_info.get("type")
        recording = Recording(startdate=startdate, equipment_code=device_type)
    else:
        recording = Recording(startdate=startdate)

    annotations = []
    for desc, onset, duration, ch_names in zip(
        raw.annotations.description,
        raw.annotations.onset,
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

    Edf(
        signals=signals,
        patient=patient,
        recording=recording,
        starttime=starttime,
        data_record_duration=data_record_duration,
        annotations=annotations,
    ).write(fname)
