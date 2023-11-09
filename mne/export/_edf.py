# Authors: MNE Developers
#
# License: BSD-3-Clause

from contextlib import contextmanager
from datetime import date

import numpy as np
import pyedflib

# _check_pyedflib_installed()
from pyedflib import EdfWriter

from ..utils import warn


def _try_to_set_value(header, key, value, channel_index=None):
    """Set key/value pairs in EDF header."""
    # many pyedflib set functions are set<X>
    # for example "setPatientName()"
    func_name = f"set{key}"
    func = getattr(header, func_name)
    # some setter functions are indexed by channels
    try:
        if channel_index is None:
            func(value)
        else:
            func(channel_index, value)
    except RuntimeWarning:
        warn(
            f"Setting {key} with {value} "
            f"returned an error. "
            f"Setting to None instead."
        )
    except RuntimeError:
        raise RuntimeError(f"Setting {key} with {value} " f"returned an error.")
    # setDatarecordDuration cannot accept values larger than 2**32
    # except OverflowError:
    #     warn(
    #         f"Setting {key} with {value} "
    #         f"returned an error. "
    #         f"setDatarecordDuration() cannot accept values larger than 2**32. "
    #         f"Setting to None instead."
    #     )

    # pyedflib setSamplefrequency returns warning:
    #  DeprecationWarning: `sample_rate` is deprecated and
    # will be removed in a future release.
    # Please use `sample_frequency` instead
    # This causes test to fail, so we catch it here
    except DeprecationWarning:
        pass


@contextmanager
def _auto_close(fid):
    # try to close the handle no matter what
    try:
        yield fid
    finally:
        try:
            fid.close()
        except Exception:
            pass  # we did our best


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
    file_type = pyedflib.FILETYPE_EDFPLUS

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
    n_channels = len(ch_names)
    n_times = raw.n_times

    # Sampling frequency in EDF only supports integers, so to allow for
    # float sampling rates from Raw, we adjust the output sampling rate
    # for all channels and the data record duration.
    # ATTENTION: the argument "duration" is expressed in units of seconds!
    # As of 10/5/23 there is a error in the documentation of pyedflib
    # that states that the duration is expressed in units of nanoseconds.
    # See: https://github.com/holgern/pyedflib/issues/242
    sfreq = raw.info["sfreq"]
    if float(sfreq).is_integer():
        out_sfreq = int(sfreq)
        data_record_duration = None
    else:
        out_sfreq = np.floor(sfreq).astype(int)
        data_record_duration = out_sfreq / sfreq

        warn(
            f"Data has a non-integer sampling rate of {sfreq}; writing to "
            "EDF format may cause a small change to sample times."
        )

    # get any filter information applied to the data
    lowpass = raw.info["lowpass"]
    highpass = raw.info["highpass"]
    linefreq = raw.info["line_freq"]
    filter_str_info = f"HP:{highpass}Hz LP:{lowpass}Hz N:{linefreq}Hz"

    # get the entire dataset in uV
    data = raw.get_data(units=units, picks=ch_names)

    if physical_range == "auto":
        # get max and min for each channel type data
        ch_types_phys_max = dict()
        ch_types_phys_min = dict()

        for _type in np.unique(ch_types):
            _picks = np.nonzero(ch_types == _type)[0]
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

    # create instance of EDF Writer
    with _auto_close(
        EdfWriter(fname, n_channels=n_channels, file_type=file_type)
    ) as hdl:
        # set channel data
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
            for key, val in [
                ("PhysicalMaximum", pmax),
                ("PhysicalMinimum", pmin),
                ("DigitalMaximum", digital_max),
                ("DigitalMinimum", digital_min),
                ("PhysicalDimension", phys_dims),
                ("Samplefrequency", out_sfreq),
                ("Label", signal_label),
                ("Prefilter", filter_str_info),
            ]:
                _try_to_set_value(hdl, key, val, channel_index=idx)

        # set patient info
        subj_info = raw.info.get("subject_info")
        if subj_info is not None:
            # get the full name of subject if available
            first_name = subj_info.get("first_name", "")
            middle_name = subj_info.get("middle_name", "")
            last_name = subj_info.get("last_name", "")
            # pyedflib does not support spaces in the patient name
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
            if len(additional_patient_info) == 0:
                additional_patient_info = None
            else:
                # pyedflib does not support spaces in the patient info
                additional_patient_info = "_".join(additional_patient_info)

            if birthday is not None:
                _try_to_set_value(hdl, "Birthdate", date(*birthday))

            for key, val in [
                ("PatientCode", subj_info.get("his_id", "")),
                ("PatientName", name),
                ("Gender", sex),
                ("PatientAdditional", additional_patient_info),
            ]:
                # EDFwriter compares integer encodings of sex and will
                # raise a TypeError if value is None as returned by
                # subj_info.get(key) if key is missing.
                if val is not None:
                    _try_to_set_value(hdl, key, val)

        # set measurement date
        meas_date = raw.info["meas_date"]
        if meas_date is not None:
            _try_to_set_value(hdl, "Startdatetime", meas_date)

        device_info = raw.info.get("device_info")
        if device_info is not None:
            device_type = device_info.get("type")
            _try_to_set_value(hdl, "Equipment", device_type)

        # set data record duration
        if data_record_duration is not None:
            _try_to_set_value(hdl, "DatarecordDuration", data_record_duration)

        # compute number of data records to loop over
        n_blocks = np.ceil(n_times / out_sfreq).astype(int)

        # increase the number of annotation signals if necessary
        annots = raw.annotations
        if annots is not None:
            n_annotations = len(raw.annotations)
            n_annot_chans = int(n_annotations / n_blocks) + 1
            if n_annot_chans > 1:
                hdl.set_number_of_annotation_signals(n_annot_chans)

        # Write each data record sequentially
        for idx in range(n_blocks):
            end_samp = (idx + 1) * out_sfreq
            if end_samp > n_times:
                end_samp = n_times
            start_samp = idx * out_sfreq

            # then for each datarecord write each channel
            for jdx in range(n_channels):
                # create a buffer with sampling rate
                buf = np.zeros(out_sfreq, np.float64, "C")

                # get channel data for this block
                ch_data = data[jdx, start_samp:end_samp]

                # assign channel data to the buffer and write to EDF
                buf[: len(ch_data)] = ch_data
                hdl.writePhysicalSamples(buf)
                # if err != 0:
                #     raise RuntimeError(
                #         f"writeSamples() for channel{ch_names[jdx]} "
                #         f"returned error: {err}"
                #     )

            # there was an incomplete datarecord
            if len(ch_data) != len(buf):
                warn(
                    f"EDF format requires equal-length data blocks, "
                    f"so {(len(buf) - len(ch_data)) / sfreq} seconds of "
                    "zeros were appended to all channels when writing the "
                    "final block."
                )
        # Round data to nearest 2 decimal places
        # data = np.round(data, decimals=2)

        # write data
        # hdl.writeSamples(data)

        # write annotations
        if annots is not None:
            for desc, onset, duration, ch_names in zip(
                raw.annotations.description,
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.ch_names,
            ):
                # annotations are written in terms of seconds
                if ch_names:
                    for ch_name in ch_names:
                        try:
                            hdl.writeAnnotation(onset, duration, desc + f"@@{ch_name}")
                        except Exception as e:
                            raise RuntimeError(
                                f"writeAnnotation() returned an error "
                                f"trying to write {desc} at {onset} "
                                f"for {duration} seconds."
                            ) from e
    del hdl
