# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

from contextlib import contextmanager
import numpy as np

from ..utils import _check_edflib_installed, warn
_check_edflib_installed()
from EDFlib.edfwriter import EDFwriter  # noqa: E402


def _try_to_set_value(header, key, value, channel_index=None):
    """Set key/value pairs in EDF header."""
    # all EDFLib set functions are set<X>
    # for example "setPatientName()"
    func_name = f'set{key}'
    func = getattr(header, func_name)

    # some setter functions are indexed by channels
    if channel_index is None:
        return_val = func(value)
    else:
        return_val = func(channel_index, value)

    # a nonzero return value indicates an error
    if return_val != 0:
        raise RuntimeError(f"Setting {key} with {value} "
                           f"returned an error value "
                           f"{return_val}.")


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
    phys_dims = 'uV'

    # get EEG-related data in uV
    units = dict(eeg='uV', ecog='uV', seeg='uV', eog='uV', ecg='uV', emg='uV',
                 bio='uV', dbs='uV')

    digital_min = -32767
    digital_max = 32767
    file_type = EDFwriter.EDFLIB_FILETYPE_EDFPLUS

    # load data first
    raw.load_data()

    # remove extra STI channels
    orig_ch_types = raw.get_channel_types()
    drop_chs = []
    if 'stim' in orig_ch_types:
        stim_index = np.argwhere(np.array(orig_ch_types) == 'stim')
        stim_index = np.atleast_1d(stim_index.squeeze()).tolist()
        drop_chs.extend([raw.ch_names[idx] for idx in stim_index])

    # Add warning if any channel types are not voltage based.
    # Users are expected to only export data that is voltage based,
    # such as EEG, ECoG, sEEG, etc.
    # Non-voltage channels are dropped by the export function.
    # Note: we can write these other channels, such as 'misc'
    # but these are simply a "catch all" for unknown or undesired
    # channels.
    voltage_types = list(units) + ['stim', 'misc']
    non_voltage_ch = [ch not in voltage_types for ch in orig_ch_types]
    if any(non_voltage_ch):
        warn(f"Non-voltage channels detected: {non_voltage_ch}. MNE-Python's "
             'EDF exporter only supports voltage-based channels, because the '
             'EDF format cannot accommodate much of the accompanying data '
             'necessary for channel types like MEG and fNIRS (channel '
             'orientations, coordinate frame transforms, etc). You can '
             'override this restriction by setting those channel types to '
             '"misc" but no guarantees are made of the fidelity of that '
             'approach.')

    ch_names = [ch for ch in raw.ch_names if ch not in drop_chs]
    ch_types = np.array(raw.get_channel_types(picks=ch_names))
    n_channels = len(ch_names)
    n_times = raw.n_times

    # Sampling frequency in EDF only supports integers, so to allow for
    # float sampling rates from Raw, we adjust the output sampling rate
    # for all channels and the data record duration.
    sfreq = raw.info['sfreq']
    if float(sfreq).is_integer():
        out_sfreq = int(sfreq)
        data_record_duration = None
    else:
        out_sfreq = np.floor(sfreq).astype(int)
        data_record_duration = int(np.around(
            out_sfreq / sfreq, decimals=6) * 1e6)

        warn(f'Data has a non-integer sampling rate of {sfreq}; writing to '
             'EDF format may cause a small change to sample times.')

    # get any filter information applied to the data
    lowpass = raw.info['lowpass']
    highpass = raw.info['highpass']
    linefreq = raw.info['line_freq']
    filter_str_info = f"HP:{highpass}Hz LP:{lowpass}Hz N:{linefreq}Hz"

    # get the entire dataset in uV
    data = raw.get_data(units=units, picks=ch_names)

    if physical_range == 'auto':
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
            raise RuntimeError(f'The maximum μV of the data {data.max()} is '
                               f'more than the physical max passed in {pmax}.')
        if data.min() < pmin:
            raise RuntimeError(f'The minimum μV of the data {data.min()} is '
                               f'less than the physical min passed in {pmin}.')

    # create instance of EDF Writer
    with _auto_close(EDFwriter(fname, file_type, n_channels)) as hdl:
        # set channel data
        for idx, ch in enumerate(ch_names):
            ch_type = ch_types[idx]
            signal_label = f'{ch_type.upper()} {ch}' if add_ch_type else ch
            if len(signal_label) > 16:
                raise RuntimeError(f'Signal label for {ch} ({ch_type}) is '
                                   f'longer than 16 characters, which is not '
                                   f'supported in EDF. Please shorten the '
                                   f'channel name before exporting to EDF.')

            if physical_range == 'auto':
                # take the channel type minimum and maximum
                pmin = ch_types_phys_min[ch_type]
                pmax = ch_types_phys_max[ch_type]
            for key, val in [('PhysicalMaximum', pmax),
                             ('PhysicalMinimum', pmin),
                             ('DigitalMaximum', digital_max),
                             ('DigitalMinimum', digital_min),
                             ('PhysicalDimension', phys_dims),
                             ('SampleFrequency', out_sfreq),
                             ('SignalLabel', signal_label),
                             ('PreFilter', filter_str_info)]:
                _try_to_set_value(hdl, key, val, channel_index=idx)

        # set patient info
        subj_info = raw.info.get('subject_info')
        if subj_info is not None:
            birthday = subj_info.get('birthday')

            # get the full name of subject if available
            first_name = subj_info.get('first_name')
            last_name = subj_info.get('last_name')
            first_name = first_name or ''
            last_name = last_name or ''
            joiner = ''
            if len(first_name) and len(last_name):
                joiner = ' '
            name = joiner.join([first_name, last_name])

            hand = subj_info.get('hand')
            sex = subj_info.get('sex')

            if birthday is not None:
                if hdl.setPatientBirthDate(birthday[0], birthday[1],
                                           birthday[2]) != 0:
                    raise RuntimeError(
                        f"Setting patient birth date to {birthday} "
                        f"returned an error")
            for key, val in [('PatientName', name),
                             ('PatientGender', sex),
                             ('AdditionalPatientInfo', f'hand={hand}')]:
                # EDFwriter compares integer encodings of sex and will
                # raise a TypeError if value is None as returned by
                # subj_info.get(key) if key is missing.
                if val is not None:
                    _try_to_set_value(hdl, key, val)

        # set measurement date
        meas_date = raw.info['meas_date']
        if meas_date:
            subsecond = int(meas_date.microsecond / 100)
            if hdl.setStartDateTime(year=meas_date.year, month=meas_date.month,
                                    day=meas_date.day, hour=meas_date.hour,
                                    minute=meas_date.minute,
                                    second=meas_date.second,
                                    subsecond=subsecond) != 0:
                raise RuntimeError(f"Setting start date time {meas_date} "
                                   f"returned an error")

        device_info = raw.info.get('device_info')
        if device_info is not None:
            device_type = device_info.get('type')
            _try_to_set_value(hdl, 'Equipment', device_type)

        # set data record duration
        if data_record_duration is not None:
            _try_to_set_value(hdl, 'DataRecordDuration', data_record_duration)

        # compute number of data records to loop over
        n_blocks = np.ceil(n_times / out_sfreq).astype(int)

        # increase the number of annotation signals if necessary
        annots = raw.annotations
        if annots is not None:
            n_annotations = len(raw.annotations)
            n_annot_chans = int(n_annotations / n_blocks)
            if np.mod(n_annotations, n_blocks):
                n_annot_chans += 1
            if n_annot_chans > 1:
                hdl.setNumberOfAnnotationSignals(n_annot_chans)

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
                buf[:len(ch_data)] = ch_data
                err = hdl.writeSamples(buf)
                if err != 0:
                    raise RuntimeError(
                        f"writeSamples() for channel{ch_names[jdx]} "
                        f"returned error: {err}")

            # there was an incomplete datarecord
            if len(ch_data) != len(buf):
                warn(f'EDF format requires equal-length data blocks, '
                     f'so {(len(buf) - len(ch_data)) / sfreq} seconds of '
                     'zeros were appended to all channels when writing the '
                     'final block.')

        # write annotations
        if annots is not None:
            for desc, onset, duration in zip(raw.annotations.description,
                                             raw.annotations.onset,
                                             raw.annotations.duration):
                # annotations are written in terms of 100 microseconds
                onset = onset * 10000
                duration = duration * 10000
                if hdl.writeAnnotation(onset, duration, desc) != 0:
                    raise RuntimeError(f'writeAnnotation() returned an error '
                                       f'trying to write {desc} at {onset} '
                                       f'for {duration} seconds.')
