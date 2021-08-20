# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import numpy as np

from ..utils import _check_edflib_installed, warn
_check_edflib_installed()
from EDFlib.edfwriter import EDFwriter  # noqa: E402


def _try_to_set_value(header, key, value, channel_index=None):
    """Set key/value pairs in EDF/BDF header."""
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


def _export_raw(fname, raw, physical_range, fmt):
    """Export Raw objects to EDF or BDF files.

    TODO: if in future the Info object supports transducer or
    technician information, allow writing those here.
    """
    phys_dims = 'uV'

    if fmt == 'bdf':
        digital_min = -8388607
        digital_max = 8388607
        file_type = EDFwriter.EDFLIB_FILETYPE_BDFPLUS
    elif fmt == 'edf':
        digital_min = -32767
        digital_max = 32767
        file_type = EDFwriter.EDFLIB_FILETYPE_EDFPLUS

    # load data first
    raw.load_data()

    # remove extra epoc and STI channels
    drop_chs = ['epoc']
    orig_ch_types = raw.get_channel_types()
    if 'stim' in orig_ch_types:
        stim_index = np.argwhere(orig_ch_types == 'stim')
        drop_chs.extend([raw.ch_names[idx] for idx in stim_index])

    ch_names = [ch for ch in raw.ch_names if ch not in drop_chs]
    n_channels = len(ch_names)
    n_times = raw.n_times

    # sampling frequency in EDF only supports integers
    sfreq = raw.info['sfreq']
    if sfreq.is_integer():
        channel_sfreq = int(sfreq)
        data_record_duration = None
    else:
        channel_sfreq = np.floor(sfreq).astype(int)
        data_record_duration = int(np.around(
            channel_sfreq / sfreq, decimals=6) * 1e6)

    n_secs = n_times / sfreq

    # get any filter information applied to the data
    lowpass = raw.info['lowpass']
    highpass = raw.info['highpass']
    linefreq = raw.info['line_freq']
    filter_str_info = f"HP:{highpass}Hz LP:{lowpass}Hz N:{linefreq}Hz"

    # get EEG-related data in uV
    units = dict(eeg='uV', ecog='uV', seeg='uV',
                 eog='uV', ecg='uV', emg='uV')

    # get the entire dataset in uV
    data = raw.get_data(units=units, picks=ch_names)

    if physical_range == 'auto':
        # get max and min for each channel type data
        ch_types_phys_max = dict()
        ch_types_phys_min = dict()

        ch_types = raw.get_channel_types(picks=raw.ch_names)

        for idx, ch_type in enumerate(ch_types):
            ch_type_data = data[idx, :]
            if ch_type not in ch_types_phys_max:
                ch_types_phys_max[ch_type] = ch_type_data.max()
                ch_types_phys_min[ch_type] = ch_type_data.min()
            else:
                ch_types_phys_max[ch_type] = max(
                    ch_type_data.max(), ch_types_phys_max[ch_type])
                ch_types_phys_min[ch_type] = min(
                    ch_type_data.min(), ch_types_phys_min[ch_type])
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
            raise RuntimeError(f'The maximum uV of the data {data.max()} '
                               f'is more then physical max passed in {pmax}.')
        if data.min() < pmin:
            raise RuntimeError(f'The minimum uV of the data {data.min()} '
                               f'is less then physical min passed in {pmin}.')

    # create instance of EDF/BDF Writer
    hdl = EDFwriter(fname, file_type, n_channels)

    # set channel data
    for ichan, ch in enumerate(ch_names):
        if physical_range == 'auto':
            # take the channel type minimum and maximum
            ch_type = ch_types[ichan]
            pmin, pmax = ch_types_phys_min[ch_type], ch_types_phys_max[ch_type]

        for key, val in [('PhysicalMaximum', pmax),
                         ('PhysicalMinimum', pmin),
                         ('DigitalMaximum', digital_max),
                         ('DigitalMinimum', digital_min),
                         ('PhysicalDimension', phys_dims),
                         ('SampleFrequency', channel_sfreq),
                         ('SignalLabel', ch),
                         ('PreFilter', filter_str_info)]:
            _try_to_set_value(hdl, key, val, channel_index=ichan)

    # set patient info
    subj_info = raw.info.get('subject_info')
    if subj_info is not None:
        birthday = subj_info.get('birthday')
        first_name = subj_info.get('first_name')
        last_name = subj_info.get('last_name')
        name = None
        if first_name is not None:
            name = first_name
        if last_name is not None:
            name += last_name
        if name is None:
            name = ''

        hand = subj_info.get('hand')
        sex = subj_info.get('sex')

        if birthday is not None:
            if hdl.setPatientBirthDate(birthday[0], birthday[1],
                                       birthday[2]) != 0:
                raise RuntimeError(f"Setting Patient Birth Date to {birthday} "
                                   f"returned an error")
        for key, val in [('PatientName', name),
                         ('PatientGender', sex),
                         ('AdditionalPatientInfo', f'hand={hand}')]:
            _try_to_set_value(hdl, key, val)

    # set measurement date
    meas_date = raw.info['meas_date']
    if meas_date:
        subsecond = meas_date.microsecond / 100.
        if hdl.setStartDateTime(year=meas_date.year, month=meas_date.month,
                                day=meas_date.day, hour=meas_date.hour,
                                minute=meas_date.minute,
                                second=meas_date.second,
                                subsecond=subsecond) != 0:  # noqa
            raise RuntimeError(f"Setting Start Date Time {meas_date} "
                               f"returned an error")

    device_info = raw.info.get('device_info')
    if device_info is not None:
        device_type = device_info.get('type')
        _try_to_set_value(hdl, 'Equipment', device_type)

    # set data record duration
    _try_to_set_value(hdl, 'DataRecordDuration', data_record_duration)

    # compute number of seconds to loop over
    n_secs = n_times / channel_sfreq

    # Write each second (i.e. datarecord) separately.
    # buffer_sfreq = (int(sfreq * data_record_duration))
    for isec in range(np.ceil(n_secs).astype(int)):
        end_samp = (isec + 1) * channel_sfreq
        if end_samp > n_times:
            end_samp = n_times
        start_samp = isec * channel_sfreq

        # then for each second write each channel
        for ich in range(n_channels):
            # create a buffer with sampling rate
            buf = np.zeros(channel_sfreq, np.float64, "C")

            # get channel data for this second
            ch_data = data[ich, start_samp:end_samp]

            buf[:len(ch_data)] = ch_data
            err = hdl.writeSamples(buf)
            if err != 0:  # noqa
                raise RuntimeError(f"writeSamples() returned error: {err}")

        # there was an incomplete datarecord
        if len(ch_data) != len(buf):
            warn(f'EDF format requires equal-length data blocks, '
                 f'so {(len(buf) - len(ch_data)) / sfreq} seconds of zeros '
                 f'were appended to all channels when writing the final '
                 f'block.')

    # write annotations
    # XXX: possibly writing multiple annotations per data record is not
    # possible, but can be expanded if we write to more then one channel
    if raw.annotations:
        for desc, onset, duration in zip(raw.annotations.description,
                                         raw.annotations.onset,
                                         raw.annotations.duration):
            if hdl.writeAnnotation(onset, duration, desc) != 0:  # noqa
                raise RuntimeError(f'writeAnnotation() returned an error '
                                   f'trying to write {desc} at {onset} '
                                   f'for {duration} seconds.')
    hdl.close()
