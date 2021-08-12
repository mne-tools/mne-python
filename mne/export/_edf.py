# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import numpy as np

from ..utils import _check_edflib_installed, warn
_check_edflib_installed()
from EDFlib.edfwriter import EDFwriter  # noqa: E402


def _try_to_set_value(header, key, value, channel_index=None):
    """Helper function to set key/value pairs in EDF/BDF header."""
    # all EDFLib set functions are set<X>
    # for example "setPatientName()"
    func_name = f'set{key}'
    func = getattr(header, func_name)

    # some setter functions are indexed by channels
    if channel_index is None:
        return_val = func(value)
    else:
        return_val = func(channel_index, value)

    # a return value of 0 indicates an error
    if return_val != 0:
        raise RuntimeError(f"Setting {key} with {value} "
                           f"returned an error")


def _export_raw(fname, raw, physical_range, fmt):
    """Export Raw objects to EDF or BDF files.

    TODO:
    - if Info stores transducer information, allow writing here.
    - if Info stores techniciain information, allow writing here.
    """
    phys_dims = 'uV'

    if fmt == 'bdf':
        digital_min = -8388607
        digital_max = 8388607
    elif fmt == 'edf':
        digital_min = -32767
        digital_max = 32767

    # load data first
    raw.load_data()

    # remove extra epoc and STI channels
    drop_chs = ['epoc']
    if not (raw.filenames[0].endswith('.fif')):
        drop_chs.append('STI 014')

    ch_names = [ch for ch in raw.ch_names if ch not in drop_chs]
    n_chs = len(ch_names)
    n_times = raw.n_times
    sfreq = int(raw.info['sfreq'])
    n_secs = n_times / sfreq

    # get any filter information applied to the data
    lowpass = raw.info['lowpass']
    highpass = raw.info['highpass']
    linefreq = raw.info['line_freq']
    filter_str_info = f"HP:{highpass}Hz LP:{lowpass}Hz N:{linefreq}Hz"

    # get data in uV
    # XXX: what about non-EEG data?... How to handle?
    units = dict()
    units['eeg'] = 'uV'
    units['ecog'] = 'uV'
    units['seeg'] = 'uV'

    # get the entire dataset in uV
    data = raw.get_data(units=units, picks=ch_names)

    if physical_range == 'auto':
        # get max and min for each channel type data
        ch_types_phys_max = dict()
        ch_types_phys_min = dict()

        for idx, ch_name in enumerate(raw.ch_names):
            ch_type = raw.get_channel_types(picks=ch_name)[0]

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
        pmin, pmax = -3200, 3200

        # check that physical min and max is not exceeded
        if data.max() > pmax:
            raise RuntimeError(f'The maximum uV of the data {data.max()} '
                               f'is more then physical max passed in {pmax}.')
        if data.min() < pmin:
            raise RuntimeError('The minimum uV of the data {data.min()} '
                               f'is less then physical min passed in {pmin}.')

    # create instance of EDF Writer
    hdl = EDFwriter(fname, EDFwriter.EDFLIB_FILETYPE_EDFPLUS, n_chs)

    # set channel data
    for ichan, ch in enumerate(ch_names):
        if physical_range == 'auto':
            # take the channel type minimum and maximum
            ch_type = raw.get_channel_types(picks=ch)[0]
            pmin, pmax = ch_types_phys_min[ch_type], ch_types_phys_max[ch_type]

        for key, val in zip(
            ['PhysicalMaximum', 'PhysicalMinimum', 'DigitalMaximum',
             'DigitalMinimum', 'PhysicalDimension', 'SampleFrequency',
             'SignalLabel', 'PreFilter',
             ],
            [pmax, pmin, digital_max, digital_min, phys_dims, sfreq,
             ch, filter_str_info]
        ):
            _try_to_set_value(hdl, key, val, channel_index=ichan)

    # set patient info
    subj_info = raw.info.get('subject_info')
    if subj_info is not None:
        birthday = subj_info.get('birthday')
        name = subj_info.get('first_name') + subj_info.get('last_name')
        hand = subj_info.get('hand')
        sex = subj_info.get('sex')

        if birthday is not None:
            if hdl.setPatientBirthDate(birthday[0], birthday[1],
                                       birthday[2]) != 0:
                raise RuntimeError(f"Setting Patient Birth Date to {birthday} "
                                   f"returned an error")
        for key, val in zip(['PatientName', 'PatientGender',
                             'AdditionalPatientInfo'],
                            [name, sex, f'hand={hand}']):
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

    # Write each second (i.e. datarecord) separately.
    for isec in range(np.ceil(n_secs).astype(int)):
        end_samp = (isec + 1) * sfreq
        if end_samp > n_times:
            end_samp = n_times
        start_samp = isec * sfreq

        # then for each second write each channel
        for ich in range(n_chs):
            # create a buffer with sampling rate
            buf = np.zeros(sfreq, np.float64, "C")

            # get channel data for this second
            ch_data = data[ich, start_samp:end_samp]

            buf[:len(ch_data)] = ch_data
            err = hdl.writeSamples(buf)
            if err != 0:  # noqa
                raise RuntimeError(f"writeSamples() returned error: {err}")

        # there was an incomplete datarecord
        if len(ch_data) != len(buf):
            warn(f'A complete data record consists of {len(buf)} samples, '
                 f'but this sample window ended up having {len(ch_data)} '
                 f'samples. {len(buf) - len(ch_data)} zeros were appended '
                 f'to the datarecord.')
            warn(f'EDF format requires equal-length data blocks, '
                 f'so {(len(buf) - len(ch_data)) / sfreq} seconds of zeros '
                 f'were appended to all channels when writing the final '
                 f'block.')

    # write annotations
    # XXX: possibly writing multiple annotations per data record is not
    # possible, but can be expanded if we write to more then one channel
    if raw.annotations:
        annotations = [raw.annotations.description,
                       raw.annotations.onset,
                       raw.annotations.duration]
        for desc, onset, duration in annotations:
            if hdl.writeAnnotation(onset, duration, desc) != 0:  # noqa
                raise RuntimeError(f'writeAnnotation() returned an error '
                                   f'trying to write {desc} at {onset} '
                                   f'for {duration} seconds.')
    hdl.close()
