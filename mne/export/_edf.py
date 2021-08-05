# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import numpy as np

from ..utils import _check_edflib_installed, warn
_check_edflib_installed()
from EDFlib.edfwriter import EDFwriter  # noqa: E402


def _export_raw(fname, raw):
    phys_dims = 'uV'

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

    # get data in uV
    units = dict()
    if 'eeg' in raw:
        units['eeg'] = 'uV'
    if 'ecog' in raw:
        units['ecog'] = 'uV'
    if 'seeg' in raw:
        units['seeg'] = 'uV'
    data = raw.get_data(units=units, picks=ch_names)

    # get the physical min and max of the data
    pmin, pmax = data.min(axis=1), data.max(axis=1)

    # create instance of EDF Writer
    hdl = EDFwriter(fname, EDFwriter.EDFLIB_FILETYPE_EDFPLUS, n_chs)

    # set channel data
    for ichan, ch in enumerate(ch_names):
        if hdl.setPhysicalMaximum(ichan, pmax[ichan]) != 0:  # noqa
            raise RuntimeError("setPhysicalMaximum() returned an error")
        if hdl.setPhysicalMinimum(ichan, pmin[ichan]) != 0:  # noqa
            raise RuntimeError("setPhysicalMinimum() returned an error")
        if hdl.setDigitalMaximum(ichan, 32767) != 0:  # noqa
            raise RuntimeError("setDigitalMaximum() returned an error")
        if hdl.setDigitalMinimum(ichan, -32767) != 0:  # noqa
            raise RuntimeError("setDigitalMinimum() returned an error")
        if hdl.setPhysicalDimension(ichan, phys_dims) != 0:  # noqa
            raise RuntimeError("setPhysicalDimension() returned an error")
        if hdl.setSampleFrequency(ichan, sfreq) != 0:  # noqa
            raise RuntimeError("setSampleFrequency() returned an error")
        if hdl.setSignalLabel(ichan, ch) != 0:  # noqa
            raise RuntimeError("setSignalLabel() returned an error")
        # if hdl.setPreFilter(ichan, "HP:0.05Hz LP:40Hz N:60Hz") != 0:
        #     raise RuntimeError("setPreFilter() returned an error")
        #     sys.exit()
        # if hdl.setTransducer(ichan, "AgAgCl cup electrode") != 0:
        #     raise RuntimeError("setTransducer() returned an error")
        #     sys.exit()

    # set patient info
    subj_info = raw.info.get('subject_info')
    if subj_info is not None:
        birthday = subj_info.get('birthday')
        name = subj_info.get('first_name') + subj_info.get('last_name')
        hand = subj_info.get('hand')
        sex = subj_info.get('sex')

        if birthday is not None:
            if hdl.setPatientBirthDate(birthday[0], birthday[1], birthday[2]) != 0:  # noqa
                raise RuntimeError("setPatientBirthDate() returned an error")
        if hdl.setPatientName(name) != 0:  # noqa
            raise RuntimeError("setPatientName() returned an error")
        if hdl.setPatientGender(sex) != 0:  # noqa
            raise RuntimeError("setPatientGender() returned an error")

        if hdl.setAdditionalPatientInfo(f"hand={hand}") != 0:  # noqa
            raise RuntimeError("setAdditionalPatientInfo() returned an error")

    # set measurement date
    meas_date = raw.info['meas_date']
    if meas_date:
        subsecond = meas_date.microsecond / 100.
        if hdl.setStartDateTime(year=meas_date.year, month=meas_date.month,
                                day=meas_date.day, hour=meas_date.hour,
                                minute=meas_date.minute,
                                second=meas_date.second,
                                subsecond=subsecond) != 0:  # noqa
            raise RuntimeError("setStartDateTime() returned an error")
    # if hdl.setAdministrationCode("1234567890") != 0:
    #     raise RuntimeError("setAdministrationCode() returned an error")
    #     sys.exit()
    # if hdl.setTechnician("Black Jack") != 0:
    #     raise RuntimeError("setTechnician() returned an error")
    #     sys.exit()

    device_info = raw.info.get('device_info')
    if device_info is not None:
        device_type = device_info.get('type')
        if hdl.setEquipment(device_type) != 0:  # noqa
            raise RuntimeError("setEquipment() returned an error")
    # if hdl.setAdditionalRecordingInfo("nothing special") != 0:
    #     raise RuntimeError("setAdditionalRecordingInfo() returned an error")
    #     sys.exit()

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
