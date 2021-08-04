# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import sys
import numpy as np

from ..utils import _check_edflib_installed
_check_edflib_installed()
from edflib_python import EDFwriter  # noqa: E402


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

    # create instance of EDF Writer
    hdl = EDFwriter(fname, EDFwriter.EDFLIB_FILETYPE_EDFPLUS, n_chs)

    # set channel data
    for ichan, ch in enumerate(ch_names):
        cals = raw.info['chs'][ichan]['cal']
        digital_min = - cals / 2
        digital_max = cals / 2
        print(digital_min, digital_max)
        if hdl.setPhysicalMaximum(ichan, 3000) != 0:
            print("setPhysicalMaximum() returned an error")
            sys.exit()
        if hdl.setPhysicalMinimum(ichan, -3000) != 0:
            print("setPhysicalMinimum() returned an error")
            sys.exit()
        if hdl.setDigitalMaximum(ichan, digital_max) != 0:
            print("setDigitalMaximum() returned an error")
            sys.exit()
        if hdl.setDigitalMinimum(ichan, digital_min) != 0:
            print("setDigitalMinimum() returned an error")
            sys.exit()
        if hdl.setPhysicalDimension(ichan, phys_dims) != 0:
            print("setPhysicalDimension() returned an error")
            sys.exit()
        if hdl.setSampleFrequency(ichan, sfreq) != 0:
            print("setSampleFrequency() returned an error")
            sys.exit()
        if hdl.setSignalLabel(ichan, ch) != 0:
            print("setSignalLabel() returned an error")
            sys.exit()
        # if hdl.setPreFilter(ichan, "HP:0.05Hz LP:40Hz N:60Hz") != 0:
        #     print("setPreFilter() returned an error")
        #     sys.exit()
        # if hdl.setTransducer(ichan, "AgAgCl cup electrode") != 0:
        #     print("setTransducer() returned an error")
        #     sys.exit()

    # set patient info
    subj_info = raw.info.get('subject_info')
    if subj_info is not None:
        birthday = subj_info.get('birthday')
        name = subj_info.get('first_name') + subj_info.get('last_name')
        hand = subj_info.get('hand')
        sex = subj_info.get('sex')

        if birthday is not None:
            if hdl.setPatientBirthDate(birthday[0], birthday[1], birthday[2]) != 0:
                print("setPatientBirthDate() returned an error")
                sys.exit()
        if hdl.setPatientName(name) != 0:
            print("setPatientName() returned an error")
            sys.exit()
        if hdl.setPatientGender(sex) != 0:
            print("setPatientGender() returned an error")
            sys.exit()
        
        if hdl.setAdditionalPatientInfo(f"hand={hand}") != 0:
            print("setAdditionalPatientInfo() returned an error")
            sys.exit()

    # set measurement date
    meas_date = raw.info['meas_date']
    if meas_date:
        # TODO: add support for subseconds
        if hdl.setStartDateTime(year=meas_date.year, month=meas_date.month,
                                day=meas_date.day, hour=meas_date.hour, 
                                minute=meas_date.minute,
                                second=meas_date.second) != 0:
            print("setStartDateTime() returned an error")
            sys.exit()
    # if hdl.setAdministrationCode("1234567890") != 0:
    #     print("setAdministrationCode() returned an error")
    #     sys.exit()
    # if hdl.setTechnician("Black Jack") != 0:
    #     print("setTechnician() returned an error")
    #     sys.exit()

    device_info = raw.info.get('device_info')
    if device_info is not None:
        device_type = device_info.get('type')
        if hdl.setEquipment(device_type) != 0:
            print("setEquipment() returned an error")
            sys.exit()
    # if hdl.setAdditionalRecordingInfo("nothing special") != 0:
    #     print("setAdditionalRecordingInfo() returned an error")
    #     sys.exit()

    # get data in uV
    data = raw.get_data(units=phys_dims, picks=ch_names)

    # write each second separately
    for isec in range(np.ceil(n_secs).astype(int) - 1):
        end_samp = (isec + 1) * sfreq
        if end_samp > n_times:
            end_samp = n_times
        start_samp = isec * sfreq
        # then for each second write each channel
        for ich in range(n_chs):
            # create a buffer with sampling rate
            buf = np.empty(sfreq, np.float64, "C")

            # get channel data for this second
            ch_data = data[ich, start_samp:end_samp]
            if any(np.isnan(ch_data)):
                print('ch data is nan...')
            buf[:len(ch_data)] = ch_data
            err = hdl.writeSamples(buf)
            if err != 0:
                raise RuntimeError(f"writeSamples() returned error: {err}")

    # write annotations
    if raw.annotations:
        annotations = [raw.annotations.description,
                       raw.annotations.onset,
                       raw.annotations.duration]
        for desc, onset, duration in annotations:
            if hdl.writeAnnotation(onset, duration, desc) != 0:
                raise RuntimeError(f'writeAnnotation() returned an error '
                                   f'trying to write {desc} at {onset} '
                                   f'for {duration} seconds.')
    hdl.close()
