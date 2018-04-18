# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Thomas Hartmann & Dirk Gütlin
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import mne
import numpy
from mne.externals import pymatreader

"""
Purpose of this module is to read MEG or EEG data, created in MATLABs 'FieldTrip' toolbox and saved in .mat files,
into equivalent data structures of the MNE python module.
"""


def read_raw_ft(ft_structure_path, data_name='data'):
    """This function extracts FieldTrip single trial raw data structures (FT_DATATYPE_RAW) 
    from .mat files and converts them to MNE RawArrays.
    
    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    data_name: str, optional
        Name of heading dict/ variable name under which the data was originally saved
        in MATLAB. Default is 'data'.
        
    Returns
    -------
    mne.io.RawArray
        A MNE RawArray structure consisting of the raw array and measurement info
    
    Examples
    --------
    >>> read_raw("/home/usr/Documents/FieldTripRawFile.mat")
    <RawArray  |  None, n_channels x n_times : 402 x 60001 (60.0 sec), ~185.0 MB, data loaded>

    >>> read_raw("FieldTripRawFile2.mat", data_name="rawdata")
    <RawArray  |  None, n_channels x n_times : 323 x 35001 (35.0 sec), ~117.0 MB, data loaded>
    
    """

    ft_struct = pymatreader.read_mat(ft_structure_path, ignore_fields=['previous'], variable_names=[data_name])
    ft_struct = ft_struct[data_name]  #:load data and set ft_struct to the heading dictionary

    data = numpy.array(ft_struct["trial"])  #:create the main data array
    info = _create_info(ft_struct)  #: create info structure

    custom_raw = mne.io.RawArray(data, info)  #: create an MNE RawArray
    return custom_raw


def read_epochs_ft(ft_structure_path, data_name='data', trialinfo_map=None, omit_trialinfo_index=True, omit_non_unique_trialinfo_index=True):
    """This function extracts FieldTrip multiple trial raw data structures (FT_DATATYPE_RAW) 
    from .mat files and converts them to MNE EpochsArrays.

    .. warning:: Only epochs with the same amount of channels and samples are supported!

    The data is read as it is. Events however are represented entirely different in FieldTrip compared to MNE.

    In FieldTrip, each epoch corresponds to one row in the trialinfo field. This field can have one or more columns.
    The function first removes columns according to the two omit parameters.

    - If only one column remains, its values are used as event values in the MNE Epoch.
    - If two or more columns remain, each unique combination of these values receives a new event value. These event
      values are created automatically. In order to match these to conditions, you can use the trialinfo_map parameter.
    
    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    data_name: str, optional
        Name of heading dict/ variable name under which the data was originally saved
        in MATLAB. Default is 'data'.
    trialinfo_map: dict, optional
        A dictionary mapping condition strings (MNE's event_ids) to the trialinfo column. The values should be 1D numpy arrays.
        See examples for details.
    omit_trialinfo_index: bool, optional
        Omit trialinfo columns that look like an index of the trials, i.e. in which every row is the row before + 1.
    omit_non_unique_trialinfo_index: bool, optional
        Omit trialinfo columns that contain a different value for each row. These are most likely additional data
        like reaction times that cannot be represented in MNE.

        
    Returns
    -------
    mne.EpochsArray
        A MNE EpochsArray structure consisting of the epochs arrays, an event matrix,
        start time before event (if possible, else defaults to 0) and measurement info.
    
    Examples
    --------
    >>> read_epoched("/home/usr/Documents/FieldTripEpochsFile.mat")
    <EpochsArray  |  n_events : 123 (all good), tmin : -1.0 (s), tmax : 2.99 (s), baseline : None, ~115.7 MB, data loaded,
    '1': 21, '2': 9, '3': 12, '4': 22, '5': 20, '6': 7, '7': 9, '8': 23>

    >>> read_epoched("FieldTripEpochsFile2.mat", data_name="epoched")
    <EpochsArray  |  n_events : 10 (all good), tmin : -0.1 (s), tmax : 0.5 (s), baseline : None, ~87.5 MB, data loaded,
    '1': 5, '2': 5>

    >>> read_epoched('FieldTripEpochsFile', trialinfo_map={
    >>>    'audio/attend': numpy.array([0, 1]),
    >>>    'visual/attend': numpy.array([1, 1]),
    >>>    'audio/non_attend': numpy.array([0, 0]),
    >>>    'visual/non_attend': numpy.array([1, 0])})
    """

    ft_struct = pymatreader.read_mat(ft_structure_path, ignore_fields=['previous'], variable_names=[data_name])
    ft_struct = ft_struct[data_name]  #:load data and set ft_struct to the heading dictionary

    data = numpy.array(ft_struct["trial"])  #:create the epochs data array
    (events, event_id) = _create_events(ft_struct, trialinfo_map, omit_trialinfo_index, omit_non_unique_trialinfo_index)  #: create the events matrix
    tmin = _set_tmin(ft_struct)  #: create start time
    info = _create_info(ft_struct)  #: create info structure

    custom_epochs = mne.EpochsArray(data=data, info=info, tmin=tmin, events=events, event_id=event_id)  # create an MNE EpochsArray
    return custom_epochs


def read_evoked_ft(ft_structure_path, comment='', data_name='data'):
    """This function extracts FieldTrip timelock data structures (FT_DATATYPE_TIMELOCK) 
    from .mat files and converts them to MNE EvokedArrays.
    
    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    comment: str, optional
        Comment on dataset. Can be the condition. Default is ''
    data_name: str, optional
        Name of heading dict/ variable name under which the data was originally saved
        in MATLAB. Default is 'data'.
        
    Returns
    -------
    mne.EvokedArray
        A MNE EvokedArray structure consisting of the averaged data array,
         comment and measurement info.
    
    Examples
    --------
    >>> read_avg("/home/usr/Documents/FieldTripTimelockFile.mat")
    <Evoked  |  comment : '', kind : average, time : [0.000000, 3.990000], n_epochs : 1, n_channels x n_times : 306 x 400, ~1.7 MB>


    >>> read_avg("FieldTripTimelockFile2.mat", data_name="evoked")
    <Evoked  |  comment : '', kind : average, time : [0.000000, 2.490000], n_epochs : 1, n_channels x n_times : 306 x 250, ~1.1 MB>

    
    >>> read_avg("FieldTripTimelockFile.mat", comment= "visual stimulus")
    <Evoked  |  comment : 'visual stimulus', kind : average, time : [0.000000, 3.990000], n_epochs : 1, n_channels x n_times : 306 x 400, ~1.7 MB>

    """

    ft_struct = pymatreader.read_mat(ft_structure_path, ignore_fields=['previous'], variable_names=[data_name])
    ft_struct = ft_struct[data_name]  #:load data and set ft_struct to the heading dictionary

    data_evoked = ft_struct["avg"]  #:create evoked data
    info = _create_info(ft_struct)  #: create info structure

    evoked_array = mne.EvokedArray(data_evoked, info, comment=comment)  #:create MNE EvokedArray
    return evoked_array


def _create_info(ft_struct):
    """private function which creates an MNE info structure from a preloaded FieldTrip file"""

    ch_names = list(ft_struct["label"])
    sfreq = _set_sfreq(ft_struct)
    ch_types = _set_ch_types(ft_struct)
    montage = _create_montage(ft_struct)

    info = mne.create_info(ch_names, sfreq, ch_types, montage)
    return info


def _convert_ch_types(ch_type_array):
    """private function which converts the channel type names from filedtrip style (e.g. megmag)
    to mne style (e.g. mag)"""
    for index, name in enumerate(ch_type_array):
        if name in ("megplanar", "meggrad"):
            ch_type_array[index] = "grad"
        elif name == "megmag":
            ch_type_array[index] = "mag"
        elif name == "eeg":
            ch_type_array[index] = "eeg"
        elif name in ("refmag", "refgrad"):  # is it correct to put both of them here??
            ch_type_array[index] = "ref_meg"
        elif name in ("unknown", "clock"):
            ch_type_array[index] = "misc"
        elif name in ("analog trigger", "digital trigger", "trigger"):
            ch_type_array[index] = "stim"
        elif name.startswith('MEG'):
            if name.endswith(('2', '3')):
                ch_type_array[index] = 'grad'
            elif name.endswith('1'):
                ch_type_array[index] = 'mag'
            else:
                raise ValueError('Unknown MEG channel')
        elif name.startswith('EEG'):
            ch_type_array[index] = 'eeg'
        else:
            ch_type_array[index] = 'misc'
    return ch_type_array


def _set_ch_types(ft_struct):
    """private function which finds the channel types for every channel"""
    if 'hdr' in ft_struct and 'chantype' in ft_struct['hdr']:
        available_channels = numpy.where(numpy.in1d(ft_struct["hdr"]['label'], ft_struct['label']))
        ch_types = _convert_ch_types(ft_struct["hdr"]["chantype"][available_channels])
    elif 'grad' in ft_struct and 'chantype' in ft_struct['grad']:
        available_channels = numpy.where(numpy.in1d(ft_struct["grad"]['label'], ft_struct['label']))
        ch_types = _convert_ch_types(ft_struct["grad"]["chantype"][available_channels])
    elif 'elec' in ft_struct and 'chantype' in ft_struct['grad']:
        available_channels = numpy.where(numpy.in1d(ft_struct["elec"]['label'], ft_struct['label']))
        ch_types = _convert_ch_types(ft_struct["elec"]["chantype"][available_channels])
    elif 'label' in ft_struct:
        ch_types = _convert_ch_types(ft_struct['label'])
    else:
        raise ValueError('Cannot find channel types')

    return ch_types


def _create_montage(ft_struct):
    """private function which creates a montage from the FieldTrip data"""

    # try to create a montage
    montage_pos, montage_ch_names = list(), list()

    # see if there is a grad field in the structure
    try:
        available_channels = numpy.where(numpy.in1d(ft_struct["grad"]['label'], ft_struct['label']))
        montage_ch_names.extend(ft_struct['grad']['label'][available_channels])
        montage_pos.extend(ft_struct['grad']['chanpos'][available_channels])
    except KeyError:
        pass

    # see if there is a elec field in the structure
    try:
        available_channels = numpy.where(numpy.in1d(ft_struct["elec"]['label'], ft_struct['label']))
        montage_ch_names.extend(ft_struct['elec']['label'][available_channels])
        montage_pos.extend(ft_struct['elec']['chanpos'][available_channels])
    except KeyError:
        pass

    montage = None

    if len(montage_ch_names) > 0 and len(montage_pos) > 0 and len(montage_ch_names) == len(montage_pos):
        montage = mne.channels.DigMontage(dig_ch_pos=dict(zip(montage_ch_names, montage_pos)))
    return montage


def _set_sfreq(ft_struct):
    """private function which sets the sample frequency"""
    try:
        sfreq = ft_struct["fsample"]
    except KeyError:
        try:
            t1 = ft_struct["time"][0]
            t2 = ft_struct["time"][1]
            difference = abs(t1 - t2)
            sfreq = 1 / difference
        except KeyError:
            raise ValueError("No Source for sfreq found")
    return sfreq


def _set_tmin(ft_struct):
    """private function which sets the start time before the event in evoked data if possible"""
    times = ft_struct["time"]
    time_check = all(times[i][0] == times[i - 1][0] for i, x in enumerate(times))
    if time_check == True:
        tmin = times[0][0]
    else:
        tmin = None
    return tmin


def _create_events(ft_struct, trialinfo_map, omit_trialinfo_index, omit_non_unique_trialinfo_index):
    """private function which creates an event matrix from the FieldTrip structure"""

    # sanitize trialinfo_map
    if trialinfo_map:
        for (key, value) in trialinfo_map.items():
            trialinfo_map[key] = numpy.array(value)

    event_type = ft_struct["trialinfo"]
    event_number = range(len(event_type))
    # event_trans_val: This is only a dummy row of zeros, used until a way to implement this is found
    event_trans_val = numpy.zeros(len(event_type))
    if omit_trialinfo_index:
        index_columns = numpy.all(numpy.diff(event_type, axis=0) == 1, axis=0)
        event_type = event_type[:, numpy.logical_not(index_columns)]

    if omit_non_unique_trialinfo_index:
        unique_columns = numpy.any(numpy.diff(numpy.sort(event_type, axis=0), axis=0) == 0, axis=0)
        event_type = event_type[:, unique_columns]

    (unique_events, unique_events_index) = numpy.unique(event_type, axis=0, return_inverse=True)

    if event_type.ndim == 1:
        final_event_types = event_type
    else:
        final_event_types = unique_events_index + 1

    events = numpy.vstack([numpy.array(event_number), event_trans_val, final_event_types]).astype('int').T

    event_id = dict()
    if not trialinfo_map:
        trialinfo_map = dict()
        for cur_unique_event in unique_events:
            trialinfo_map['trialinfo {}'.format(cur_unique_event)] = cur_unique_event

    for (cur_event_id, cur_trialinfo_mat) in trialinfo_map.items():
        event_index = numpy.where(numpy.all(unique_events == cur_trialinfo_mat, axis=1))[0][0] + 1
        event_id[cur_event_id] = event_index

    return (events, event_id)
