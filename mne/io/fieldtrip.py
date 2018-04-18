# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Thomas Hartmann & Dirk Gütlin
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import mne
import numpy as np
from ..externals import pymatreader


def read_raw_fieldtrip(ft_structure_path, data_name='data'):
    """Load continuous (i.e. raw) data from a FieldTrip preprocessing structure.

    This function expects to find single trial raw data (FT_DATATYPE_RAW) in
    the structure data_name is pointing at.

    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    data_name: str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.

    Returns
    -------
    mne.io.RawArray
        A MNE RawArray structure consisting of the raw array and measurement
        info

    """
    ft_struct = pymatreader.read_mat(ft_structure_path,
                                     ignore_fields=['previous'],
                                     variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    data = np.array(ft_struct['trial'])  # create the main data array
    info = _create_info(ft_struct)  # create info structure

    custom_raw = mne.io.RawArray(data, info)  # create an MNE RawArray
    return custom_raw


def read_epochs_fieldtrip(ft_structure_path, data_name='data',
                         trialinfo_map=None,
                   omit_trialinfo_index=True,
                   omit_non_unique_trialinfo_index=True):
    """Load epoched data from a FieldTrip preprocessing structure.

    This function expects to find epoched data in the structure data_name is
    pointing at.

    .. warning:: Only epochs with the same amount of channels and samples are
                 supported!

    The data is read as it is. Events however are represented entirely
    different in FieldTrip compared to MNE.

    In FieldTrip, each epoch corresponds to one row in the trialinfo field.
    This field can have one or more columns. The function first removes
    columns according to the two omit parameters.

    - If only one column remains, its values are used as event values in the
      MNE Epoch.
    - If two or more columns remain, each unique combination of
      these values receives a new event value. These event values are created
      automatically. In order to match these to conditions, you can use the
      trialinfo_map parameter.

    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    data_name: str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.
    trialinfo_map: dict
        A dictionary mapping condition strings (MNE's event_ids) to the
        trialinfo column. The values should be 1D numpy arrays. See examples
        for details.
    omit_trialinfo_index: bool
        Omit trialinfo columns that look like an index of the trials, i.e. in
        which every row is the row before + 1.
    omit_non_unique_trialinfo_index: bool
        Omit trialinfo columns that contain a different value for each row. T
        ese are most likely additional data like reaction times that cannot
        be represented in MNE.


    Returns
    -------
    mne.EpochsArray
        A MNE EpochsArray structure consisting of the epochs arrays, an event
        matrix, start time before event (if possible, else defaults to 0) and
        measurement info.

    Examples
    -------
    >>> read_epochs_fieldtrip('FieldTripEpochsFile', # doctest: +SKIP
    >>>     trialinfo_map={ # doctest: +SKIP
    >>>         'audio/attend': np.array([0, 1]), # doctest: +SKIP
    >>>         'visual/attend': np.array([1, 1]), # doctest: +SKIP
    >>>         'audio/non_attend': np.array([0, 0]), # doctest: +SKIP
    >>>         'visual/non_attend': np.array([1, 0])}) # doctest: +SKIP
    """
    ft_struct = pymatreader.read_mat(ft_structure_path,
                                     ignore_fields=['previous'],
                                     variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    data = np.array(ft_struct['trial'])  # create the epochs data array
    (events, event_id) = _create_events(ft_struct, trialinfo_map,
                                        omit_trialinfo_index,
                                        omit_non_unique_trialinfo_index)
    tmin = _set_tmin(ft_struct)  # create start time
    info = _create_info(ft_struct)  # create info structure

    custom_epochs = mne.EpochsArray(data=data, info=info, tmin=tmin,
                                    events=events, event_id=event_id)
    return custom_epochs


def read_evoked_fieldtrip(ft_structure_path, comment='', data_name='data'):
    """Load evoked data from a FieldTrip timelocked structure.

    This function expects to find timelocked data in the structure data_name is
    pointing at.

    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    comment: str
        Comment on dataset. Can be the condition.
    data_name: str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.

    Returns
    -------
    mne.EvokedArray
        A MNE EvokedArray structure consisting of the averaged data array,
         comment and measurement info.

    """
    ft_struct = pymatreader.read_mat(ft_structure_path,
                                     ignore_fields=['previous'],
                                     variable_names=[data_name])
    ft_struct = ft_struct[data_name]

    data_evoked = ft_struct['avg']  # create evoked data
    info = _create_info(ft_struct)  # create info structure

    evoked_array = mne.EvokedArray(data_evoked, info, comment=comment)
    return evoked_array


def _create_info(ft_struct):
    """Create MNE info structure from a FieldTrip structure."""
    ch_names = list(ft_struct['label'])
    sfreq = _set_sfreq(ft_struct)
    ch_types = _set_ch_types(ft_struct)
    montage = _create_montage(ft_struct)

    info = mne.create_info(ch_names, sfreq, ch_types, montage)
    return info


def _convert_ch_types(ch_type_array):
    """Convert channel type names from FieldTrip style to mne style."""
    for index, name in enumerate(ch_type_array):
        if name in ('megplanar', 'meggrad'):
            ch_type_array[index] = 'grad'
        elif name == 'megmag':
            ch_type_array[index] = 'mag'
        elif name == 'eeg':
            ch_type_array[index] = 'eeg'
        elif name in ('refmag', 'refgrad'):
            ch_type_array[index] = 'ref_meg'
        elif name in ('unknown', 'clock'):
            ch_type_array[index] = 'misc'
        elif name in ('analog trigger', 'digital trigger', 'trigger'):
            ch_type_array[index] = 'stim'
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
    """Find the channel types for every channel."""
    if 'hdr' in ft_struct and 'chantype' in ft_struct['hdr']:
        available_channels = np.where(np.in1d(ft_struct['hdr']['label'],
                                              ft_struct['label']))
        ch_types = _convert_ch_types(
            ft_struct['hdr']['chantype'][available_channels])
    elif 'grad' in ft_struct and 'chantype' in ft_struct['grad']:
        available_channels = np.where(np.in1d(ft_struct['grad']['label'],
                                              ft_struct['label']))
        ch_types = _convert_ch_types(
            ft_struct['grad']['chantype'][available_channels])
    elif 'elec' in ft_struct and 'chantype' in ft_struct['grad']:
        available_channels = np.where(np.in1d(ft_struct['elec']['label'],
                                              ft_struct['label']))
        ch_types = _convert_ch_types(
            ft_struct['elec']['chantype'][available_channels])
    elif 'label' in ft_struct:
        ch_types = _convert_ch_types(ft_struct['label'])
    else:
        raise ValueError('Cannot find channel types')

    return ch_types


def _create_montage(ft_struct):
    """Create a montage from the FieldTrip data."""
    # try to create a montage
    montage_pos, montage_ch_names = list(), list()

    # see if there is a grad field in the structure
    try:
        available_channels = np.where(np.in1d(ft_struct['grad']['label'],
                                              ft_struct['label']))
        montage_ch_names.extend(ft_struct['grad']['label'][available_channels])
        montage_pos.extend(ft_struct['grad']['chanpos'][available_channels])
    except KeyError:
        pass

    # see if there is a elec field in the structure
    try:
        available_channels = np.where(np.in1d(ft_struct['elec']['label'],
                                              ft_struct['label']))
        montage_ch_names.extend(ft_struct['elec']['label'][available_channels])
        montage_pos.extend(ft_struct['elec']['chanpos'][available_channels])
    except KeyError:
        pass

    montage = None

    if (len(montage_ch_names) > 0 and len(montage_pos) > 0 and
            len(montage_ch_names) == len(montage_pos)):
        montage = mne.channels.DigMontage(
            dig_ch_pos=dict(zip(montage_ch_names, montage_pos)))
    return montage


def _set_sfreq(ft_struct):
    """Set the sample frequency."""
    try:
        sfreq = ft_struct['fsample']
    except KeyError:
        try:
            t1 = ft_struct['time'][0]
            t2 = ft_struct['time'][1]
            difference = abs(t1 - t2)
            sfreq = 1 / difference
        except KeyError:
            raise ValueError('No Source for sfreq found')
    return sfreq


def _set_tmin(ft_struct):
    """Set the start time before the event in evoked data if possible."""
    times = ft_struct['time']
    time_check = all(times[i][0] == times[i - 1][0]
                     for i, x in enumerate(times))
    if time_check:
        tmin = times[0][0]
    else:
        tmin = None
    return tmin


def _create_events(ft_struct, trialinfo_map, omit_trialinfo_index,
                   omit_non_unique_trialinfo_index):
    """Create an event matrix from the FieldTrip structure."""
    # sanitize trialinfo_map
    if trialinfo_map:
        for (key, value) in trialinfo_map.items():
            trialinfo_map[key] = np.array(value)

    event_type = ft_struct['trialinfo']
    event_number = range(len(event_type))

    event_trans_val = np.zeros(len(event_type))
    if omit_trialinfo_index:
        index_columns = np.all(np.diff(event_type, axis=0) == 1, axis=0)
        event_type = event_type[:, np.logical_not(index_columns)]

    if omit_non_unique_trialinfo_index:
        unique_columns = np.any(
            np.diff(np.sort(event_type, axis=0), axis=0) == 0, axis=0)
        event_type = event_type[:, unique_columns]

    (unique_events, unique_events_index) = np.unique(event_type, axis=0,
                                                     return_inverse=True)

    if event_type.ndim == 1:
        final_event_types = event_type
    else:
        final_event_types = unique_events_index + 1

    events = np.vstack([np.array(event_number), event_trans_val,
                        final_event_types]).astype('int').T

    event_id = dict()
    if not trialinfo_map:
        trialinfo_map = dict()
        for cur_unique_event in unique_events:
            trialinfo_map['trialinfo {}'.format(
                cur_unique_event)] = cur_unique_event

    for (cur_event_id, cur_trialinfo_mat) in trialinfo_map.items():
        event_index = np.where(
            np.all(unique_events == cur_trialinfo_mat, axis=1))[0][0] + 1
        event_id[cur_event_id] = event_index

    return events, event_id
