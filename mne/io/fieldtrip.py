# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Thomas Hartmann & Dirk Gütlin
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import numpy as np

from . import RawArray
from ..epochs import EpochsArray
from ..evoked import EvokedArray
from .meas_info import create_info
from ..channels import DigMontage


def _check_pymatreader():
    """Check if pymatreader and h5py are installed.

    Returns the pymatreader module on success.
    """
    try:
        from ..externals import pymatreader
    except ImportError:
        raise ImportError('The h5py module is required to use the FieldTrip '
                          'import functions.')
    return pymatreader


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
    pymatreader = _check_pymatreader()

    ft_struct = pymatreader.read_mat(ft_structure_path,
                                     ignore_fields=['previous'],
                                     variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    data = np.array(ft_struct['trial'])  # create the main data array
    info = _create_info(ft_struct)  # create info structure

    custom_raw = RawArray(data, info)  # create an MNE RawArray
    return custom_raw


def read_epochs_fieldtrip(ft_structure_path, data_name='data',
                          trialinfo_column=0):
    """Load epoched data from a FieldTrip preprocessing structure.

    This function expects to find epoched data in the structure data_name is
    pointing at.

    .. warning:: Only epochs with the same amount of channels and samples are
                 supported!

    Parameters
    ----------
    ft_structure_path: str
        Path and filename of the .mat file containing the data.
    data_name: str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.
    trialinfo_column: int
        Column of the trialinfo matrix to use for the event codes


    Returns
    -------
    mne.EpochsArray
        A MNE EpochsArray structure consisting of the epochs arrays, an event
        matrix, start time before event (if possible, else defaults to 0) and
        measurement info.


    """
    pymatreader = _check_pymatreader()

    ft_struct = pymatreader.read_mat(ft_structure_path,
                                     ignore_fields=['previous'],
                                     variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    data = np.array(ft_struct['trial'])  # create the epochs data array
    events = _create_events(ft_struct, trialinfo_column)
    tmin = _set_tmin(ft_struct)  # create start time
    info = _create_info(ft_struct)  # create info structure

    custom_epochs = EpochsArray(data=data, info=info, tmin=tmin,
                                events=events)
    return custom_epochs


def read_evoked_fieldtrip(ft_structure_path, comment=None, data_name='data'):
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
    pymatreader = _check_pymatreader()

    ft_struct = pymatreader.read_mat(ft_structure_path,
                                     ignore_fields=['previous'],
                                     variable_names=[data_name])
    ft_struct = ft_struct[data_name]

    data_evoked = ft_struct['avg']  # create evoked data
    info = _create_info(ft_struct)  # create info structure

    evoked_array = EvokedArray(data_evoked, info, comment=comment)
    return evoked_array


def _create_info(ft_struct):
    """Create MNE info structure from a FieldTrip structure."""
    ch_names = list(ft_struct['label'])
    sfreq = _set_sfreq(ft_struct)
    ch_types = _set_ch_types(ft_struct)
    montage = _create_montage(ft_struct)

    info = create_info(ch_names, sfreq, ch_types, montage)
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
        montage = DigMontage(
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


def _create_events(ft_struct, trialinfo_column):
    """Create an event matrix from the FieldTrip structure."""

    event_type = ft_struct['trialinfo']
    event_number = range(len(event_type))

    if trialinfo_column < 0:
        raise ValueError('trialinfo_column must be positive')

    if trialinfo_column > (event_type.shape[0]+1):
        raise ValueError('trialinfo_column is higher than the amount of'
                         'columns in trialinfo.')

    (unique_events, unique_events_index) = np.unique(event_type, axis=0,
                                                     return_inverse=True)

    event_trans_val = np.zeros(len(event_type))

    final_event_types = event_type[:, trialinfo_column]

    events = np.vstack([np.array(event_number), event_trans_val,
                        final_event_types]).astype('int').T

    return events
