# -*- coding: UTF-8 -*-
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
from ..utils import warn
from .constants import FIFF
from ..externals.pymatreader.pymatreader import read_mat
from .. import transforms

_unit_dict = {'m': 1,
              'cm': 1e-2,
              'mm': 1e-3,
              'V': 1,
              'mV': 1e-3,
              'uV': 1e-6,
              'T': 1,
              'T/m': 1,
              'T/cm': 1e2}

_supported_megs = ['neuromag306']


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
    ft_struct = read_mat(ft_structure_path,
                         ignore_fields=['previous'],
                         variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    data = np.array(ft_struct['trial'])  # create the main data array
    info = _create_info(ft_struct)  # create info structure

    if data.ndim > 2:
        data = np.squeeze(data)

    if data.ndim == 1:
        data = data[np.newaxis, ...]

    if data.ndim != 2:
        raise RuntimeError('The data you are trying to load does not seem to'
                           'be raw data')

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
    ft_struct = read_mat(ft_structure_path,
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
    ft_struct = read_mat(ft_structure_path,
                         ignore_fields=['previous'],
                         variable_names=[data_name])
    ft_struct = ft_struct[data_name]

    data_evoked = ft_struct['avg']  # create evoked data
    info = _create_info(ft_struct)  # create info structure

    evoked_array = EvokedArray(data_evoked, info, comment=comment)
    return evoked_array


def _create_info(ft_struct):
    """Create MNE info structure from a FieldTrip structure."""
    sfreq = _set_sfreq(ft_struct)
    montage = _create_montage(ft_struct)
    chs = _create_info_chs(ft_struct)
    ch_names = [ch['ch_name'] for ch in chs]

    info = create_info(ch_names, sfreq, montage=montage)
    info['chs'] = chs
    info._update_redundant()

    return info


def _create_info_chs(ft_struct):
    """Create the chs info field from the FieldTrip structure."""
    all_channels = ft_struct['label']
    ch_defaults = dict(coord_frame=FIFF.FIFFV_COORD_HEAD,
                       cal=1.0,
                       range=1.0,
                       unit_mul=FIFF.FIFF_UNITM_NONE,
                       loc=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
                       unit=FIFF.FIFF_UNIT_V)
    try:
        elec = ft_struct['elec']
    except KeyError:
        elec = None

    try:
        grad = ft_struct['grad']
    except KeyError:
        grad = None

    if elec is None and grad is None:
        warn('The supplied FieldTrip structure does not have an elec or grad '
             'field. No channel locations will extracted and the kind of '
             'channel might be inaccurate.')

    if grad['type'] not in _supported_megs:
        warn('Unsupported MEG type %s. Values for the kind of coils '
             'are guessed with best effort. Please verify those. '
             'Please also verify whether the channel locations '
             'and orientations match!' % (grad['type'],))

    chs = list()
    for idx_chan, cur_channel_label in enumerate(all_channels):
        cur_ch = ch_defaults.copy()
        cur_ch['ch_name'] = cur_channel_label
        cur_ch['logno'] = idx_chan + 1
        cur_ch['scanno'] = idx_chan + 1
        if elec and cur_channel_label in elec['label']:
            cur_ch = _process_channel_eeg(cur_ch, elec)

        elif grad and cur_channel_label in grad['label']:
            cur_ch = _process_channel_meg(cur_ch, grad)
        else:
            if cur_channel_label.startswith('EOG'):
                cur_ch['kind'] = FIFF.FIFFV_EOG_CH
                cur_ch['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
            elif cur_channel_label.startswith('ECG'):
                cur_ch['kind'] = FIFF.FIFFV_ECG_CH
                cur_ch['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR
            else:
                warn('Cannot guess the correct type of channel %s. Making '
                     'it a MISC channel.' % (cur_channel_label,))
                cur_ch['kind'] = FIFF.FIFFV_MISC_CH
                cur_ch['coil_type'] = FIFF.FIFFV_COIL_NONE

        chs.append(cur_ch)

    return chs


def _create_montage(ft_struct):
    """Create a montage from the FieldTrip data."""
    # try to create a montage
    montage_pos, montage_ch_names = list(), list()

    for cur_ch_type in ('grad', 'elec'):
        if cur_ch_type in ft_struct:
            cur_ch_struct = ft_struct[cur_ch_type]
            available_channels = np.where(np.in1d(cur_ch_struct['label'],
                                                  ft_struct['label']))[0]
            cur_labels = np.asanyarray(cur_ch_struct['label'])
            montage_ch_names.extend(
                cur_labels[available_channels])
            montage_pos.extend(
                cur_ch_struct['chanpos'][available_channels])

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

    if trialinfo_column > (event_type.shape[0] + 1):
        raise ValueError('trialinfo_column is higher than the amount of'
                         'columns in trialinfo.')

    event_trans_val = np.zeros(len(event_type))

    final_event_types = event_type[:, trialinfo_column]

    events = np.vstack([np.array(event_number), event_trans_val,
                        final_event_types]).astype('int').T

    return events


def _process_channel_eeg(cur_ch, elec):
    """Convert EEG channel from FieldTrip to MNE.

    Parameters
    ----------
    cur_ch: dict
        Channel specific dictionary to populate.

    elec: dict
        elec dict as loaded from the FieldTrip structure

    Returns
    -------
    dict: The original dict (cur_ch) with the added information
    """
    all_labels = np.asanyarray(elec['label'])
    chan_idx_in_elec = np.where(all_labels == cur_ch['ch_name'])[0][0]
    position = np.squeeze(elec['chanpos'][chan_idx_in_elec, :])
    chantype = elec['chantype'][chan_idx_in_elec]
    chanunit = elec['chanunit'][chan_idx_in_elec]
    position_unit = elec['unit']

    if chantype != 'eeg':
        raise RuntimeError('The current channel is contained in '
                           'the "elec" sub-structure. This should '
                           'be an EEG channel. However, the channel '
                           'type is: %s.' % (chantype,))

    position = position * _unit_dict[position_unit]
    cur_ch['loc'] = np.hstack((position, np.zeros((9,))))
    cur_ch['unit'] = FIFF.FIFF_UNIT_V
    cur_ch['unit_mul'] = np.log10(_unit_dict[chanunit[0]])
    cur_ch['kind'] = FIFF.FIFFV_EEG_CH
    cur_ch['coil_type'] = FIFF.FIFFV_COIL_EEG

    return cur_ch


def _process_channel_meg(cur_ch, grad):
    """Convert MEG channel from FieldTrip to MNE.

    Parameters
    ----------
    cur_ch: dict
        Channel specific dictionary to populate.

    grad: dict
        grad dict as loaded from the FieldTrip structure

    Returns
    -------
    dict: The original dict (cur_ch) with the added information
    """
    all_labels = np.asanyarray(grad['label'])
    chan_idx_in_grad = np.where(all_labels == cur_ch['ch_name'])[0][0]
    position = np.squeeze(grad['chanpos'][chan_idx_in_grad, :])
    orientation = transforms.rotation3d(
        *np.squeeze(grad['chanori'][chan_idx_in_grad, :]).tolist())
    orientation = orientation.flatten()
    position_unit = grad['unit']
    chantype = grad['chantype'][chan_idx_in_grad]
    chanunit = grad['chanunit'][chan_idx_in_grad]
    position = position * _unit_dict[position_unit]
    gradtype = grad['type']

    cur_ch['loc'] = np.hstack((position, orientation))
    cur_ch['kind'] = FIFF.FIFFV_MEG_CH
    if gradtype == 'neuromag306':
        if chantype == 'megmag':
            cur_ch['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T1
            cur_ch['unit'] = FIFF.FIFF_UNIT_T
        elif chantype == 'megplanar':
            cur_ch['coil_type'] = FIFF.FIFFV_COIL_VV_PLANAR_T1
            cur_ch['unit'] = FIFF.FIFF_UNIT_T_M
        else:
            raise RuntimeError('Unexpected coil type: %s.' % (
                chantype,))
    else:
        raise NotImplemented('This needs to be implemented!')

    cur_ch['unit_mul'] = np.log10(_unit_dict[chanunit[0]])

    return cur_ch
