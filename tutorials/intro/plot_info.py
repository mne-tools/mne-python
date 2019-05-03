"""
.. _tut-info-class:

The :class:`~mne.Info` data structure
=====================================

The :class:`~mne.Info` data object is typically created
when data is imported into MNE-Python and contains details such as:

- date, subject information, and other recording details
- the sampling rate
- information about the data channels (name, type, position, etc.)
- digitized points
- sensor–head coordinate transformation matrices

and so forth. See the :class:`the API reference <mne.Info>`
for a complete list of all data fields. Once created, this object is passed
around throughout the data analysis pipeline.
"""

import mne
import os.path as op

###############################################################################
# :class:`mne.Info` behaves as a nested Python dictionary:

# Read the info object from an example recording
info = mne.io.read_info(
    op.join(mne.datasets.sample.data_path(), 'MEG', 'sample',
            'sample_audvis_raw.fif'), verbose=False)

###############################################################################
# List all the fields in the info object
print('Keys in info dictionary:\n', info.keys())

###############################################################################
# Obtain the sampling rate of the data
print(info['sfreq'], 'Hz')

###############################################################################
# List all information about the first data channel
print(info['chs'][0])

###############################################################################
# .. _picking_channels:
#
# Obtaining subsets of channels
# -----------------------------
#
# There are a number of convenience functions to obtain channel indices, given
# an :class:`mne.Info` object.

###############################################################################
# Get channel indices by name
channel_indices_two = mne.pick_channels(
    info['ch_names'], ['MEG 0312', 'EEG 005'])

###############################################################################
# Get channel indices by regular expression
channel_indices_meg_re = mne.pick_channels_regexp(info['ch_names'], 'MEG *')

###############################################################################
# Channel types
# -------------
#
# MNE supports different channel types:
#
# - eeg : For EEG channels with data stored in Volts (V)
# - meg (mag) : For MEG magnetometers channels stored in Tesla (T)
# - meg (grad) : For MEG gradiometers channels stored in Tesla/Meter (T/m)
# - ecg : For ECG channels stored in Volts (V)
# - seeg : For Stereotactic EEG channels in Volts (V).
# - ecog : For Electrocorticography (ECoG) channels in Volts (V).
# - fnirs (HBO) : Functional near-infrared spectroscopy oxyhemoglobin data.
# - fnirs (HBR) : Functional near-infrared spectroscopy deoxyhemoglobin data.
# - emg : For EMG channels stored in Volts (V)
# - bio : For biological channels (AU).
# - stim : For the stimulus (a.k.a. trigger) channels (AU)
# - resp : For the response-trigger channel (AU)
# - chpi : For HPI coil channels (T).
# - exci : Flux excitation channel used to be a stimulus channel.
# - ias : For Internal Active Shielding data (maybe on Triux only).
# - syst : System status channel information (on Triux systems only).
#
# Get channel indices by type
channel_indices_meg = mne.pick_types(info, meg=True)  # MEG only
channel_indices_eeg = mne.pick_types(info, meg=False, eeg=True)  # EEG only

###############################################################################
# MEG gradiometers and EEG channels
channel_indices_grad_eeg = mne.pick_types(info, meg='grad', eeg=True)

###############################################################################
# Get a dictionary of channel indices, grouped by channel type
channel_indices_by_type = mne.io.pick.channel_indices_by_type(info)
print('The first three magnetometers:', channel_indices_by_type['mag'][:3])

###############################################################################
# Obtaining information about channels
# ------------------------------------

# Channel type of a specific channel
channel_type = mne.io.pick.channel_type(info, 75)
print('Channel #75 is of type:', channel_type)

###############################################################################
# Channel types of a collection of channels
meg_channels = mne.pick_types(info, meg=True)[:10]
channel_types = [mne.io.pick.channel_type(info, ch) for ch in meg_channels]
print('First 10 MEG channels are of type:\n', channel_types)

###############################################################################
# Dropping channels from an info structure
# ----------------------------------------
#
# It is possible to limit the info structure to only include a subset of
# channels with the :func:`mne.pick_info` function:

# Only keep EEG channels
eeg_indices = mne.pick_types(info, meg=False, eeg=True)
reduced_info = mne.pick_info(info, eeg_indices)

print(reduced_info)
