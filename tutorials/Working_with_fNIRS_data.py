"""
.. _tut_fnirs_data

Use MNE with fNIRS data
=======================
# Author: Matteo Caffini  
# Contact: <matteo.caffini@unitn.it>

"""
import numpy as np

import mne

raw_ndarray = np.load('/home/matteo/raw_ndarray.npy')

###############################################################################
# MNE Python was implemented with EEG and MEG in mind, but can be useful to
# process fNIRS data as #well. A common fNIRS data set (similarly to EEG and
# MEG data) contains data recorded over time(a.k.a. samples) at many locations
# on the scalp (a.k.a. channels).
# Usually in fNIRS one looks for brain responses to external sensorial
# stimulations (visual, acoustic, etc.). In order to do so, block design
# studies are usually employed and data are later epoched to get evoked
# responses.

###############################################################################
# fNIRS raw data come in various formats and the best way to dive into MNE is
# to deal with your own format first and fit your data into a 2D numpy array
# [channels x samples]. Once you prepared your time series for HbO and HbR
# concentrations, you can create a RawArray instance from the data array using
# mne.io.RawArray. If you have a time series of events synchronous with your
# data you can append this as a last row to your array.
# 
# In this tutorial, I don't preprocess the data. If your fNIRS device outputs
# data in the form of photon fluences at detectors, you can compute HbO and HbR
# concentrations using the modified Lambert-Beer model.
# 
# In the following example we have data from 20 channels plus one events
# channel and we end up # with 20 HbO time series, 20 HbR time series and one
# events time series.

###############################################################################
# Often fNIRS data contain time series for oxyhemoglobin (HbO) and
# deoxyhemoglobin (HbR) concentrations and you want to process them both. Using
# the *channel types* field in a mne.Info class instance, you can assign either
#'hbo' or 'hbr' channel type to your channel.
nChannels = 20 # number of physical channels
sampling_frequency = 15.625
channel_names_fnirs = ['HbO '+"%.2d" % i for i in range(1,nChannels+1)] +
    ['HbR '+"%.2d" % i for i in range(1,nChannels+1)]
channel_names = channel_names_fnirs + ['Events']
channel_types = ['hbo' for i in range(nChannels)] +
    ['hbr' for i in range(nChannels)] + ['stim']
info = mne.create_info(ch_names=channel_names, sfreq=sampling_frequency,
    ch_types=channel_types, montage=None)
info['lowpass'] = 0.0
info['highpass'] = 0.0

###############################################################################
# Import fNIRS data from numpy array using mne.io.RawArray.
raw_data = mne.io.RawArray(data=raw_ndarray, info=info, first_samp=0,
    verbose=None)

###############################################################################
# In our data array the last row contains event data, coded as different
# in a time series synchronous with the fNIRS recording. Let's find such events
# using mne.find_events and plot the events timeline with mne.viz.plot_events.
events = mne.find_events(raw_data, stim_channel='Events', shortest_event=1)
event_id = {'ISI':1, 'Condition1': 2, 'Condition2': 4, 'Condition3': 8,
    'Pause':16}
color = {1:'black', 2:'green', 4:'magenta', 8:'cyan', 16:'yellow'}
mne.viz.plot_events(events, raw_data.info['sfreq'], raw_data.first_samp,
    color=color, event_id=event_id)

###############################################################################
# Sometimes we want to filter out unwanted frequencies from time series. For
# example, here we bandpass filter our data between 0.01 and 2 Hz.
l_freq = 0.01 # high-pass filter cutoff ( __/¯¯¯ )
h_freq = 2    #  low-pass filter cutoff ( ¯¯¯\__ )
raw_data.filter(l_freq, h_freq)

###############################################################################
# In order to have a general glimpse of the dataset, I start by plotting all
# channels into a single plot window. I choose to color-code HbO, HbR and
# events data using red, blue and black respectively. I also mark color-coded
# events positions, using the same colors previously used in
# mne.viz.plot_events.
scalings = dict(hbo=10e-6, hbr=10e-6, stim=1)
fig_title = 'fNIRS Raw Bandpass filtered [' + str(l_freq) + ' Hz, '
    + str(h_freq) + ' Hz]'
plot_colors = dict(hbo='r', hbr='b', stim='k')
raw_data.plot(title=fig_title, events=events, start=0.0, color=plot_colors,
    event_color=color, duration=np.max(raw_data.times), scalings=scalings,
    order='original', n_channels=len(channel_names), remove_dc=False,
    highpass=None, lowpass=None)

###############################################################################
# I can later dive into the dataset and look at the raw time series more
# closely. Same color-codes through the plots is usually a good idea.
scalings = dict(hbo=10e-6, hbr=10e-6, stim=1)
fig_title = 'fNIRS Raw Bandpass filtered [' + str(l_freq) + ' Hz, '
    + str(h_freq) + ' Hz]'
plot_colors = dict(hbo='r', hbr='b', stim='k')
fig = raw_data.plot(title=fig_title, events=events, start=0.0,
    color=plot_colors, event_color=color, scalings=scalings, order='original',
    duration=36, remove_dc=False, highpass=None, lowpass=None)

###############################################################################
# We can now select epochs (defined as windows [-2,10] s around events) and
# drop unwanted ones (in this case I drop epochs with peak-to-peak amplitude
# larger than 1.5e-5 uM). Finally, we display the epochs time series.
tmin = -2
tmax = 10
reject = dict(hbo=1.5e-5, hbr=1.5e-5)
epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, proj=True,
    baseline=(None, 0), preload=True, reject=reject)    
fig_title = 'fNIRS Epochs'
epochs.plot(title=fig_title)

###############################################################################
# Finally, we calculate and display the evoked signals, for example the one
# corresponding to the first condition in the paradigm.
evoked_1 = epochs['Condition1'].average()
evoked_1.plot()
