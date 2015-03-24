"""
=========================================
Creating MNE objects from data arrays
=========================================

In this simple example, the creation of MNE objects from 
numpy arrays is demonstrated.
"""
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import mne
from mne.io import RawArray

import numpy as np
import matplotlib.pyplot as plt

sfreq = 1000                    # Sampling frequency
times = np.arange(0, 10, 0.001) # Use 10000 samples (10s)

sin = np.sin(times) 
cos = np.cos(times)
sinX2 = sin * 2
cosX2 = cos * 2

data = np.array([sin, cos, sinX2, cosX2]) # Matrix of size 4 X 10000 

ch_names = ['sin', 'cos', 'sinX2', 'cosX2']

# Creating of info dictionary.
# It is also possible to use info from another raw object.
info = mne.create_info(ch_names=ch_names, sfreq=sfreq) 

raw = RawArray(data, info)

# Scaling of the figure. 
# For actual EEG/MEG data a different scaling factor should be used.
scalings = {'misc':2} 

raw.plot(scalings=scalings)
plt.show()


#######################################################################
# EpochsArray

event_id = 1
events = np.array([[200, 0, event_id],
                   [1200, 0, event_id]]) # List of two arbitrary events
                   
# Here a data set of 700 ms epochs from 2 channels is 
# created from sin and cos data.
# Any data in shape (n_epochs, n_channels, n_times) can be used.
epochs_data = [[sin[:700], sin[1000:1700]],
               [cos[:700], cos[1000:1700]]] 
               
info = mne.create_info(ch_names=['sin', 'cos'], sfreq=sfreq)

epochs = mne.EpochsArray(epochs_data, info=info, events=events, 
                         event_id={'arbitrary':1})

picks = mne.pick_types(info, meg=False, eeg=False, misc=True)
epochs.plot(picks=picks)
plt.show()


#######################################################################
# EvokedArray

evoked_data = np.mean(epochs_data, axis=0)
nave = len(epochs_data[0]) # Number of averaged epochs

evokeds = mne.EvokedArray(evoked_data, info=info, tmin= -0.2, 
                          comment='Arbitrary', nave=nave)
evokeds.plot(picks=picks)