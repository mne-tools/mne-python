"""
.. _tut-make-fixed-length-epochs:

Creating epochs of equal length
========================

This tutorial shows how to create equal length epochs.

First, we import necessary modules and read in a sample raw
data set.
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file)

# Create the epochs

epochs = mne.make_fixed_length_epochs()


# Visualize the results and compare to other types of epoched data

timeseries_plot = epochs.plot(n_epochs=8, picks=picks, group_by='selection',
                              butterfly=True)

event_related_plot = epochs.plot_image(picks=['MEG1142'])

# Use case example 1: Connectivity



# Use case example 2: Classification
