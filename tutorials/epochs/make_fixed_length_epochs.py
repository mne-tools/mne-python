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

picks = ['mag', 'grad']

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file)

###############################################################################
# To create fixed length epochs, we simply call the function and provide it
# with the appropriate parameters indicating the desired duration of epochs in
# seconds, whether or not to preload data, whether or not to reject epochs that
# overlap with raw data segments annotated as bad, whether or not to include
# projectors, and finally whether or not to be verbose. Here, we choose a long
# epoch duration (10 seconds). To conserve memory, we set preload to
# False. We elect to reject segments of data marked as bad. In this case,
# the MEG sample data set includes SSP projectors that help control room noise
# in the data and will be left in for noise suppression.


epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=False,
                                      reject_by_annotation=True, proj=True,
                                      verbose=True)

###############################################################################
# A key characteristic of fixed length epochs is that they are generally
# unsuitable for event-related analyses. Two quick visualizations will help
# illustrate why. First, we create a butterfly plot grouping channels together
# by spatial region.

# Visualize the results and compare to other types of epoched data

timeseries_plot = epochs.plot(n_epochs=8, picks=picks, group_by='selection',
                              butterfly=True)

###############################################################################
# Clear peaks of event-related activity corresponding to stimuli onsets are
# seen in each 10-second epoch.

event_related_plot = epochs.plot_image(picks=['MEG1142'])

# Use case example 1: Connectivity



# Use case example 2: Classification
