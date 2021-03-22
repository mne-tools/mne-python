# -*- coding: utf-8 -*-
"""
.. _tut-fixed-length-epochs:

Creating epochs of equal length
===============================

This tutorial shows how to create equal length epochs and briefly demonstrates
an example of their use in connectivity analysis.

First, we import necessary modules and read in a sample raw
data set. This data set contains brain activity that is event-related, i.e.
synchronized to the onset of auditory stimuli. However, rather than creating
epochs by segmenting the data around the onset of each stimulus, we will
create 30 second epochs that allow us to perform non-event-related analyses of
the signal.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import compute_proj_ecg

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file)

###############################################################################
# For this tutorial we'll crop and resample the raw data to a manageable size
# for our web server to handle, ignore EEG channels, and remove the heartbeat
# artifact so we don't get spurious correlations just because of that.

raw.crop(tmax=150).resample(100).pick('meg')
ecg_proj, _ = compute_proj_ecg(raw, ch_name='MEG 0511')  # No ECG chan
raw.add_proj(ecg_proj)
raw.apply_proj()

###############################################################################
# To create fixed length epochs, we simply call the function and provide it
# with the appropriate parameters indicating the desired duration of epochs in
# seconds, whether or not to preload data, whether or not to reject epochs that
# overlap with raw data segments annotated as bad, whether or not to include
# projectors, and finally whether or not to be verbose. Here, we choose a long
# epoch duration (30 seconds). To conserve memory, we set ``preload`` to
# ``False``.

epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=False)

###############################################################################
# Characteristics of Fixed Length Epochs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Fixed length epochs are generally unsuitable for event-related analyses. This
# can be seen in an image map of our fixed length
# epochs. When the epochs are averaged, as seen at the bottom of the plot,
# misalignment between onsets of event-related activity results in noise.

event_related_plot = epochs.plot_image(picks=['MEG 1142'])

###############################################################################
# For information about creating epochs for event-related analyses, please see
# :ref:`tut-epochs-class`.
#
# Example Use Case for Fixed Length Epochs: Connectivity Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Fixed lengths epochs are suitable for many types of analysis, including
# frequency or time-frequency analyses, connectivity analyses, or
# classification analyses. Here we briefly illustrate their utility in a sensor
# space connectivity analysis.
#
# The data from our epochs object has shape ``(n_epochs, n_sensors, n_times)``
# and is therefore an appropriate basis for using MNE-Python's envelope
# correlation function to compute power-based connectivity in sensor space. The
# long duration of our fixed length epochs, 30 seconds, helps us reduce edge
# artifacts and achieve better frequency resolution when filtering must
# be applied after epoching.
#
# Let's examine the alpha band. We allow default values for filter parameters
# (for more information on filtering, please see :ref:`tut-filter-resample`).

epochs.load_data().filter(l_freq=8, h_freq=12)
alpha_data = epochs.get_data()

###############################################################################
# If desired, separate correlation matrices for each epoch can be obtained.
# For envelope correlations, this is done by passing ``combine=None`` to the
# envelope correlations function.

corr_matrix = mne.connectivity.envelope_correlation(alpha_data, combine=None)

###############################################################################
# Now we can plot correlation matrices. We'll compare the first and last
# 30-second epochs of the recording:

first_30 = corr_matrix[0]
last_30 = corr_matrix[-1]
corr_matrices = [first_30, last_30]
color_lims = np.percentile(np.array(corr_matrices), [5, 95])
titles = ['First 30 Seconds', 'Last 30 Seconds']

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Correlation Matrices from First 30 Seconds and Last 30 Seconds')
for ci, corr_matrix in enumerate(corr_matrices):
    ax = axes[ci]
    mpbl = ax.imshow(corr_matrix, clim=color_lims)
    ax.set_xlabel(titles[ci])
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.2, 0.025, 0.6])
cbar = fig.colorbar(ax.images[0], cax=cax)
cbar.set_label('Correlation Coefficient')
