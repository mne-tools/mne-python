"""
==============================
Generate simulated source data
==============================
"""
# Author: Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#         Samuel Deslauriers-Gauthier <sam.deslauriers@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

import mne
from mne.datasets import sample

print(__doc__)

# For this example, we will be using the information of the sample subject.
# This will download the data if it not already on your machine. We also set
# the subjects directory so we don't need to give it to functions.
data_path = sample.data_path()
mne.set_config('SUBJECTS_DIR', op.join(data_path, 'subjects'))
subject = 'sample'

# First, we get an info structure from the test subject.
ave_filename = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
info = mne.io.read_info(ave_filename)
tstep = 1 / info['sfreq']

# To simulate sources, we also need a source space. It can be obtained from the
# forward solution of the sample subject.
fwd_fname = op.join(data_path, 'MEG', subject,
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']

# To select a region to activate, we use the caudal middle frontal to grow
# a region of interest.
atlas = mne.read_labels_from_annot(subject)
selected_label = next(l for l in atlas if l.name == 'caudalmiddlefrontal-lh')
location = 'center'  # Use the center of the region as a seed.
extent = 10.  # Extent in mm of the region.
label = mne.label.select_sources(
    subject, selected_label, location=location, extent=extent)

# Define the time course of the activity for each source of the region to
# activate. Here we use a sine wave.
source_time_series = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10e-9

# Define when the activity occurs using events. The first column is the sample
# of the event, the second is not used, and the third is the event id. Here the
# events occur every 200 samples.
nb_events = 50
events = np.zeros((nb_events, 3))
events[:, 0] = np.arange(100, 200 * nb_events + 1, 200)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

# Create simulated source activity. Here we use a SourceSimulator whose
# add_data method is key. It specified where (label), what
# (source_time_series), and when (events) an event type will occur.
source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
source_simulator.add_data(label, source_time_series, events)

# Project the source time series to sensor space and add some noise. The source
# simulator can be given directly to the simulate_raw function.
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])
raw.plot()

# Plot evoked data to get another view of the simulated raw data.
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
evoked = epochs.average()
evoked.plot()
