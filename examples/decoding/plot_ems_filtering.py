"""
==============================================
Compute effect-matched-spatial filtering (EMS)
==============================================

This example computes the EMS to reconstruct the time course of
the experimental effect as described in:

Aaron Schurger, Sebastien Marti, and Stanislas Dehaene, "Reducing multi-sensor
data to a single time course that reveals experimental effects",
BMC Neuroscience 2013, 14:122


This technique is used to create spatial filters based on the
difference between two conditions. By projecting the trial onto the
corresponding spatial filters, surrogate single trials are created
in which multi-sensor activity is reduced to one time series which
exposes experimental effects, if present.

We will first plot a trials x times image of the single trials and order the
trials by condition. A second plot shows the average time series for each
condition. Finally a topographic plot is created which exhibits the
temporal evolution of the spatial filters.
"""
# Author: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.decoding import compute_ems

print(__doc__)

data_path = sample.data_path()

# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_ids = {'AudL': 1, 'VisL': 3, 'AudR': 2, 'VisR': 4}
tmin = -0.2
tmax = 0.5

# Read data and create epochs
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 45)
events = mne.read_events(event_fname)

include = []  # or stim channels ['STI 014']
ch_type = 'grad'
picks = mne.pick_types(raw.info, meg=ch_type, eeg=False, stim=False, eog=True,
                       include=include, exclude='bads')

reject = dict(grad=4000e-13, eog=150e-6)

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks,
                    baseline=None, reject=reject)

# Let's equalize the trial counts in each condition
epochs.equalize_event_counts(epochs.event_id, copy=False)

# compute surrogate time series
surrogates, filters, conditions = compute_ems(epochs, ['AudL', 'VisL'])

times = epochs.times * 1e3
plt.figure()
plt.title('single trial surrogates')
plt.imshow(surrogates[conditions.argsort()], origin='lower', aspect='auto',
           extent=[times[0], times[-1], 1, len(surrogates)],
           cmap='RdBu_r')
plt.xlabel('Time (ms)')
plt.ylabel('Trials (reordered by condition)')

plt.figure()
plt.title('Average EMS signal')

mappings = [(k, v) for k, v in event_ids.items() if v in conditions]
for key, value in mappings:
    ems_ave = surrogates[conditions == value]
    ems_ave *= 1e13
    plt.plot(times, ems_ave.mean(0), label=key)
plt.xlabel('Time (ms)')
plt.ylabel('fT/cm')
plt.legend(loc='best')


# visualize spatial filters across time
plt.show()
evoked = epochs.average()
evoked.data = filters
evoked.plot_topomap(ch_type=ch_type)
