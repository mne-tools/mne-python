"""
==============================================
Compute effect-matched-spatial filtering (EMS)
==============================================

This example computes the EMS to reconstruct the time course of
the experimental effect as described in:

Aaron Schurger, Sebastien Marti, and Stanislas Dehaene, "Reducing multi-sensor
data to a single time course that reveals experimental effects",
BMC Neuroscience 2013, 14:122

XXX : can you explain a bit the idea? and what figures display?

XXX : how do you explain a difference in the baseline?
"""

# Author: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne import fiff
from mne.datasets import sample
from mne.decoding import compute_ems
data_path = sample.data_path()

# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_ids = {'AudL': 1, 'VisL': 3}
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
raw.filter(1, 45)
events = mne.read_events(event_fname)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                        include=include, exclude='bads')
# Read epochs

reject = dict(grad=4000e-13, eog=150e-6)

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks,
                    baseline=None, reject=reject)

# Let's equalize the trial counts in each condition
epochs.equalize_event_counts(['AudL', 'VisL'], copy=False)

picks = fiff.pick_types(epochs.info, meg='grad', exclude='bads')

surrogate_trials, spatial_filters = compute_ems(epochs, ['AudL', 'VisL'],
                                                picks=picks)

order = epochs.events[:, 2].argsort()
times = epochs.times * 1e3

import matplotlib.pyplot as plt

plt.figure()
plt.title('single trial surrogates')
plt.imshow(surrogate_trials[order], origin='lower', aspect='auto',
           extent=[times[0], times[-1], 1, len(epochs)])
plt.xlabel('Time (ms)')
plt.ylabel('Trials (reordered by condition)')

plt.figure()
plt.title('Average EMS signal')
for key, value in epochs.event_id.items():
    ems_ave = surrogate_trials[epochs.events[:, 2] == value]
    ems_ave /= 4e-11
    plt.plot(times, ems_ave.mean(0), label=key)
plt.xlabel('Time (ms)')
plt.ylabel('fT/cm')
plt.legend(loc='best')

# visualize spatial filters across time
evoked = epochs.average()
evoked.data = spatial_filters
evoked.plot_topomap(ch_type='grad')
