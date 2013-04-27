"""
====================================================================
Mass-univariate Twoway Repeated Measures ANOVA on Single Trial Power
====================================================================

This script shows how to conduct a mass-univariate repeated measures
ANOVA. As the model to be fitted assumes two fully crossed factors,
we will study the interplay between perceptual modality
(auditory VS visual) and the location of stimulus presentation
(left VS right). Here we use single trials as replications (subject)
and use time slices and frequency bands for mass-univariate
observations. We will conclude with visualizing each effect by
creating a corresponding mass-univariate effect image.
"""
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.time_frequency import single_trial_power
from mne.stats.parametric import r_anova_twoway
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                        stim=False, include=include, exclude='bads')

ch_name = raw.info['ch_names'][picks[0]]

# Load conditionw
reject = dict(grad=4000e-13, eog=150e-6)
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                reject=reject)

# make sure all conditions have the same counts, this is crucial for the ANOVA
epochs.equalize_event_counts(event_id, copy=False)
# Time vector
times = 1e3 * epochs.times  # change unit to ms

# Factor to downs-sample the temporal dimension of the PSD computed by
# single_trial_power. 
decim = 2
frequencies = np.arange(7, 30, 3)  # define frequencies of interest
Fs = raw.info['sfreq']  # sampling in Hz
n_cycles = 1.5
baseline_mask = times[::decim] < 0

# now create TFR representations for all conditions
epochs_power = []
for condition in [epochs[k].get_data()[:, 97:98, :] for k in event_id]:
    this_power = single_trial_power(condition, Fs=Fs, frequencies=frequencies,
        n_cycles=n_cycles, use_fft=False, decim=decim)
    this_power = this_power[:, 0, :, :]  # we only have one channel.
    # Compute ratio with baseline power (be sure to correct time vector with
    # decimation factor)
    epochs_baseline = np.mean(this_power[:, :, baseline_mask], axis=2)
    this_power /= epochs_baseline[..., np.newaxis]
    epochs_power.append(this_power)


n_conditions = len(epochs.event_id)
n_replications = epochs.events.shape[0] / n_conditions
factor_levels = [2, 2]  # number of levels in each factor
# assemble data matrix and swap axes so the trial replications
# are the first dimension and the conditions are the second dimension
data = np.swapaxes(np.asarray(epochs_power), 1, 0)
# reshape last two dimensions in one mass-univariate observation-vector
data = data.reshape(n_replications, n_conditions, 8 * 211)
# so we have replications * conditions * observations:
print data.shape

# now we can run our repeated measures ANOVA.
fvals, _ = r_anova_twoway(data, factor_levels, return_pvals=False, n_jobs=2)

effect_labels = ['modality', 'location', 'modality by location']
import pylab as pl
for effect, effect_label in zip(np.split(fvals, 3,  axis=1), effect_labels):
    pl.figure()
    pl.imshow(effect.reshape(8, 211), cmap=pl.cm.jet, extent=[times[0],
        times[-1], frequencies[0], frequencies[-1]], aspect='auto',
        origin='lower')
    pl.colorbar()
    pl.xlabel('time (ms)')
    pl.ylabel('Frequency (Hz)')
    pl.title(r"Induced F-values '%s'(%s)" % (effect_label, ch_name))
    pl.show()
