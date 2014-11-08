"""
Plot phase amplitude plot showing coupling for synthetic signal and surrogates.
The modulation index is also calculated and plotted.
"""

import numpy as np
from mne.connectivity.cfc import (generate_pac_signal,
                                  phase_amplitude_coupling,
                                  modulation_index)
from mne.viz.misc import plot_phase_amplitude_coupling
from mne.viz.utils import tight_layout

# First, generate the synthetic data
sfreq = 1000.  # sampling frequency

duration, n_epochs = 1., 100  # let us create 100 trials of 1s each

# set the phase modulating freq and amplitude modulated freq
f_phase, f_amplitude = 8., 80.

# To produce a realistic PAC signal, include multiple levels of modulation
data = generate_pac_signal(sfreq, duration, n_epochs, f_phase, f_amplitude,
                           sigma=5)

n_bins = 18  # number of bins
fp_low, fp_high = 8, 13  # range of phase modulating freq
fa_low, fa_high = 60, 100  # range of amplitude modulated freq

# Calculate amplitude distribution and the phase bins
amplitude_distribution, phase_bins = phase_amplitude_coupling(
    data, sfreq,
    fp_low, fp_high, fa_low, fa_high,
    n_bins, n_jobs=1)

# Calculate the modulation index using the amplitude distribution
mi_trials = modulation_index(amplitude_distribution)

# Calculate the mean modulation index and amplitude distribution values
# across trials, we use this for plotting

mean_mi = mi_trials.mean()
mean_amplitude_distribution = amplitude_distribution.mean(axis=0)

# Surrogate analysis, perform similar calculations for surrogate data
surr_amp_dist, surr_phase_bins = phase_amplitude_coupling(
    data,
    sfreq, fp_low, fp_high, fa_low,
    fa_high, n_bins, n_jobs=1,
    surrogates=True)

surr_mi_trials = modulation_index(surr_amp_dist)
surr_mean_mi = surr_mi_trials.mean()
surr_mean_amp_dist = surr_amp_dist.mean(axis=0)

zscore = (mi_trials - np.mean(surr_mi_trials)) / np.std(surr_mi_trials)

from scipy import stats
from mne.stats import fdr_correction
p_values = stats.norm.pdf(zscore)
accept, _ = fdr_correction(p_values, alpha=0.001)
z_threshold = np.abs(mi_trials[accept]).min()

# plot the phase amplitude plots of the mean modulation compared
# to that of the surrogates obtained by shuffling the phase series
# and the amplitude envelope.

import matplotlib.pyplot as plt
plt.figure('Phase Amplitude(PA) Plots')
plt.subplot(1, 3, 1)
plot_phase_amplitude_coupling(phase_bins, mean_amplitude_distribution,
                              title='PA plot data', show=False)
plt.xlim(0, 360)
plt.subplot(1, 3, 2)
plot_phase_amplitude_coupling(surr_phase_bins, surr_mean_amp_dist,
                              title='PA plot surrogates', show=False)
plt.xlim(0, 360)
axes = plt.subplot(1, 3, 3)
ticks = axes.get_xticks().tolist()
plt.xlim(0, 3)
plt.bar([1, 2], [mean_mi, surr_mean_mi], width=0.9, align='center')
ticks[2] = 'data'
ticks[4] = 'surrogates'
ticks[0] = ticks[3] = ticks[1] = ticks[5] = ''
axes.set_xticklabels(ticks)
axes.axhline(z_threshold, 0, 3, color='k', linestyle='--')
plt.title('MI values')
plt.ylabel('Modulation Index (MI)')
tight_layout()
plt.show()
