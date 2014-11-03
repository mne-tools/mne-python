"""
Plot phase amplitude plots showing coupling for synthetic signal and surrogates.
The modulation index is also calculated and shown.
"""

from mne.connectivity.cfc import (generate_pac_signal,
                                  phase_amplitude_coupling,
                                  make_surrogate_data,
                                  modulation_index)
from mne.viz.misc import plot_phase_amplitude_coupling

# First, generate the synthetic data
fs = 1000. # sampling frequency

times, trials = 10, 100 # let us create 100 trials of 10s each

# set the phase modulating freq and amplitude modulated freq
f_phase, f_amplitude = 10., 60.

data = generate_pac_signal(fs, times, trials, f_phase, f_amplitude)

bin_num = 18 # number of bins
fp_low, fp_high = 5, 10 # range of phase modulating freq
fa_low, fa_high = 60, 140 # range of amplitude modulated freq

# Calculate amplitude distribution and the phase bins
amplitude_distribution, phase_bins = phase_amplitude_coupling(data, fs,
                                     fp_low, fp_high, fa_low, fa_high,
                                     bin_num, n_jobs=2)

# Calculate the modulation index using the amplitude distribution
mi_trials = modulation_index(amplitude_distribution)

# Calculate the mean modulation index and amplitude distribution values
# across trials, we use this for plotting

mean_mi = mi_trials.mean()
mean_amplitude_distribution = amplitude_distribution.mean(axis=0)

# Surrogate analysis, perform similar calculation for surrogate data
surrogate_data = make_surrogate_data(data)

surr_amp_dist, surr_phase_bins = phase_amplitude_coupling(surrogate_data,
                                          fs, fp_low, fp_high, fa_low, fa_high,
                                          bin_num, n_jobs=2, surrogates=True)

surr_mi_trials = modulation_index(surr_amp_dist)

surr_mean_mi = surr_mi_trials.mean()
surr_mean_amp_dist = surr_amp_dist.mean(axis=0)

import matplotlib.pyplot as plt

# plot the phase amplitude plots of the mean modulation compared
# to that of the surrogates obtained by shuffling the phase series
#  and the amplitude envelope.

plt.figure('Phase Amplitude(PA) Plots')
plt.subplot(1, 3, 1)
plot_phase_amplitude_coupling(phase_bins, mean_amplitude_distribution, title='PA plot data', show=False)
plt.xlim(0, 360)
plt.subplot(1, 3, 2)
plot_phase_amplitude_coupling(surr_phase_bins, surr_mean_amp_dist, title='PA plot surrogates', show=False)
plt.xlim(0, 360)
axes = plt.subplot(1, 3, 3)
ticks = axes.get_xticks().tolist()
plt.xlim(0, 3)
plt.bar([1, 2], [mean_mi, surr_mean_mi], width=0.9, align='center')
ticks[2] = 'MI data'
ticks[4] = 'MI surrogates'
ticks[0] = ticks[3] = ticks[1] = ticks[5] = ''
axes.set_xticklabels(ticks)
plt.title('MI values')
plt.ylabel('Modulation Index (MI)')
plt.tight_layout()
plt.show()

