"""
=====================================================================
Compute normalized amplitude traces showing phase amplitude coupling.
=====================================================================

Computes the normalized amplitude traces for a cross frequency coupled
signal across a given range of frequencies and displays it along with
the event related average response.

References:
High gamma power is phase-locked to theta oscillations in human neocortex.
Canolty RT1, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
Berger MS, Barbaro NM, Knight RT.
(Science. 2006)
"""

# Authors: Praveen Sripad <praveen.sripad@rwth-aachen.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

print(__doc__)

from mne.connectivity.cfc import (simulate_cfc_data, cross_frequency_coupling,
                                  compute_cfc_stats)
from mne.viz import plot_cross_frequency_coupling
import numpy as np

# set the parameters
sfreq, phase_freq = 1000., 8.
n_epochs = 400

# simulate the data with 6. Hz signal coupled with 80.-120. Hz activity
data = simulate_cfc_data(sfreq, n_epochs, phase_freq,
                         l_amp_freq=80., h_amp_freq=120., surrogates=False)

l_amp_freq, h_amp_freq, n_freqs = 60., 150., 100
freqs = np.logspace(np.log10(l_amp_freq), np.log10(h_amp_freq), n_freqs)
n_cycles, alpha = 10, 0.001
n_samples = data.size

# computing the amplitude traces and the average signal
times, ampmat, traces, avg, trigger_inds = cross_frequency_coupling(data,
                             sfreq, phase_freq, n_cycles, freqs, -0.4, 0.4)

# perform statistics on the amplitude traces along with surrogates
ztraces, z_threshold = compute_cfc_stats(sfreq, ampmat, traces, freqs,
                       trigger_inds, n_surrogates=10000, alpha=alpha,
                       random_state=None)

# plotting the amplitude traces and average for various frequency points
plot_cross_frequency_coupling(times, freqs, traces, ztraces, z_threshold, avg)
