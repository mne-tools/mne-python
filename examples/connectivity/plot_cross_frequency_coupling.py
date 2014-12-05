"""
=====================================================================
Compute normalized amplitude traces showing phase amplitude coupling.
=====================================================================

Computes the normalized amplitude traces for a cross frequency coupled
signal across a given range of frequencies and displays it along with
the evoked related potential.

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

from mne.connectivity.cfc import simulate_cfc_data, cross_frequency_coupling
from mne.viz.misc import plot_cross_frequency_coupling

# set the parameters
sfreq, phase_freq = 1000., 6.
n_epochs = 400

# simulate the data with 6. Hz signal coupled with 80.-120. Hz activity
data = simulate_cfc_data(sfreq, n_epochs, phase_freq,
                         l_amp_freq=80., h_amp_freq=120., n_jobs=1,
                         random_state=42, surrogates=False)

l_amp_freq, h_amp_freq, n_freqs = 50., 150., 100
n_cycles, alpha = 10, 0.001
n_samples = data.size
# computing the amplitude traces and the erp signal
times, freqs, traces, ztraces, z_threshold, erp = cross_frequency_coupling(
    data, sfreq, phase_freq, n_cycles, l_amp_freq, h_amp_freq, n_freqs,
    alpha, random_state=42)

# plotting the amplitude traces and erp for various frequency points
plot_cross_frequency_coupling(times, freqs, traces, ztraces, z_threshold, erp)
