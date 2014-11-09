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

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Praveen Sripad <praveen.sripad@rwth-aachen.de>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import mne
from mne.connectivity.cfc import cross_frequency_coupling
from mne.viz.misc import plot_cross_frequency_coupling


# Data signal obtained from math.bu.edu/people/mak/MA666/data_1.mat
# Converted to .npy and reshaped.
data_dir = op.join(op.dirname(mne.__file__), 'data')
data_fname = op.join(data_dir, 'cfc_data.npy')

# read the data
data = np.load(data_fname)

# set the parameters
sfreq, phase_freq, l_amp_freq, h_amp_freq, n_freqs = 1000., 8, 60., 100., 100
n_cycles, n_jobs, alpha = 10, 4, 0.001
n_samples = data.size

# computing the amplitude traces and the erp signal
times, freqs, traces, ztraces, z_threshold, erp = cross_frequency_coupling(
    data, sfreq, phase_freq, n_cycles, l_amp_freq, h_amp_freq, n_freqs,
    alpha, n_jobs=1)

# plotting the amplitude traces and erp for various frequency points
plot_cross_frequency_coupling(times, freqs, traces, ztraces, z_threshold, erp)
