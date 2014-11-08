"""
Plot phase amplitude plot showing coupling for synthetic signal and surrogates.
"""
import mne
import numpy as np
from mne.connectivity.cfc import cross_frequency_coupling
from mne.viz.misc import plot_cross_frequency_coupling
import os.path as op

# Data signal obtained from math.bu.edu/people/mak/MA666/data_1.mat
# Converted to .npy and reshaped.
data_dir = op.join(op.dirname(mne.__file__), 'data')
data_fname = op.join(data_dir, 'cfc_data.npy')

# read the data
data = np.load(data_fname)

# set the parameters
sfreq, phase_freq, fa_high, fa_low, f_n = 1000., 8, 60., 100., 100
n_cycles, n_jobs, alpha = 10, 4, 0.001
n_samples = data.size

# computing the amplitude traces and the erp signal
times, freqs, traces, ztraces, z_threshold, erp = cross_frequency_coupling(
    data, sfreq, phase_freq, n_cycles, fa_low, fa_high, f_n, alpha, n_jobs=1)

# plotting the amplitude traces and erp for various frequency points
plot_cross_frequency_coupling(times, freqs, traces, ztraces, z_threshold, erp)
