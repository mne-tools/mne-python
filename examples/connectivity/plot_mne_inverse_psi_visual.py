"""
=====================================================================
Compute Phase Slope Index (PSI) in source space for a visual stimulus
=====================================================================

This example demonstrates how the Phase Slope Index (PSI) [1] can be computed
in source space based on single trial dSPM source estimates. In addition,
the example shows advanced usage of the connectivity estimation routines
by first extracting a label time course for each epoch and then combining
the label time course with the single trial source estimates to compute the
connectivity.

The result clearly shows how the activity in the visual label precedes more
widespread activity (a postivive PSI means the label time course is leading).

References
----------
[1] Nolte et al. "Robustly Estimating the Flow Direction of Information in
Complex Physical Systems", Physical Review Letters, vol. 100, no. 23,
pp. 1-4, Jun. 2008.
"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)


import numpy as np

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.connectivity import seed_target_indices, phase_slope_index

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
fname_label = data_path + '/MEG/sample/labels/Vis-lh.label'

event_id, tmin, tmax = 4, -0.2, 0.3
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
inverse_operator = read_inverse_operator(fname_inv)
raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# Compute inverse solution and for each epoch. Note that since we are passing
# the output to both extract_label_time_course and the phase_slope_index
# functions, we have to use "return_generator=False", since it is only possible
# to iterate over generators once.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=False)

# Now, we generate seed time series by averaging the activity in the left
# visual corex
label = mne.read_label(fname_label)
src = inverse_operator['src']  # the source space used
seed_ts = mne.extract_label_time_course(stcs, label, src, mode='mean_flip')

# Combine the seed time course with the source estimates. There will be a total
# of 7500 signals:
# index 0: time course extracted from label
# index 1..7499: dSPM source space time courses
comb_ts = zip(seed_ts, stcs)

# Construct indices to estimate connectivity between the label time course
# and all source space time courses
vertices = [src[i]['vertno'] for i in range(2)]
n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])

indices = seed_target_indices([0], np.arange(1, n_signals_tot))

# Compute the PSI in the frequency range 8Hz..30Hz. We exclude the baseline
# period from the connectivity estimation
fmin = 8.
fmax = 30.
tmin_con = 0.
sfreq = raw.info['sfreq']  # the sampling frequency

psi, freqs, times, n_epochs, _ = phase_slope_index(
    comb_ts, mode='multitaper', indices=indices, sfreq=sfreq,
    fmin=fmin, fmax=fmax, tmin=tmin_con)

# Generate a SourceEstimate with the PSI. This is simple since we used a single
# seed (inspect the indices variable to see how the PSI scores are arranged in
# the output)
psi_stc = mne.SourceEstimate(psi, vertices=vertices, tmin=0, tstep=1,
                             subject='sample')

# Now we can visualize the PSI using the plot method. We use a custom colormap
# to show signed values
v_max = np.max(np.abs(psi))
brain = psi_stc.plot(surface='inflated', hemi='lh',
                     time_label='Phase Slope Index (PSI)',
                     subjects_dir=subjects_dir,
                     clim=dict(kind='percent', pos_lims=(95, 97.5, 100)))
brain.show_view('medial')
brain.add_label(fname_label, color='green', alpha=0.7)
