"""
==============================
Generate simulated evoked data
==============================

"""
# Author: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from mne import (read_proj, read_forward_solution, read_cov, read_label,
                 pick_types_forward, pick_types, read_evokeds)
from mne.io import Raw
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, generate_evoked

print(__doc__)

###############################################################################
# Load real data as templates
data_path = sample.data_path()

raw = Raw(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = read_proj(data_path + '/MEG/sample/sample_audvis_ecg_proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])

cov = read_cov(cov_fname)

condition = 'Left Auditory'
evoked_template = read_evokeds(ave_fname, condition=condition, baseline=None)
evoked_template.pick_types(meg=True, eeg=True, exclude=raw.info['bads'])

label_names = ['Aud-lh', 'Aud-rh']
labels = [read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
          for ln in label_names]

###############################################################################
# Generate source time courses and the correspond evoked data
snr = 6  # dB
tmin = -0.1
sfreq = raw.info['sfreq']  # Hz
tstep = 1. / sfreq
n_samples = 600
times = np.arange(n_samples, dtype=np.float) * tstep + tmin

# Generate times series for 2 dipoles
stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                          random_state=42, labels=labels)

###############################################################################
# Generate noisy evoked data
picks = pick_types(raw.info, meg=True, exclude='bads')
iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=180)[1]
evoked = generate_evoked(fwd, stc, evoked_template, cov, snr,
                         tmin=0.0, tmax=0.2, iir_filter=iir_filter)

###############################################################################
# Plot
plot_sparse_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.5, high_resolution=True)

plt.figure()
plt.psd(evoked.data[0])

evoked.plot()
