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
from mne.time_frequency import fit_iir_model_raw, morlet
from mne.viz import plot_sparse_source_estimates
from mne.simulation import generate_sparse_stc, generate_evoked

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
sfreq = 1000.  # Hz
tstep = 1. / sfreq
n_samples = 600
times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

# Generate times series from 2 Morlet wavelets
stc_data = np.zeros((len(labels), len(times)))
Ws = morlet(sfreq, [3, 10], n_cycles=[1, 1.5])
stc_data[0][:len(Ws[0])] = np.real(Ws[0])
stc_data[1][:len(Ws[1])] = np.real(Ws[1])
stc_data *= 100 * 1e-9  # use nAm as unit

# time translation
stc_data[1] = np.roll(stc_data[1], 80)
stc = generate_sparse_stc(fwd['src'], labels, stc_data, tmin, tstep,
                          random_state=0)

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
