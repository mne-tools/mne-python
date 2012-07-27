"""
==============================
Generate simulated evoked data
==============================

"""
# Author: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import pylab as pl

import mne
from mne.fiff.pick import pick_types_evoked, pick_types_forward
from mne.datasets import sample
from mne.time_frequency import iir_filter_raw, morlet
from mne.viz import plot_evoked, plot_sparse_source_estimates
from mne.simulation import generate_sparse_stc, generate_evoked

###############################################################################
# Load real data as templates
data_path = sample.data_path('.')

raw = mne.fiff.Raw(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/ecg_proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])

cov = mne.read_cov(cov_fname)

evoked_template = mne.fiff.read_evoked(ave_fname, setno=0, baseline=None)
evoked_template = pick_types_evoked(evoked_template, meg=True, eeg=True,
                                    exclude=raw.info['bads'])

label_names = ['Aud-lh', 'Aud-rh']
labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
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
picks = mne.fiff.pick_types(raw.info, meg=True)
iir_filter = iir_filter_raw(raw, order=5, picks=picks, tmin=60, tmax=180)
evoked = generate_evoked(fwd, stc, evoked_template, cov, snr,
                         tmin=0.0, tmax=0.2, iir_filter=iir_filter)

###############################################################################
# Plot
plot_sparse_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
                                opacity=0.5, high_resolution=True)

pl.figure()
pl.psd(evoked.data[0])

pl.figure()
plot_evoked(evoked)
