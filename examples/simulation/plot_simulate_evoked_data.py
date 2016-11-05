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

import mne
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked

print(__doc__)

###############################################################################
# Load real data as templates
data_path = sample.data_path()

raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg-proj.fif')
raw.info['projs'] += proj
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels

fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(ave_fname)

label_names = ['Aud-lh', 'Aud-rh']
labels = [mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
          for ln in label_names]

###############################################################################
# Generate source time courses from 2 dipoles and the correspond evoked data

times = np.arange(300, dtype=np.float) / raw.info['sfreq'] - 0.1
rng = np.random.RandomState(42)


def data_fun(times):
    """Function to generate random source time courses"""
    return (1e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.15 + 0.05 * rng.randn(1)) ** 2 / 0.01))

stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                          random_state=42, labels=labels, data_fun=data_fun)

###############################################################################
# Generate noisy evoked data
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=180)[1]
snr = 6.  # dB
evoked = simulate_evoked(fwd, stc, info, cov, snr, iir_filter=iir_filter)

###############################################################################
# Plot
plot_sparse_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.5, high_resolution=True)

plt.figure()
plt.psd(evoked.data[0])

evoked.plot()
