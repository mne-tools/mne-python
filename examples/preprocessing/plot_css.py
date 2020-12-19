"""
=================================================================
Use Cortical Singal Suppression (CSS) to remove cortical signals
=================================================================

This script shows an easy example of how to use CSS. CSS
suppresses the cortical contribution to the signal subspace
in magnetometer or EEG data, facilitating detection of subcortical
signals. We will illustrate how it works by simulating one cortical
and one subcortical oscillation at different frequencies; 40 Hz and
239 Hz for cortical and subcortical activity, respectively, then
process it with CSS and look at the power spectral density of the
raw and processed data.

Presented in Samuelsson J, Khan S, Sundaram P, Peled N, Hamalainen M
(2018) Cortical Signal Suppression (CSS) for detection of subcortical activity
using MEG and EEG, Brain topography 32 (2), 215-228.
Please refer to this paper.
"""
# Authors: John G Samuelsson <johnsam@mit.edu>

import numpy as np
import mne
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_evoked

###############################################################################
# Load sample subject data
data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
bem_fname = subjects_dir + '/sample' + '/bem' + '/sample-5120-bem-sol.fif'

raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
cov = mne.read_cov(cov_fname)
all_labels = mne.read_labels_from_annot(subject='sample',
                                        subjects_dir=subjects_dir)
hiplab, postcenlab, precenlab = [all_labels[x] for x in [34, 44, 48]]

###############################################################################
# Simulate one cortical dipole (40 Hz) and one subcortical (239 Hz)


def cortical_waveform(times):
    return 1e-8 * np.cos(times * 2 * np.pi * 40)


def subcortical_waveform(times):
    return 1e-7 * np.cos(times * 2 * np.pi * 239)

times = np.linspace(0, 0.5, int(0.5 * raw.info['sfreq']))
stc = simulate_sparse_stc(fwd['src'], n_dipoles=3, times=times,
                          location='center', subjects_dir=subjects_dir,
                          labels=[postcenlab, precenlab, hiplab],
                          data_fun=cortical_waveform)
stc.data[np.where(np.isin(stc.vertices[0], hiplab.vertices))[0], :] = \
    subcortical_waveform(times)
evoked = simulate_evoked(fwd, stc, raw.info, cov, nave=5, random_state=6)

###############################################################################
# Process with CSS and plot PSD
evoked_subcortical = \
    mne.preprocessing.filter_cortical(evoked, projection_vectors=6,
                                      show_psd=True)
