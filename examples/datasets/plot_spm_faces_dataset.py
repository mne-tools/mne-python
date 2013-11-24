"""
==========================================
From raw data to dSPM on SPM Faces dataset
==========================================

Runs a full pipeline using MNE-Python:
- artifact removal
- averaging Epochs
- forward model computation
- source reconstruction using dSPM on the contrast : "faces - scrambled"

"""
print __doc__

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import spm_face
from mne.preprocessing import ICA
from mne import fiff
from mne.minimum_norm import make_inverse_operator, apply_inverse


data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'

###############################################################################
# Load and filter data, set up epochs

raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D_raw.fif'

raw = fiff.Raw(raw_fname % 1, preload=True) # Take first run

picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 45, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')
event_ids = {"faces":1, "scrambled":2}

tmin, tmax = -0.2, 0.6
baseline = None  # no baseline as high-pass is applied
reject = dict(mag=1.5e-12)

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax,  picks=picks,
                    baseline=baseline, preload=True, reject=reject)

# Fit ICA, find and remove major artifacts

ica = ICA(None, 50).decompose_epochs(epochs, decim=2)

for ch_name in ['MRT51-2908', 'MLF14-2908']:  # ECG, EOG contaminated chs
    scores = ica.find_sources_epochs(epochs, ch_name, 'pearsonr')
    ica.exclude += list(np.argsort(np.abs(scores))[-2:])

ica.plot_topomap(np.unique(ica.exclude))  # plot components found


# select ICA sources and reconstruct MEG signals, compute clean ERFs

epochs = ica.pick_sources_epochs(epochs)

evoked = [epochs[k].average() for k in event_ids]

contrast = evoked[1] - evoked[0]

evoked.append(contrast)

for e in evoked:
    e.plot(ylim=dict(mag=[-400, 400]))

plt.show()

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))

###############################################################################
# Compute forward model

# Make source space
src = mne.setup_source_space('spm', spacing='oct6', subjects_dir=subjects_dir,
                             overwrite=True)

mri = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'
forward = mne.make_forward_solution(contrast.info, mri=mri, src=src, bem=bem)
forward = mne.convert_forward_solution(forward, surf_ori=True)

###############################################################################
# Compute inverse solution

snr = 5.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

inverse_operator = make_inverse_operator(contrast.info, forward, noise_cov,
                                         loose=0.2, depth=0.8)

# Compute inverse solution on contrast
stc = apply_inverse(contrast, inverse_operator, lambda2, method,
                    pick_normal=False)
# stc.save('spm_%s_dSPM_inverse' % constrast.comment)

# plot constrast
# Plot brain in 3D with PySurfer if available. Note that the subject name
# is already known by the SourceEstimate stc object.
brain = stc.plot(surface='inflated', hemi='both', subjects_dir=subjects_dir)
brain.set_data_time_index(173)
brain.scale_data_colormap(fmin=4, fmid=6, fmax=8, transparent=True)
brain.show_view('ventral')
# brain.save_image('dSPM_map.png')
