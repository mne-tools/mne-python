"""
========================================
Plot activations for subcortical volumes
========================================

"""

# Author: Alan Leggitt <alan.leggitt@ucsf.edu>
#
# License: BSD (3-clause)

print(__doc__)

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mne
from mne import io
from mne.preprocessing import ICA
from mne.datasets import spm_face
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# supress text output
mne.set_log_level(False)

# get the data paths
data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'
subject_dir = subjects_dir + '/spm'
meg_dir = data_path + '/MEG/spm'

# get the data files
bem_fname = subject_dir + '/bem/spm-5120-5120-5120-bem-sol.fif'
mri_fname = meg_dir + '/SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
epo_fname = meg_dir + '/SPM_CTF_MEG_example_faces1_3D_epo.fif'

# load the epoch data if it already exists
if os.path.exists(epo_fname):

    # read the epoch data
    epochs = mne.read_epochs(epo_fname)

else:  # must generate epochs from raw

    # Load and filter data, set up epochs
    raw_fname = meg_dir + '/SPM_CTF_MEG_example_faces1_3D_raw.fif'

    raw = io.Raw(raw_fname, preload=True)  # Take first run

    # use the meg channels
    picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')

    # bandpass filter
    raw.filter(1, 45, method='iir')

    # find the events
    events = mne.find_events(raw, stim_channel='UPPT001')
    event_ids = {"faces": 1, "scrambled": 2}

    # setup epoch parameters
    tmin, tmax = -0.2, 0.6
    baseline = None  # no baseline as high-pass is applied
    reject = dict(mag=1.5e-12)

    # epoch the data
    epochs = mne.Epochs(raw, events, event_ids, tmin, tmax,  picks=picks,
                        baseline=baseline, preload=True, reject=reject)

    # Fit ICA, find and remove major artifacts
    ica = ICA(None, 50).decompose_epochs(epochs, decim=2)

    # exclude sources that resemble ECG or EOG data
    for ch_name in ['MRT51-2908', 'MLF14-2908']:  # ECG, EOG contaminated chs
        scores = ica.find_sources_epochs(epochs, ch_name, 'pearsonr')
        ica.exclude += list(np.argsort(np.abs(scores))[-2:])

    # select ICA sources and reconstruct MEG signals, compute clean ERFs
    epochs = ica.pick_sources_epochs(epochs)

    # save the epoched data
    epochs.save(epo_fname)

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))

# get positions and orientations of subcortical space
pos = mne.source_space.get_segment_positions('spm', 'Right-Amygdala',
                                             random_ori=True,
                                             subjects_dir=subjects_dir)

# setup the right amygdala volume source space
vol_src = mne.setup_volume_source_space('spm', pos=pos)

# setup the cortical surface source space
src = mne.setup_source_space('spm', overwrite=True)

# combine the source spaces
src.append(vol_src[0])

# setup the forward model
forward = mne.make_forward_solution(epochs.info, mri=mri_fname, src=src,
                                    bem=bem_fname)
forward = mne.convert_forward_solution(forward, surf_ori=True)

# Compute inverse solution
snr = 5.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))

# make the inverse operator
inverse_operator = make_inverse_operator(epochs.info, forward, noise_cov,
                                         loose=0.2, depth=0.8)

# Apply inverse solution to both stim types
stc_faces = apply_inverse_epochs(epochs['faces'], inverse_operator, lambda2,
                                 method)
stc_scrambled = apply_inverse_epochs(epochs['scrambled'], inverse_operator,
                                     lambda2, method)

# compare face vs. scrambled trials

# empty array
X = np.zeros((len(epochs.events), len(epochs.times)))

# loop through the epochs
for i, s in enumerate(stc_faces):
    # extract the right amygdala vertices
    x = s.data[8196:].mean(0)
    X[i] = x
for i, s in enumerate(stc_scrambled):
    # extract the right amygdala vertices
    x = s.data[8196:].mean(0)
    X[i+83] = x

t, p = stats.ttest_ind(X[:83], X[83:])
sig, p = mne.stats.fdr_correction(p)  # apply fdr correction

# plot the results
t = epochs.times
s1 = X[:83]
s2 = X[83:]

ax = plt.axes()

l1, = ax.plot(t, s1.mean(0), 'b')  # faces
l2, = ax.plot(t, s2.mean(0), 'g')  # scrambled

ylim = ax.get_ylim()
ax.fill_between(t, ylim[1]*np.ones(t.shape), ylim[0]*np.ones(t.shape), sig,
                facecolor='k', alpha=0.3)
ax.set_xlim((t.min(), t.max()))
ax.set_xlabel('Time (s)')
ax.set_title('Right Amygdala Activation')
ax.legend((l1, l2), ('Faces', 'Scrambled'))
ax.set_ylim(ylim)

plt.show()
