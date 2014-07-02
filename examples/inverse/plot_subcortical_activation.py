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
from mne.preprocessing import ICA, create_eog_epochs
from mne.datasets import spm_face
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# supress text output
mne.set_log_level(False)

# get the data paths
data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'
subject_dir = subjects_dir + '/spm'
meg_dir = data_path + '/MEG/spm'
fstring = meg_dir + '/SPM_CTF_MEG_example_faces1_3D'

# get the meg data files
trans_fname = fstring + '_raw-trans.fif'
epo_fname = fstring + '-epo.fif'
fwd_fname = fstring + '-oct-6-amyg-vol-fwd.fif'
evo_fname = fstring + '-ave.fif'

# get the freesurfer data files
bem_fname = subject_dir + '/bem/spm-5120-5120-5120-bem-sol.fif'

##############################################################################
# get the epoch data

# load the epoch data if it already exists
if os.path.exists(epo_fname):

    # read the epoch data
    epochs = mne.read_epochs(epo_fname)
    event_ids = {"faces": 1, "scrambled": 2}

else:  # generate evoked from raw

    raw_fname = fstring + '_raw.fif'

    raw = io.Raw(raw_fname, preload=True)  # Take first run

    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    raw.filter(1, 30, method='iir')

    events = mne.find_events(raw, stim_channel='UPPT001')
    event_ids = {"faces": 1, "scrambled": 2}

    tmin, tmax = -0.2, 0.6
    baseline = None  # no baseline as high-pass is applied
    reject = dict(mag=5e-12)

    epochs = mne.Epochs(raw, events, event_ids, tmin, tmax,  picks=picks,
                        baseline=baseline, preload=True, reject=reject)

    # Fit ICA, find and remove major artifacts
    ica = ICA(n_components=0.95).fit(raw, decim=6, reject=reject)

    # compute correlation scores, get bad indices sorted by score
    eog_epochs = create_eog_epochs(raw, ch_name='MRT31-2908', reject=reject)
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name='MRT31-2908')
    ica.exclude += eog_inds[:1]
    ica.apply(epochs)  # clean data

    # save the epoch data
    epochs.save(epo_fname)


# calculate the evoked data
evoked = [epochs[k].average() for k in event_ids]
contrast = evoked[1] - evoked[0]
evoked.append(contrast)

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs, tmax=0)

##############################################################################
# get the forward operator

# load the forward solution if it already exists
if os.path.exists(fwd_fname):

    # read the forward solution
    forward = mne.read_forward_solution(fwd_fname)

else:  # merge cortical and subcortical sources and computer forward solution

    # setup the cortical surface source space
    src = mne.setup_source_space('spm', overwrite=True)

    # add a subcortical volumes
    src = mne.source_space.add_subcortical_volumes(src, ['Left-Amygdala',
                                                         'Right-Amygdala'])

    # setup the forward model
    forward = mne.make_forward_solution(epochs.info, mri=trans_fname, src=src,
                                        bem=bem_fname)

    # save the forward solution
    mne.write_forward_solution(fwd_fname, forward, overwrite=True)

##############################################################################
# Compute inverse solution

snr = 5.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))

# make the inverse operator
inverse_operator = make_inverse_operator(epochs.info, forward, noise_cov,
                                         loose=None, depth=None)

# Apply inverse solution to both stim types
stc_faces = apply_inverse_epochs(epochs['faces'], inverse_operator, lambda2,
                                 method)
stc_scrambled = apply_inverse_epochs(epochs['scrambled'], inverse_operator,
                                     lambda2, method)

##############################################################################
# compare face vs. scrambled trials

# empty arrays
s1 = np.zeros((len(stc_faces), len(epochs.times)))
s2 = np.zeros((len(stc_scrambled), len(epochs.times)))

# loop through the epochs
for i, s in enumerate(stc_faces):
    # extract the amygdala vertices
    x = s.data[len(s.vertno[0])+len(s.vertno[1]):].mean(0)
    s1[i] = x

for i, s in enumerate(stc_scrambled):
    # extract the amygdala vertices
    x = s.data[len(s.vertno[0])+len(s.vertno[1]):].mean(0)
    s2[i] = x

t, p = stats.ttest_ind(s1, s2)
sig, p = mne.stats.fdr_correction(p)  # apply fdr correction

###############################################################################
# plot the results
t = epochs.times

ax = plt.axes()

l1, = ax.plot(t, s1.mean(0), 'b')  # faces
l2, = ax.plot(t, s2.mean(0), 'g')  # scrambled

ax.fill_between(t, ylim[1]*np.ones(t.shape), ylim[0]*np.ones(t.shape), sig,
                facecolor='k', alpha=0.3)
ax.set_xlim((t.min(), t.max()))
ax.set_xlabel('Time (s)')
ax.set_title('Right Amygdala Activation')
ax.legend((l1, l2), ('Faces', 'Scrambled'))
ax.set_ylim(t[0, -1])

plt.show()
