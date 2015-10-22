"""
=======================================
Compute spatial filter based on DeFleCT
=======================================

Compute spatial filters obeying a combination of several constraints in the
DeFleCT framework. Apply spatial filters to evoked data and plot SNR curves.
Apply spatial filters to epoched data and compute between-labels spectral
connectivity.

The constraints that can be implemented are:
- Suppress cross-talk within a small set of labels
- Minimize cross-talk from other locations anywhere in the brain
- Minimize the effect of noise (represented by noise covariance matrix).

Different sensor types are combined by whitening with noise covariance matrix.
First label is the target. Cross-talk with other labels should be zero. 
Cross-talk with other vertices will be minimized.
For reference, see http://www.ncbi.nlm.nih.gov/pubmed/23616402
"""

# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

print(__doc__)

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import DeFleCT, read_inverse_operator
from mne.source_estimate import SourceEstimate
import numpy as np
import scipy as sp


### SPECIFY PATHS AND FILES

# path to sample data set
data_path = sample.data_path()
# forward solution
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
# noise covariance matrix
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'

# evoked data for SNR computation
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

# raw data for functional connectivity
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# 4 MNE-sample labels, for which to construct spatial filters
fname_label = [data_path + '/MEG/sample/labels/Aud-rh.label',
               data_path + '/MEG/sample/labels/Aud-lh.label',
               data_path + '/MEG/sample/labels/Vis-rh.label',
               data_path + '/MEG/sample/labels/Vis-lh.label',
               ]
nr_lbls = len( fname_label )


#### SPECIFY ANALYSIS PARAMETERS

# channel types
pick_meg = True # or True/False/'grad'/'mag'
pick_eeg = False

# filename for cross-talk-function (CTF) output as STC file
# CTFs for separate estimators as different time samples
stc_fname_out = 'deflect_ctf'

# for epoching raw data before connectivity analysis
event_id, tmin, tmax = 1, -0.2, 0.5
if (pick_eeg and pick_meg):
    reject = dict(mag=4e-12, grad=4000e-13, eeg=120e-6, eog=150e-6)
if pick_eeg and ~pick_meg:
    reject = dict(eeg=120e-6, eog=150e-6)
if ~pick_eeg and pick_meg:
    reject = dict(mag=4e-12, grad=4000e-13, eog=150e-6)

# parameters for summarising sub-leadfields

# how to summarise sub-leadfields for labels ('svd', 'sum', 'mean')
mode = 'svd'
# how many SVD components per label/sub-leadfield (list)
# 1 for first recommended, others may depend on size of label
n_svd_comp = [1,2,2,2]

SNR = 3
lambda2_S = 1. / SNR**2.      # reg param for Gram matrix S in DeFleCT
# reg params for covariance matrix
lambda2_cov = {'mag': 1./10., 'gra': 1./10., 'eeg': 1./10.}

#### READ FORWARD SOLUTION, NOISE COVMAT AND LABELS

# read forward solution (sources in surface-based coordinates)
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)
forw_ch_names = forward['info']['ch_names']

# read noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# read label(s)
labels = [mne.read_label(ss) for ss in fname_label]


#### COMPUTE DeFleCT ESTIMATORS
spatial_filters = list()
names = [ll.name for ll in labels]    # for later labelling of plots etc.

# 1st estimator
# original sequence, Aud-rh first
labels_use = [ labels[perm] for perm in [0,1,2,3] ]
w, ch_names, F, P, noise_cov_reg, _ = DeFleCT.DeFleCT_make_estimator(forward,
                           noise_cov, labels_use, lambda2_cov, lambda2_S,
                           pick_meg, pick_eeg, mode='svd',
                           n_svd_comp=n_svd_comp, verbose=None)
spatial_filters.append(w)

# 2nd estimator
labels_use = [ labels[perm] for perm in [1,0,2,3] ] # Aud-lh first
w, ch_names, F, P, _, _ = DeFleCT.deflect_make_estimator(forward,
                           noise_cov, labels_use, lambda2_cov, lambda2_S,
                           pick_meg, pick_eeg, mode='svd',
                           n_svd_comp=n_svd_comp, verbose=None)
spatial_filters.append(w)

# 3rd estimator
labels_use = [ labels[perm] for perm in [2,0,1,3] ] # Vis-rh first
w, ch_names, F, P, _, _ = DeFleCT.deflect_make_estimator(forward,
                           noise_cov, labels_use, lambda2_cov, lambda2_S,
                           pick_meg, pick_eeg, mode='svd',
                           n_svd_comp=n_svd_comp, verbose=None)
spatial_filters.append(w)

# 4th estimator
labels_use = [ labels[perm] for perm in [3,0,1,2] ] # Vis-lh first
w, ch_names, F, P, _, _ = DeFleCT.deflect_make_estimator(forward,
                           noise_cov, labels_use, lambda2_cov, lambda2_S,
                           pick_meg, pick_eeg, mode='svd',
                           n_svd_comp=n_svd_comp, verbose=None)
spatial_filters.append(w)

n_filter = len(spatial_filters)

noise_cov_mat = noise_cov_reg['data']

#### COMPUTE CTFs (to check if DeFleCT produced desired spatial filters)
ctfs = [w.dot(F) for w in spatial_filters]
# normalize to maximum for easier display
ctfs = [ctf/np.abs(ctf).max() for ctf in ctfs]

# create source estimate object
vertno = [ss['vertno'] for ss in forward['src']]
stc_ctf = SourceEstimate(np.squeeze(ctfs).T, vertno, tmin=0., tstep=0.001)

# save CTF as STC file for mne_analyze
print "STC filename for CTFs: %s " % stc_fname_out
stc_ctf.save(stc_fname_out)


#### APPLY TO EVOKED DATA AND COMPUTE SNRs
# (e.g. to check whether regularization is reasonable)
# (assumes as many conditions in evoked as spatial filters in spatial_filters)

# read and prepare evoked data
evoked = [mne.read_evokeds(fname_evoked, condition=cc, baseline=(None, 0),
                            proj=True) for cc in range(n_filter)]
evoked = [mne.pick_channels_evoked(ee, ch_names) for ee in evoked]
evoked_mat = [ee.data for ee in evoked]

# Apply estimator to data
evoked_tc = [spatial_filters[ff].dot(evoked_mat[ff]) for ff in range(n_filter)]

# Compute SNR as ration with standard deviation of baseline (to 0ms)
base_idx = [np.abs(ee.times).argmin() for ee in evoked]
evoked_tc_snr = [evoked_tc[ff] / np.std(evoked_tc[:base_idx[ff]]) for ff
                                                            in range(n_filter)]

print "\n Evoked data:"
for ff in np.arange(len(spatial_filters)):
    print "Max SNR (std to baseline) %f: " % np.abs(evoked_tc_snr[ff]).max()
print "\n"


### APPLY TO EPOCHED DATA, COMPUTE CONNECTIVITY BETWEEN LABELS

# Load raw data
raw = mne.io.Raw(fname_raw, preload=True)
events = mne.read_events(fname_event)
picks = mne.pick_types(raw.info, meg=pick_meg, eeg=pick_eeg, eog=True,
                                                    stim=False, exclude='bads')
# Define epochs for left-auditory condition
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, proj=True,
                    baseline=(None, 0), reject=reject, preload=True)

# apply spatial filters to epochs
label_tc = DeFleCT.apply_spatial_filters_epochs(spatial_filters, epochs)

fmin, fmax = 5., 40.       # min/max frequencies for coherence 
sfreq = raw.info['sfreq']  # the sampling frequency of data

# spectral connectivity
con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
              label_tc, method='wpli2_debiased', mode='multitaper', sfreq=sfreq,
                               fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=1)

n_rows, n_cols = con.shape[:2]
fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
axes[0, 0].set_ylim([-.1, 0.8])
plt.suptitle('Between labels connectivity')
for i in range(n_rows):
    for j in range(i + 1):
        if i == j:
            axes[i, j].set_axis_off()
            continue

        axes[i, j].plot(freqs, con[i, j, :])
        axes[j, i].plot(freqs, con[i, j, :])

        if j == 0:
            axes[i, j].set_ylabel(names[i])
            axes[0, i].set_title(names[i])
        if i == (n_rows - 1):
            axes[i, j].set_xlabel(names[j])
        axes[i, j].set_xlim([fmin, fmax])
        axes[j, i].set_xlim([fmin, fmax])

        # Show band limits
        for f in [8, 12, 18, 35]:
            axes[i, j].axvline(f, color='k')
            axes[j, i].axvline(f, color='k')
plt.show()

# Done