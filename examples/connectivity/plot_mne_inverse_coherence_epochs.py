"""
==============================================================
Compute coherence in source space using a MNE inverse solution
==============================================================

This examples computes the coherence between a seed in the left
auditory cortex and the rest of the brain based on single-trial
MNE-dSPM inverse soltions.

"""

# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import apply_inverse, apply_inverse_epochs,\
                             read_inverse_operator
from mne.connectivity import seed_target_indices, spectral_connectivity


data_path = sample.data_path('..')
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
label_name_lh = 'Aud-lh'
fname_label_lh = data_path + '/MEG/sample/labels/%s.label' % label_name_lh
label_name_rh = 'Aud-rh'
fname_label_rh = data_path + '/MEG/sample/labels/%s.label' % label_name_rh

event_id, tmin, tmax = 1, -0.2, 0.5
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
inverse_operator = read_inverse_operator(fname_inv)
label_lh = mne.read_label(fname_label_lh)
label_rh = mne.read_label(fname_label_rh)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# Set up pick list
include = []
exclude = raw.info['bads'] + ['EEG 053']  # bads + 1 more

# pick MEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                   include=include, exclude=exclude)
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# First, we find the most active vertex in the left auditory cortex, which
# we will later use as seed for the connectivity computation
snr = 3.0
lambda2 = 1.0 / snr ** 2
evoked = epochs.average()
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_normal=True)

# Restrict the source estimate to the label in the left auditory cortex
stc_label = stc.label_stc(label_lh)

# Find number and index of vertex with most power
src_pow = np.sum(stc_label.data ** 2, axis=1)
seed_vertno = stc_label.vertno[0][np.argmax(src_pow)]
seed_idx = np.searchsorted(stc.vertno[0], seed_vertno)  # index in original stc

# Generate index parameter for seed-based connectivity analysis
n_sources = stc.data.shape[0]
indices = seed_target_indices([seed_idx], np.arange(n_sources))

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list. This allows us so to compute
# the coherence without having to keep all source estimates in memory.

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_normal=True, return_generator=True)

# Now we are ready to compute the coherence in the alpha and beta band
fmin, fmax = (8., 20.), (13., 30.)
sfreq = raw.info['sfreq']  # the sampling frequency

# Now we compute connectivity. To speed things up, we use 2 parallel jobs
# and do not use adaptive weights for the multi-taper spectral estimation.
# By using faverage=True, we directly average the coherence in the alpha
# and beta band, i.e., we will only get 2 frequency bins
coh, freqs, n_epochs, n_tapers = spectral_connectivity(stcs,
    method='coh', indices=indices, sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=True, adaptive=False, n_jobs=2)

print 'Frequencies in Hz over which coherence was averaged for alpha: '
print freqs[0]
print 'Frequencies in Hz over which coherence was averaged for beta: '
print freqs[1]

# Generate a source estimate with the coherence. This is simple since we
# used a single seed. For more than one seeds we would have to split coh.
# Note: We use a hack to save the frequency axis as time
aud_rh_coh = dict()  # store the coherence for each band
for i, band in enumerate(['alpha', 'beta']):
    tstep = np.mean(np.diff(freqs[i])) / 1e3

    coh_stc = mne.SourceEstimate(coh[:, i][:, None], vertices=stc.vertno,
        tmin=1e-3 * np.mean(freqs[i]), tstep=1)

    # save the cohrence to plot later
    aud_rh_coh[band] = np.mean(coh_stc.label_stc(label_rh).data, axis=0)

    # We could save the coherence, for visualization using e.g. mne_analyze
    #coh_stc.save('seed_coh_%s_vertno_%d' % (band, seed_vertno))

pl.figure()
width = 0.5
pos = np.arange(2)
pl.bar(pos, [aud_rh_coh['alpha'], aud_rh_coh['beta']], width)
pl.ylabel('Coherence')
pl.title('Cohrence left-right auditory')
pl.xticks(pos + width / 2, ('alpha', 'beta'))
pl.show()

