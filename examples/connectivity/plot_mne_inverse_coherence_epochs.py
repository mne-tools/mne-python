"""
==============================================================
Compute coherency in source space using a MNE inverse solution
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
from mne.connectivity import idx_seed_con, coherence, freq_connectivity


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
idx = idx_seed_con([seed_idx], np.arange(n_sources))

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list. This allows us so to compute
# the coherence without having to keep all source estimates in memory.

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_normal=True, return_generator=True)

# Now we are ready to compute the coherence
fmin, fmax = 1.0, 30
sfreq = raw.info['sfreq']  # the sampling frequency

# test: custom connectivity method that computes the phase histogram
def _hist_con(csd_xy):
    # compute phase angle 0..2pi and quantize it
    angle = np.angle(csd_xy) + np.pi
    n_bins = 20
    n_sig, n_freq = csd_xy.shape
    hist = np.zeros((n_sig, n_freq, n_bins))

    for i in xrange(n_sig):
        for j in xrange(n_freq):
            bin_idx = int(min(np.floor(angle[i, j] * n_bins / (2 * np.pi)),
                          n_bins - 1))
            hist[i, j, bin_idx] += 1

    return hist


def _hist_norm(hist, psd_xx, psd_yy, n_epochs):
    # Do nothing
    return hist

my_phase_method = (_hist_con, _hist_norm)

con, freqs, n_epochs, n_tapers = freq_connectivity(stcs,
                                                   method=('coh', 'pli', my_phase_method),
                                                   idx=idx, sfreq=sfreq,
                                                   fmin=fmin, fmax=fmax, adaptive=True)

# only get the coherence
coh = np.abs(con[0])

# Generate a source estimate with the coherence. This is simple since we
# used a single seed. For more than one seeds we would have to split coh.
# Note: We use a hack to save the frequency axis as time
tstep = np.mean(np.diff(freqs)) / 1e3
coh_stc = mne.SourceEstimate(coh, vertices=stc.vertno,
                             tmin=freqs[0] / 1e3, tstep=tstep)

# Plot average coherence left-right auditory cortex
aud_rh_coh = np.mean(coh_stc.label_stc(label_rh).data, axis=0)

pl.plot(freqs, aud_rh_coh)
pl.xlabel('Frequency (Hz)')
pl.ylabel('Coherence left-right auditory cortex')
pl.show()

# We could save the coherence, for visualization using e.g. mne_analyze
coh_stc.save('seed_coh_vertno_%d' % seed_vertno)
