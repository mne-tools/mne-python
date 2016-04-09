"""
=====================================================================
Compute Power Spectral Density of inverse solution from single epochs
=====================================================================

Compute PSD of dSPM inverse solution on single trial epochs restricted
to a brain label. The PSD is computed using a multi-taper method with
Discrete Prolate Spheroidal Sequence (DPSS) windows.

"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs

print(__doc__)

data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

event_id, tmin, tmax = 1, -0.2, 0.5
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
inverse_operator = read_inverse_operator(fname_inv)
label = mne.read_label(fname_label)
raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)

# Set up pick list
include = []
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       include=include, exclude='bads')
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# define frequencies of interest
fmin, fmax = 0., 70.
bandwidth = 4.  # bandwidth of the windows in Hz

# compute source space psd in label

# Note: By using "return_generator=True" stcs will be a generator object
# instead of a list. This allows us so to iterate without having to
# keep everything in memory.

stcs = compute_source_psd_epochs(epochs, inverse_operator, lambda2=lambda2,
                                 method=method, fmin=fmin, fmax=fmax,
                                 bandwidth=bandwidth, label=label,
                                 return_generator=True)

# compute average PSD over the first 10 epochs
n_epochs = 10
for i, stc in enumerate(stcs):
    if i >= n_epochs:
        break

    if i == 0:
        psd_avg = np.mean(stc.data, axis=0)
    else:
        psd_avg += np.mean(stc.data, axis=0)

psd_avg /= n_epochs
freqs = stc.times  # the frequencies are stored here

plt.figure()
plt.plot(freqs, psd_avg)
plt.xlabel('Freq (Hz)')
plt.ylabel('Power Spectral Density')
plt.show()
