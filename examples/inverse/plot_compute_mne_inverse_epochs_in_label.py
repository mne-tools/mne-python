"""
==================================================
Compute MNE-dSPM inverse solution on single epochs
==================================================

Compute dSPM inverse solution on single trial epochs restricted
to a brain label.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator


data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

event_id, tmin, tmax = 1, -0.2, 0.5
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
inverse_operator = read_inverse_operator(fname_inv)
label = mne.read_label(fname_label)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# Set up pick list
include = []

# Add a bad channel
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick MEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                   include=include, exclude='bads')
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# Compute inverse solution and stcs for each epoch
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, label,
                            pick_ori="normal")

mean_stc = sum(stcs) / len(stcs)

# compute sign flip to avoid signal cancelation when averaging signed values
flip = mne.label_sign_flip(label, inverse_operator['src'])

label_mean = np.mean(mean_stc.data, axis=0)
label_mean_flip = np.mean(flip[:, np.newaxis] * mean_stc.data, axis=0)

###############################################################################
# View activation time-series
plt.figure()
h0 = plt.plot(1e3 * stcs[0].times, mean_stc.data.T, 'k')
h1, = plt.plot(1e3 * stcs[0].times, label_mean, 'r', linewidth=3)
h2, = plt.plot(1e3 * stcs[0].times, label_mean_flip, 'g', linewidth=3)
plt.legend((h0[0], h1, h2), ('all dipoles in label', 'mean',
                             'mean with sign flip'))
plt.xlabel('time (ms)')
plt.ylabel('dSPM value')
plt.show()
