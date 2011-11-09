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

print __doc__

import numpy as np
import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator


data_path = sample.data_path('..')
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

event_id, tmin, tmax = 1, -0.2, 0.5
snr = 3.0
lambda2 = 1.0 / snr ** 2
dSPM = True

# Load data
inverse_operator = read_inverse_operator(fname_inv)
label = mne.read_label(fname_label)
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

# Compute inverse solution and stcs for each epoch
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, dSPM, label,
                            pick_normal=True)

data = sum(stc.data for stc in stcs) / len(stcs)

# compute sign flip to avoid signal cancelation when averaging signed values
flip = mne.label_sign_flip(label, inverse_operator['src'])

label_mean = np.mean(data, axis=0)
label_mean_flip = np.mean(flip[:, np.newaxis] * data, axis=0)

###############################################################################
# View activation time-series
h0 = pl.plot(1e3 * stcs[0].times, data.T, 'k')
h1 = pl.plot(1e3 * stcs[0].times, label_mean, 'r', linewidth=3)
h2 = pl.plot(1e3 * stcs[0].times, label_mean_flip, 'g', linewidth=3)
pl.legend((h0[0], h1, h2), ('all dipoles in label', 'mean',
                            'mean with sign flip'))
pl.xlabel('time (ms)')
pl.ylabel('dSPM value')
pl.show()
