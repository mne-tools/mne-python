"""
=============================================
Compute sLORETA inverse solution on raw data
=============================================

Compute sLORETA inverse solution on raw dataset restricted
to a brain label and stores the solution in stc files for
visualisation.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_raw, read_inverse_operator

print(__doc__)

data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "sLORETA"  # use sLORETA method (could also be MNE or dSPM)

# Load data
raw = mne.io.read_raw_fif(fname_raw)
inverse_operator = read_inverse_operator(fname_inv)
label = mne.read_label(fname_label)

raw.set_eeg_reference()  # set average reference.
start, stop = raw.time_as_index([0, 15])  # read the first 15s of data

# Compute inverse solution
stc = apply_inverse_raw(raw, inverse_operator, lambda2, method, label,
                        start, stop, pick_ori=None)

# Save result in stc files
stc.save('mne_%s_raw_inverse_%s' % (method, label_name))

###############################################################################
# View activation time-series
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()
