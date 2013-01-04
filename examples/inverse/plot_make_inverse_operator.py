"""
===============================================================
Assemble inverse operator and compute MNE-dSPM inverse solution
===============================================================

Assemble M/EEG, MEG, and EEG inverse operators and compute dSPM
inverse solution on MNE evoked dataset and stores the solution
in stc files for visualisation.

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
import mne
from mne.datasets import sample
from mne.fiff import Evoked
from mne.minimum_norm import make_inverse_operator, apply_inverse, \
                             write_inverse_operator

data_path = sample.data_path()
fname_fwd_meeg = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_fwd_eeg = data_path + '/MEG/sample/sample_audvis-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

snr = 3.0
lambda2 = 1.0 / snr ** 2

# Load data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
forward_meeg = mne.read_forward_solution(fname_fwd_meeg, surf_ori=True)
noise_cov = mne.read_cov(fname_cov)

# regularize noise covariance
noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                               mag=0.05, grad=0.05, eeg=0.1, proj=True)

# Restrict forward solution as necessary for MEG
forward_meg = mne.fiff.pick_types_forward(forward_meeg, meg=True, eeg=False)
# Alternatively, you can just load a forward solution that is restricted
forward_eeg = mne.read_forward_solution(fname_fwd_eeg, surf_ori=True)

# make an M/EEG, MEG-only, and EEG-only inverse operators
info = evoked.info
inverse_operator_meeg = make_inverse_operator(info, forward_meeg, noise_cov,
                                              loose=0.2, depth=0.8)
inverse_operator_meg = make_inverse_operator(info, forward_meg, noise_cov,
                                              loose=0.2, depth=0.8)
inverse_operator_eeg = make_inverse_operator(info, forward_eeg, noise_cov,
                                              loose=0.2, depth=0.8)

write_inverse_operator('sample_audvis-meeg-oct-6-inv.fif',
                       inverse_operator_meeg)
write_inverse_operator('sample_audvis-meg-oct-6-inv.fif',
                       inverse_operator_meg)
write_inverse_operator('sample_audvis-eeg-oct-6-inv.fif',
                       inverse_operator_eeg)

# Compute inverse solution
stcs = dict()
stcs['meeg'] = apply_inverse(evoked, inverse_operator_meeg, lambda2, "dSPM",
                        pick_normal=False)
stcs['meg'] = apply_inverse(evoked, inverse_operator_meg, lambda2, "dSPM",
                        pick_normal=False)
stcs['eeg'] = apply_inverse(evoked, inverse_operator_eeg, lambda2, "dSPM",
                        pick_normal=False)

# Save result in stc files
names = ['meeg', 'meg', 'eeg']
for name in names:
    stcs[name].save('mne_dSPM_inverse-%s' % name)

###############################################################################
# View activation time-series
pl.close('all')
pl.figure(figsize=(8, 6))
for ii in range(len(stcs)):
    name = names[ii]
    stc = stcs[name]
    pl.subplot(len(stcs), 1, ii + 1)
    pl.plot(1e3 * stc.times, stc.data[::150, :].T)
    pl.ylabel('%s\ndSPM value' % str.upper(name))
pl.xlabel('time (ms)')
pl.show()