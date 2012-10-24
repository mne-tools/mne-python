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

data_path = sample.data_path('..')
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

snr = 3.0
lambda2 = 1.0 / snr ** 2

# Load data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)
noise_cov = mne.read_cov(fname_cov)

# regularize noise covariance
noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                               mag=0.05, grad=0.05, eeg=0.1, proj=True)


# make an M/EEG, MEG-only, and EEG-only inverse operators
names = ['meg-eeg', 'meg', 'eeg']
meg_bool = [True, True, False]
eeg_bool = [True, False, True]
forwards = [mne.fiff.pick_types_forward(forward, meg=m, eeg=e)
    for m, e in zip(meg_bool, eeg_bool)]

info = evoked.info
inverse_operators = [make_inverse_operator(info, f, noise_cov, loose=0.2,
                                           depth=0.8) for f in forwards]

# Save inverse operators to vizualize with mne_analyze
stcs = [None] * len(names)
for ii, (name, inv) in enumerate(zip(names, inverse_operators)):
    write_inverse_operator('sample_audvis-eeg-oct-6-%s-inv.fif' % name, inv)

    # Compute inverse solution
    stcs[ii] = apply_inverse(evoked, inv, lambda2, "dSPM", pick_normal=False)

    # Save result in stc files
    stcs[ii].save('mne_dSPM_inverse-%s' % name)

###############################################################################
# View activation time-series
pl.close('all')
for ii, (name, stc) in enumerate(zip(names, stcs)):
    pl.subplot(len(stcs), 1, ii)
    pl.plot(1e3 * stc.times, stc.data[::150, :].T)
    pl.ylabel('%s\ndSPM value' % str.upper(name))
pl.xlabel('time (ms)')
pl.show()