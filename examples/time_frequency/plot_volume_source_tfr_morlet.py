"""
==========================================================================
Compute induced power and itc for VolSourceEstimates using morlet wavelets
==========================================================================

Compute induced power and in source space, using a morlet wavelet
transform on a list of VolSourceEstimate objects.

"""
# Authors: Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

import mne
from mne.datasets import somato
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.time_frequency import tfr_morlet
from mne.source_space import setup_volume_source_space
from mne.forward import make_forward_solution


print(__doc__)

###############################################################################
# Prepare the data
data_path = somato.data_path()
fname_raw = op.join(data_path, 'MEG', 'somato', 'sef_raw_sss.fif')
fname_trans = op.join(data_path, 'MEG', 'somato', 'sef_raw_sss-trans.fif')

subject = 'somato'
subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')
fname_aseg = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')
fname_model = op.join(bem_dir, '%s-5120-bem.fif' % subject)
fname_bem = op.join(bem_dir, '%s-5120-bem-sol.fif' % subject)

raw = mne.io.read_raw_fif(fname_raw)

# Set picks, use a single sensor type
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# Read epochs
events = mne.find_events(raw)[:20]  # crop the events to save computation time
tmin, tmax= -0.5, 1.5
epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax, picks=picks)

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', rank=None)

# setup a volume source space of the left cortex
labels_vol = 'Brain-Stem'
src = setup_volume_source_space( subject, mri=fname_aseg, pos=10.0,
                                 bem=fname_model, volume_label=labels_vol,
                                 subjects_dir=subjects_dir)

# compute the fwd matrix
fwd = make_forward_solution(fname_raw, fname_trans, src, fname_bem,
                            mindist=5.0, n_jobs=1)

# make an inverse operator
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         depth=None, fixed=False)

###############################################################################
# Apply the Inverse solution.
# If we set return_generator to True, the time frequency transform will process
# each element of stcs subsequently. This will reduce the memory load.
# If there are more dipoles than epochs, `delayed=True` can help reducing the
# computational time, allowing the time frequency transform to be calculated
# on the channels instead of the dipoles. However, since we have fewer dipoles
# than channels, `delayed=False` will be faster.
stcs  = apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2,
                             method="dSPM", pick_ori="normal", prepared=False,
                             return_generator=True, delayed=False)

###############################################################################
# Compute the average power, using a morlet wavelet analysis.
# We will investigate the beta band from 12 Hz to 30 Hz, in steps of 3 Hz.
freqs = np.arange(12, 30, 3)
power = tfr_morlet(stcs, freqs=freqs, n_cycles=4, use_fft=True,
                       average=True, return_itc=False)

###############################################################################
# As a result, we get SourceTFR objects, a class to store time frequency data
# in source space. The underlying source type is automatically adopted from the
# SourceEstimate type used to create the data.
power

###############################################################################
# Make a volume plot of the induced power, averaged over all used frequencies.
initial_time = 0.05
fmin, fmax = 4, 8
power.plot(fmin=4, fmax=12, src=src, subject=subject,
           subjects_dir=subjects_dir, transparent=True)
