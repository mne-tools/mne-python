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
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.time_frequency import tfr_morlet
from mne.source_space import setup_volume_source_space
from mne.forward import make_forward_solution


print(__doc__)

###############################################################################
# Prepare the data

# Set dir
data_path = sample.data_path()
subject = 'sample'
data_dir = op.join(data_path, 'MEG', subject)
subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')

# Set file names
fname_aseg = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')

fname_model = op.join(bem_dir, '%s-5120-bem.fif' % subject)
fname_bem = op.join(bem_dir, '%s-5120-bem-sol.fif' % subject)

fname_raw = op.join(data_dir, 'sample_audvis_raw.fif')
fname_trans = op.join(data_dir, 'sample_audvis_raw-trans.fif')
fname_cov = op.join(data_dir, 'ernoise-cov.fif')
fname_event = op.join(data_dir, 'sample_audvis_filt-0-40_raw-eve.fif')

# set up a volume source space
src = setup_volume_source_space(
    subject, mri=fname_aseg, pos=10.0, bem=fname_model,
    add_interpolator=False,  # just for speed, usually use True
    subjects_dir=subjects_dir)

# compute the fwd matrix
fwd = make_forward_solution(fname_raw, fname_trans, src, fname_bem,
                            mindist=5.0, meg=True, eeg=False, n_jobs=1)

# read data
raw = mne.io.read_raw_fif(fname_raw)
noise_cov = mne.read_cov(fname_cov)

# pick types and remove bads
raw.info['bads'] += ['MEG 2443']
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Read epochs
events = mne.find_events(raw)[:20]  # crop the events to save computation time
tmin, tmax = -0.5, 1.5
epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax, picks=picks)

# make the inverse operator
inv = make_inverse_operator(epochs.info, fwd, noise_cov,
                            depth=None, fixed=False)

###############################################################################
# Apply the Inverse solution.
# If we set return_generator to True, the time frequency transform will process
# each element of stcs subsequently. This will reduce the memory load.
# If there are more dipoles than epochs, `delayed=True` can help reducing the
# computational time, allowing the time frequency transform to be calculated
# on the channels instead of the dipoles. However, since we have fewer dipoles
# than channels, `delayed=False` will be faster.
snr = 3.0
lambda2 = 1.0 / snr ** 2

stcs = apply_inverse_epochs(epochs, inv, lambda2=lambda2, method="dSPM",
                            pick_ori="normal", prepared=False,
                            return_generator=True, delayed=True)

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
# Make a volume plot of the induced power, averaged for the lower beta band
# (12 Hz - 16 Hz).
initial_time = 0.05
fmin, fmax = 12, 16
power.plot(fmin=fmin, fmax=fmax, src=src, subject=subject,
           subjects_dir=subjects_dir, transparent=True)
