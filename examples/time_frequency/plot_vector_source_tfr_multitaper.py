"""
=================================================================
Compute induced power for VectorSourceEstimates using multitapers
=================================================================

Compute induced power for multiple epochs in source space,
using a multitaper time-frequency transform on a list of source
estimate objects.

"""
# Authors: Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

import mne
from mne.datasets import somato
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.time_frequency import tfr_multitaper


print(__doc__)

###############################################################################
# Prepare the data

data_path = somato.data_path()
raw_fname = op.join(data_path, 'MEG', 'somato', 'sef_raw_sss.fif')
fname_fwd = op.join(data_path, 'MEG', 'somato', 'somato-meg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')

raw = mne.io.read_raw_fif(raw_fname)

# Set picks, use a single sensor type
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# Read epochs
events = mne.find_events(raw)[:5]  # crop the events to save computation time
tmin, tmax= -0.2, 1
epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax, picks=picks)

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', rank=None)

# Read forward operator
fwd = mne.read_forward_solution(fname_fwd)

# make an inverse operator
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'
inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

###############################################################################
# Apply the Inverse solution.
# If we set return_generator to True, the time frequency transform will process
# each element of stcs subsequently. This will reduce the memory load.
# We can also reduce computational time drastically by setting delayed to True.
# This will allow the time frequency transform to be computed on the sensor
# space data, and then automatically project it into source space.
# .. note:: If you call `stc.data` for any stc in stcs, before doing the time
#           frequency transform, the full source time series will be computed.
#           In this case, the time frequency transform will be calculated for
#           each dipole instead of each sensor (more time consuming).
stcs  = apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2,
                             method="dSPM", pick_ori="vector", prepared=False,
                             return_generator=True, delayed=True)

###############################################################################
# Compute the power for all epochs, using a multitaper analysis.
# We will investigate the alpha band from 8 Hz to 12 Hz, in steps of 1 Hz.
freqs = np.arange(8, 12)
power = tfr_multitaper(stcs, freqs=freqs, n_cycles=7, use_fft=True,
                       average=False, return_itc=False)

###############################################################################
# As a result, we get SourceTFR objects, a class to store time frequency data
# in source space. The underlying source type is automatically adopted from the
# SourceEstimate type used to create the data.
power

###############################################################################
# Plot the induced power for the first two epochs, for the frequency of 8 Hz.
initial_time = 0.05
epoch = 2
fmin, fmax = 8, 8

for epoch_idx in range(2):
    power.plot(epoch=epoch_idx, fmin=fmin, fmax=fmax,
               subjects_dir=subjects_dir, initial_time=initial_time,
               scale_factor=10)




