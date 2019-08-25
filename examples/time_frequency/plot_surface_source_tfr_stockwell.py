"""
===========================================================================
Compute induced power and itc for SourceEstimates using stockwell transform
===========================================================================

Compute induced power and inter-trial-coherence in source space,
using a multitaper time-frequency transform on a list of SourceEstimate
objects.

"""
# Authors: Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne.datasets import somato
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.time_frequency import tfr_stockwell


print(__doc__)

###############################################################################
# Prepare the data

data_path = somato.data_path()
fname_raw = op.join(data_path, 'sub-01', 'meg', 'sub-01_task-somato_meg.fif')
fname_fwd = op.join(data_path, 'derivatives', 'sub-01', 'sub-01_task-somato-fwd.fif')
subjects_dir = op.join(data_path, 'derivatives', 'freesurfer', 'subjects')

raw = mne.io.read_raw_fif(fname_raw)

# Set picks, use a single sensor type
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

# Read epochs
events = mne.find_events(raw)[:20]  # crop the events to save computation time
tmin, tmax= -0.2, 0.648  # use 256 samples for avoid stockwell zero-padding
epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax, picks=picks)

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', rank=None)

# Read forward operator
fwd = mne.read_forward_solution(fname_fwd)

# make an inverse operator
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
snr = 3.0
lambda2 = 1.0 / snr ** 2

stcs  = apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2,
                             method="dSPM", pick_ori="normal", prepared=False,
                             return_generator=True, delayed=True)

###############################################################################
# Compute power and inter trial coherence, using a stockwell transform.
# We will investigate the alpha band from 8 Hz to 12 Hz.
fmin, fmax = 8, 12
power, itc = tfr_stockwell(stcs, fmin=fmin, fmax=fmax, width=1.2,
                           return_itc=True)

###############################################################################
# As a result, we get SourceTFR objects, a class to store time frequency data
# in source space. The underlying source type is automatically adopted from the
# SourceEstimate type used to create the data.
power

###############################################################################
# Plot the induced power and itc, averaged over the first two stockwell
# frequencies.
fmin, fmax = power.freqs[:2]
initial_time = 0.1

power.plot(fmin=fmin, fmax=fmax, subjects_dir=subjects_dir,
           subject='01', initial_time=initial_time)
itc.plot(fmin=fmin, fmax=fmax, subjects_dir=subjects_dir,
           subject='01', initial_time=initial_time)
