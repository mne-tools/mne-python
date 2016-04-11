"""
.. _tut_compute_covariance:

Computing covariance matrix
===========================
"""
import os.path as op

import mne
from mne.datasets import sample

###############################################################################
# Source estimation method such as MNE require a noise estimations from the
# recordings. In this tutorial we cover the basics of noise covariance and
# construct a noise covariance matrix that can be used when computing the
# inverse solution. For more information, see :ref:`BABDEEEB`.

data_path = sample.data_path()
erm_fname = op.join(data_path, 'MEG', 'sample', 'ernoise_raw.fif')
erm = mne.io.read_raw_fif(erm_fname)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname)

###############################################################################
# The definition of noise depends on the paradigm. In MEG it is quite common
# to use empty room measurements for the estimation of sensor noise. However if
# you are dealing with evoked responses, you might want to also consider
# resting state brain activity as noise.
# First we compute the noise using empty room recording. Note that you can also
# use only a part of the recording with tmin and tmax arguments. That can be
# useful if you use resting state as a noise baseline. Here we use the whole
# empty room recording to compute the noise covariance (tmax=None is the same
# as the end of the recording, see :func:`mne.compute_raw_covariance`).

noise_cov_erm = mne.compute_raw_covariance(erm, tmin=0, tmax=None)

###############################################################################
# You can also use the pre-stimulus baseline to estimate the noise covariance.
# First we have to construct the epochs. When computing the covariance, you
# should use baseline correction when constructing the epochs. Otherwise the
# covariance matrix will be inaccurate. In MNE this is done by default, but
# just to be sure, we define it here manually.

events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.0,
                    baseline=(-0.2, 0.0))

###############################################################################
# Note that this method also attenuates the resting state activity in your
# source estimates.

noise_cov_baseline = mne.compute_covariance(epochs)

###############################################################################
# Plot the covariance matrices.

noise_cov_erm.plot(erm.info)
noise_cov_baseline.plot(epochs.info)
