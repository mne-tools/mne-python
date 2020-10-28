"""
.. _tut_compute_covariance:

Computing a covariance matrix
=============================

Many methods in MNE, including source estimation and some classification
algorithms, require covariance estimations from the recordings.
In this tutorial we cover the basics of sensor covariance computations and
construct a noise covariance matrix that can be used when computing the
minimum-norm inverse solution. For more information, see
:ref:`minimum_norm_estimates`.
"""

import os.path as op

import mne
from mne.datasets import sample

###############################################################################
# Source estimation method such as MNE require a noise estimations from the
# recordings. In this tutorial we cover the basics of noise covariance and
# construct a noise covariance matrix that can be used when computing the
# inverse solution. For more information, see :ref:`minimum_norm_estimates`.

data_path = sample.data_path()
raw_empty_room_fname = op.join(
    data_path, 'MEG', 'sample', 'ernoise_raw.fif')
raw_empty_room = mne.io.read_raw_fif(raw_empty_room_fname)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference('average', projection=True)
raw.info['bads'] += ['EEG 053']  # bads + 1 more


###############################################################################
# The definition of noise depends on the paradigm. In MEG it is quite common
# to use empty room measurements for the estimation of sensor noise. However if
# you are dealing with evoked responses, you might want to also consider
# resting state brain activity as noise.
# First we compute the noise using empty room recording. Note that you can also
# use only a part of the recording with tmin and tmax arguments. That can be
# useful if you use resting state as a noise baseline. Here we use the whole
# empty room recording to compute the noise covariance (``tmax=None`` is the
# same as the end of the recording, see :func:`mne.compute_raw_covariance`).
#
# Keep in mind that you want to match your empty room dataset to your
# actual MEG data, processing-wise. Ensure that filters
# are all the same and if you use ICA, apply it to your empty-room and subject
# data equivalently. In this case we did not filter the data and
# we don't use ICA. However, we do have bad channels and projections in
# the MEG data, and, hence, we want to make sure they get stored in the
# covariance object.

raw_empty_room.info['bads'] = [
    bb for bb in raw.info['bads'] if 'EEG' not in bb]
raw_empty_room.add_proj(
    [pp.copy() for pp in raw.info['projs'] if 'EEG' not in pp['desc']])

noise_cov = mne.compute_raw_covariance(
    raw_empty_room, tmin=0, tmax=None)

###############################################################################
# Now that you have the covariance matrix in an MNE-Python object you can
# save it to a file with :func:`mne.write_cov`. Later you can read it back
# using :func:`mne.read_cov`.
#
# You can also use the pre-stimulus baseline to estimate the noise covariance.
# First we have to construct the epochs. When computing the covariance, you
# should use baseline correction when constructing the epochs. Otherwise the
# covariance matrix will be inaccurate. In MNE this is done by default, but
# just to be sure, we define it here manually.
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                    baseline=(-0.2, 0.0), decim=3,  # we'll decimate for speed
                    verbose='error')  # and ignore the warning about aliasing

###############################################################################
# Note that this method also attenuates any activity in your
# source estimates that resemble the baseline, if you like it or not.
noise_cov_baseline = mne.compute_covariance(epochs, tmax=0)

###############################################################################
# Plot the covariance matrices
# ----------------------------
#
# Try setting proj to False to see the effect. Notice that the projectors in
# epochs are already applied, so ``proj`` parameter has no effect.
noise_cov.plot(raw_empty_room.info, proj=True)
noise_cov_baseline.plot(epochs.info, proj=True)

###############################################################################
# .. _plot_compute_covariance_howto:
#
# How should I regularize the covariance matrix?
# ----------------------------------------------
#
# The estimated covariance can be numerically
# unstable and tends to induce correlations between estimated source amplitudes
# and the number of samples available. The MNE manual therefore suggests to
# regularize the noise covariance matrix (see
# :ref:`cov_regularization_math`), especially if only few samples are
# available. Unfortunately it is not easy to tell the effective number of
# samples, hence, to choose the appropriate regularization.
# In MNE-Python, regularization is done using advanced regularization methods
# described in [1]_. For this the 'auto' option can be used. With this
# option cross-validation will be used to learn the optimal regularization:

noise_cov_reg = mne.compute_covariance(epochs, tmax=0., method='auto',
                                       rank=None)

###############################################################################
# This procedure evaluates the noise covariance quantitatively by how well it
# whitens the data using the
# negative log-likelihood of unseen data. The final result can also be visually
# inspected.
# Under the assumption that the baseline does not contain a systematic signal
# (time-locked to the event of interest), the whitened baseline signal should
# be follow a multivariate Gaussian distribution, i.e.,
# whitened baseline signals should be between -1.96 and 1.96 at a given time
# sample.
# Based on the same reasoning, the expected value for the :term:`global field
# power (GFP) <GFP>` is 1 (calculation of the GFP should take into account the
# true degrees of freedom, e.g. ``ddof=3`` with 2 active SSP vectors):

evoked = epochs.average()
evoked.plot_white(noise_cov_reg, time_unit='s')

###############################################################################
# This plot displays both, the whitened evoked signals for each channels and
# the whitened :term:`GFP`. The numbers in the GFP panel represent the
# estimated rank of the data, which amounts to the effective degrees of freedom
# by which the squared sum across sensors is divided when computing the
# whitened :term:`GFP`. The whitened :term:`GFP` also helps detecting spurious
# late evoked components which can be the consequence of over- or
# under-regularization.
#
# Note that if data have been processed using signal space separation
# (SSS) [2]_,
# gradiometers and magnetometers will be displayed jointly because both are
# reconstructed from the same SSS basis vectors with the same numerical rank.
# This also implies that both sensor types are not any longer statistically
# independent.
# These methods for evaluation can be used to assess model violations.
# Additional
# introductory materials can be found `here <https://goo.gl/ElWrxe>`_.
#
# For expert use cases or debugging the alternative estimators can also be
# compared (see
# :ref:`sphx_glr_auto_examples_visualization_plot_evoked_whitening.py`) and
# :ref:`sphx_glr_auto_examples_inverse_plot_covariance_whitening_dspm.py`):

noise_covs = mne.compute_covariance(
    epochs, tmax=0., method=('empirical', 'shrunk'), return_estimators=True,
    rank=None)
evoked.plot_white(noise_covs, time_unit='s')


##############################################################################
# This will plot the whitened evoked for the optimal estimator and display the
# :term:`GFPs <GFP>` for all estimators as separate lines in the related panel.


##############################################################################
# Finally, let's have a look at the difference between empty room and
# event related covariance, hacking the "method" option so that their types
# are shown in the legend of the plot.

evoked_meg = evoked.copy().pick('meg')
noise_cov['method'] = 'empty_room'
noise_cov_baseline['method'] = 'baseline'
evoked_meg.plot_white([noise_cov_baseline, noise_cov], time_unit='s')

##############################################################################
# Based on the negative log-likelihood, the baseline covariance
# seems more appropriate. See :ref:`ex-covariance-whitening-dspm` for more
# information.

###############################################################################
# References
# ----------
#
# .. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
#     covariance estimation and spatial whitening of MEG and EEG signals,
#     vol. 108, 328-342, NeuroImage.
#
# .. [2] Taulu, S., Simola, J., Kajola, M., 2005. Applications of the signal
#    space separation method. IEEE Trans. Signal Proc. 53, 3359-3372.
