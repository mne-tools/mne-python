# -*- coding: utf-8 -*-
r"""
.. _tut-mvpa:

===============
Decoding (MVPA)
===============

.. include:: ../../links.inc

Design philosophy
=================
Decoding (a.k.a. MVPA) in MNE largely follows the machine
learning API of the scikit-learn package.
Each estimator implements ``fit``, ``transform``, ``fit_transform``, and
(optionally) ``inverse_transform`` methods. For more details on this design,
visit scikit-learn_. For additional theoretical insights into the decoding
framework in MNE :footcite:`KingEtAl2018`.

For ease of comprehension, we will denote instantiations of the class using
the same name as the class but in small caps instead of camel cases.

Let's start by loading data for a simple two-class problem:
"""

# %%
# sphinx_gallery_thumbnail_number = 6

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

data_path = sample.data_path()

subjects_dir = data_path / 'subjects'
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
tmin, tmax = -0.200, 0.500
event_id = {'Auditory/Left': 1, 'Visual/Left': 3}  # just use two
raw = mne.io.read_raw_fif(raw_fname)
raw.pick_types(meg='grad', stim=True, eog=True, exclude=())

# The subsequent decoding analyses only capture evoked responses, so we can
# low-pass the MEG data. Usually a value more like 40 Hz would be used,
# but here low-pass at 20 so we can more heavily decimate, and allow
# the example to run faster. The 2 Hz high-pass helps improve CSP.
raw.load_data().filter(2, 20)
events = mne.find_events(raw, 'STI 014')

# Set up bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443']  # bads + 2 more

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('grad', 'eog'), baseline=(None, 0.), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6), decim=3,
                    verbose='error')
epochs.pick_types(meg=True, exclude='bads')  # remove stim and EOG
del raw

X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
y = epochs.events[:, 2]  # target: auditory left vs visual left

# %%
# Transformation classes
# ======================
#
# Scaler
# ^^^^^^
# The :class:`mne.decoding.Scaler` will standardize the data based on channel
# scales. In the simplest modes ``scalings=None`` or ``scalings=dict(...)``,
# each data channel type (e.g., mag, grad, eeg) is treated separately and
# scaled by a constant. This is the approach used by e.g.,
# :func:`mne.compute_covariance` to standardize channel scales.
#
# If ``scalings='mean'`` or ``scalings='median'``, each channel is scaled using
# empirical measures. Each channel is scaled independently by the mean and
# standand deviation, or median and interquartile range, respectively, across
# all epochs and time points during :class:`~mne.decoding.Scaler.fit`
# (during training). The :meth:`~mne.decoding.Scaler.transform` method is
# called to transform data (training or test set) by scaling all time points
# and epochs on a channel-by-channel basis. To perform both the ``fit`` and
# ``transform`` operations in a single call, the
# :meth:`~mne.decoding.Scaler.fit_transform` method may be used. To invert the
# transform, :meth:`~mne.decoding.Scaler.inverse_transform` can be used. For
# ``scalings='median'``, scikit-learn_ version 0.17+ is required.
#
# .. note:: Using this class is different from directly applying
#           :class:`sklearn.preprocessing.StandardScaler` or
#           :class:`sklearn.preprocessing.RobustScaler` offered by
#           scikit-learn_. These scale each *classification feature*, e.g.
#           each time point for each channel, with mean and standard
#           deviation computed across epochs, whereas
#           :class:`mne.decoding.Scaler` scales each *channel* using mean and
#           standard deviation computed across all of its time points
#           and epochs.
#
# Vectorizer
# ^^^^^^^^^^
# Scikit-learn API provides functionality to chain transformers and estimators
# by using :class:`sklearn.pipeline.Pipeline`. We can construct decoding
# pipelines and perform cross-validation and grid-search. However scikit-learn
# transformers and estimators generally expect 2D data
# (n_samples * n_features), whereas MNE transformers typically output data
# with a higher dimensionality
# (e.g. n_samples * n_channels * n_frequencies * n_times). A Vectorizer
# therefore needs to be applied between the MNE and the scikit-learn steps
# like:

# Uses all MEG sensors and time points as separate classification
# features, so the resulting filters used are spatio-temporal
clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LogisticRegression(solver='liblinear')  # liblinear is faster than lbfgs
)

scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None)

# Mean scores across cross-validation splits
score = np.mean(scores, axis=0)
print('Spatio-temporal: %0.1f%%' % (100 * score,))

# %%
# PSDEstimator
# ^^^^^^^^^^^^
# The :class:`mne.decoding.PSDEstimator`
# computes the power spectral density (PSD) using the multitaper
# method. It takes a 3D array as input, converts it into 2D and computes the
# PSD.
#
# FilterEstimator
# ^^^^^^^^^^^^^^^
# The :class:`mne.decoding.FilterEstimator` filters the 3D epochs data.
#
# Spatial filters
# ===============
#
# Just like temporal filters, spatial filters provide weights to modify the
# data along the sensor dimension. They are popular in the BCI community
# because of their simplicity and ability to distinguish spatially-separated
# neural activity.
#
# Common spatial pattern
# ^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`mne.decoding.CSP` is a technique to analyze multichannel data based
# on recordings from two classes :footcite:`Koles1991` (see also
# https://en.wikipedia.org/wiki/Common_spatial_pattern).
#
# Let :math:`X \in R^{C\times T}` be a segment of data with
# :math:`C` channels and :math:`T` time points. The data at a single time point
# is denoted by :math:`x(t)` such that :math:`X=[x(t), x(t+1), ..., x(t+T-1)]`.
# Common spatial pattern (CSP) finds a decomposition that projects the signal
# in the original sensor space to CSP space using the following transformation:
#
# .. math::       x_{CSP}(t) = W^{T}x(t)
#    :label: csp
#
# where each column of :math:`W \in R^{C\times C}` is a spatial filter and each
# row of :math:`x_{CSP}` is a CSP component. The matrix :math:`W` is also
# called the de-mixing matrix in other contexts. Let
# :math:`\Sigma^{+} \in R^{C\times C}` and :math:`\Sigma^{-} \in R^{C\times C}`
# be the estimates of the covariance matrices of the two conditions.
# CSP analysis is given by the simultaneous diagonalization of the two
# covariance matrices
#
# .. math::       W^{T}\Sigma^{+}W = \lambda^{+}
#    :label: diagonalize_p
# .. math::       W^{T}\Sigma^{-}W = \lambda^{-}
#    :label: diagonalize_n
#
# where :math:`\lambda^{C}` is a diagonal matrix whose entries are the
# eigenvalues of the following generalized eigenvalue problem
#
# .. math::      \Sigma^{+}w = \lambda \Sigma^{-}w
#    :label: eigen_problem
#
# Large entries in the diagonal matrix corresponds to a spatial filter which
# gives high variance in one class but low variance in the other. Thus, the
# filter facilitates discrimination between the two classes.
#
# .. topic:: Examples
#
#     * :ref:`ex-decoding-csp-eeg`
#     * :ref:`ex-decoding-csp-eeg-timefreq`
#
# .. note::
#
#     The winning entry of the Grasp-and-lift EEG competition in Kaggle used
#     the :class:`~mne.decoding.CSP` implementation in MNE and was featured as
#     a `script of the week <sotw_>`_.
#
# .. _sotw: http://blog.kaggle.com/2015/08/12/july-2015-scripts-of-the-week/
#
# We can use CSP with these data with:

csp = CSP(n_components=3, norm_trace=False)
clf_csp = make_pipeline(
    csp,
    LinearModel(LogisticRegression(solver='liblinear'))
)
scores = cross_val_multiscore(clf_csp, X, y, cv=5, n_jobs=None)
print('CSP: %0.1f%%' % (100 * scores.mean(),))

# %%
# Source power comodulation (SPoC)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Source Power Comodulation (:class:`mne.decoding.SPoC`)
# :footcite:`DahneEtAl2014` identifies the composition of
# orthogonal spatial filters that maximally correlate with a continuous target.
#
# SPoC can be seen as an extension of the CSP where the target is driven by a
# continuous variable rather than a discrete variable. Typical applications
# include extraction of motor patterns using EMG power or audio patterns using
# sound envelope.
#
# .. topic:: Examples
#
#     * :ref:`ex-spoc-cmc`
#
# xDAWN
# ^^^^^
# :class:`mne.preprocessing.Xdawn` is a spatial filtering method designed to
# improve the signal to signal + noise ratio (SSNR) of the ERP responses
# :footcite:`RivetEtAl2009`. Xdawn was originally
# designed for P300 evoked potential by enhancing the target response with
# respect to the non-target response. The implementation in MNE-Python is a
# generalization to any type of ERP.
#
# .. topic:: Examples
#
#     * :ref:`ex-xdawn-denoising`
#     * :ref:`ex-xdawn-decoding`
#
# Effect-matched spatial filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The result of :class:`mne.decoding.EMS` is a spatial filter at each time
# point and a corresponding time course :footcite:`SchurgerEtAl2013`.
# Intuitively, the result gives the similarity between the filter at
# each time point and the data vector (sensors) at that time point.
#
# .. topic:: Examples
#
#     * :ref:`ex-ems-filtering`
#
# Patterns vs. filters
# ^^^^^^^^^^^^^^^^^^^^
#
# When interpreting the components of the CSP (or spatial filters in general),
# it is often more intuitive to think about how :math:`x(t)` is composed of
# the different CSP components :math:`x_{CSP}(t)`. In other words, we can
# rewrite Equation :eq:`csp` as follows:
#
# .. math::       x(t) = (W^{-1})^{T}x_{CSP}(t)
#    :label: patterns
#
# The columns of the matrix :math:`(W^{-1})^T` are called spatial patterns.
# This is also called the mixing matrix. The example :ref:`ex-linear-patterns`
# discusses the difference between patterns and filters.
#
# These can be plotted with:

# Fit CSP on full data and plot
csp.fit(X, y)
csp.plot_patterns(epochs.info)
csp.plot_filters(epochs.info, scalings=1e-9)

# %%
# Decoding over time
# ==================
#
# This strategy consists in fitting a multivariate predictive model on each
# time instant and evaluating its performance at the same instant on new
# epochs. The :class:`mne.decoding.SlidingEstimator` will take as input a
# pair of features :math:`X` and targets :math:`y`, where :math:`X` has
# more than 2 dimensions. For decoding over time the data :math:`X`
# is the epochs data of shape n_epochs × n_channels × n_times. As the
# last dimension of :math:`X` is the time, an estimator will be fit
# on every time instant.
#
# This approach is analogous to SlidingEstimator-based approaches in fMRI,
# where here we are interested in when one can discriminate experimental
# conditions and therefore figure out when the effect of interest happens.
#
# When working with linear models as estimators, this approach boils
# down to estimating a discriminative spatial filter for each time instant.
#
# Temporal decoding
# ^^^^^^^^^^^^^^^^^
#
# We'll use a Logistic Regression for a binary classification as machine
# learning model.

# We will train the classifier on all left visual vs auditory trials on MEG

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear')
)

time_decod = SlidingEstimator(
    clf, n_jobs=None, scoring='roc_auc', verbose=True)
# here we use cv=3 just for speed
scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')

# %%
# You can retrieve the spatial filters and spatial patterns if you explicitly
# use a LinearModel
clf = make_pipeline(
    StandardScaler(),
    LinearModel(LogisticRegression(solver='liblinear'))
)
time_decod = SlidingEstimator(
    clf, n_jobs=None, scoring='roc_auc', verbose=True)
time_decod.fit(X, y)

coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
evoked_time_gen.plot_joint(times=np.arange(0., .500, .100), title='patterns',
                           **joint_kwargs)

# %%
# Temporal generalization
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Temporal generalization is an extension of the decoding over time approach.
# It consists in evaluating whether the model estimated at a particular
# time instant accurately predicts any other time instant. It is analogous to
# transferring a trained model to a distinct learning problem, where the
# problems correspond to decoding the patterns of brain activity recorded at
# distinct time instants.
#
# The object to for Temporal generalization is
# :class:`mne.decoding.GeneralizingEstimator`. It expects as input :math:`X`
# and :math:`y` (similarly to :class:`~mne.decoding.SlidingEstimator`) but
# generates predictions from each model for all time instants. The class
# :class:`~mne.decoding.GeneralizingEstimator` is generic and will treat the
# last dimension as the one to be used for generalization testing. For
# convenience, here, we refer to it as different tasks. If :math:`X`
# corresponds to epochs data then the last dimension is time.
#
# This runs the analysis used in :footcite:`KingEtAl2014` and further detailed
# in :footcite:`KingDehaene2014`:

# define the Temporal generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=None, scoring='roc_auc',
                                 verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')

# %%
# Plot the full (generalization) matrix:

fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal generalization')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('AUC')

# %%
# Projecting sensor-space patterns to source space
# ================================================
# If you use a linear classifier (or regressor) for your data, you can also
# project these to source space. For example, using our ``evoked_time_gen``
# from before:

cov = mne.compute_covariance(epochs, tmax=0.)
del epochs
fwd = mne.read_forward_solution(
    meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif')
inv = mne.minimum_norm.make_inverse_operator(
    evoked_time_gen.info, fwd, cov, loose=0.)
stc = mne.minimum_norm.apply_inverse(evoked_time_gen, inv, 1. / 9., 'dSPM')
del fwd, inv

# %%
# And this can be visualized using :meth:`stc.plot <mne.SourceEstimate.plot>`:
brain = stc.plot(hemi='split', views=('lat', 'med'), initial_time=0.1,
                 subjects_dir=subjects_dir)

# %%
# Source-space decoding
# =====================
#
# Source space decoding is also possible, but because the number of features
# can be much larger than in the sensor space, univariate feature selection
# using ANOVA f-test (or some other metric) can be done to reduce the feature
# dimension. Interpreting decoding results might be easier in source space as
# compared to sensor space.
#
# .. topic:: Examples
#
#     * :ref:`ex-dec-st-source`
#
# Exercise
# ========
#
#  - Explore other datasets from MNE (e.g. Face dataset from SPM to predict
#    Face vs. Scrambled)
#
# References
# ==========
# .. footbibliography::
