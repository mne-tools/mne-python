.. include:: ../git_links.inc

.. contents:: Contents
   :local:
   :depth: 3

.. _decoding:

Decoding in MNE
###############

For maximal compatibility with the Scikit-learn package, we follow the same API. Each estimator implements a ``fit``, a ``transform`` and a ``fit_transform`` method. In some cases, they also implement an ``inverse_transform`` method. For more details, visit the Scikit-learn page.

For ease of comprehension, we will denote instantiations of the class using the same name as the class but in small caps instead of camel cases.

Basic Estimators
================

Scaler
^^^^^^
The :class:`mne.decoding.Scaler` will standardize the data based on channel scales. In the simplest modes ``scalings=None`` or ``scalings=dict(...)``, each data channel type (e.g., mag, grad, eeg) is treated separately and scaled by a constant. This is the approach used by e.g., :func:`mne.compute_covariance` to standardize channel scales.

If ``scalings='mean'`` or ``scalings='median'``, each channel is scaled using empirical measures. Each channel is scaled independently by the mean and standand deviation, or median and interquartile range, respectively, across all epochs and time points during :class:`mne.decoding.Scaler.fit` (during training). The :meth:`mne.decoding.Scaler.transform` method is called to transform data (training or test set) by scaling all time points and epochs on a channel-by-channel basis. To perform both the ``fit`` and ``transform`` operations in a single call, the :meth:`mne.decoding.Scaler.fit_transform` method may be used. To invert the transform, :meth:`mne.decoding.Scaler.inverse_transform` can be used. For ``scalings='median'``, scikit-learn_ version 0.17+ is required.

.. note:: This is different from directly applying :class:`sklearn.preprocessing.StandardScaler` or :class:`sklearn.preprocessing.RobustScaler` offered by scikit-learn_. The ``StandardScaler`` and ``RobustScaler`` scale each *classification feature*, e.g. each time point for each channel, with mean and standard deviation computed across epochs, whereas ``Scaler`` scales each *channel* using mean and standard deviation computed across all of its time points and epochs.

Vectorizer
^^^^^^^^^^
Scikit-learn API provides functionality to chain transformers and estimators by using :class:`sklearn.pipeline.Pipeline`. We can construct decoding pipelines and perform cross-validation and grid-search. However scikit-learn transformers and estimators generally expect 2D data (n_samples * n_features), whereas MNE transformers typically output data with a higher dimensionality (e.g. n_samples * n_channels * n_frequencies * n_times). A Vectorizer therefore needs to be applied between the MNE and the scikit-learn steps: e.g: make_pipeline(Xdawn(), Vectorizer(), LogisticRegression())

PSDEstimator
^^^^^^^^^^^^
This estimator computes the power spectral density (PSD) using the multitaper method. It takes a 3D array as input, it into 2D and computes the PSD.

FilterEstimator
^^^^^^^^^^^^^^^
This estimator filters the 3D epochs data.

.. warning:: This is meant for use in conjunction with ``RtEpochs``. It is not recommended in a normal processing pipeline as it may result in edge artifacts.

Spatial filters
===============

Just like temporal filters, spatial filters provide weights to modify the data along the sensor dimension. They are popular in the BCI community because of their simplicity and ability to distinguish spatially-separated neural activity.

Common Spatial Pattern
^^^^^^^^^^^^^^^^^^^^^^

This is a technique to analyze multichannel data based on recordings from two classes. Let :math:`X \in R^{C\times T}` be a segment of data with :math:`C` channels and :math:`T` time points. The data at a single time point is denoted by :math:`x(t)` such that :math:`X=[x(t), x(t+1), ..., x(t+T-1)]`. Common Spatial Pattern (CSP) finds a decomposition that projects the signal in the original sensor space to CSP space using the following transformation:

.. math::       x_{CSP}(t) = W^{T}x(t)
   :label: csp

where each column of :math:`W \in R^{C\times C}` is a spatial filter and each row of :math:`x_{CSP}` is a CSP component. The matrix :math:`W` is also called the de-mixing matrix in other contexts. Let :math:`\Sigma^{+} \in R^{C\times C}` and :math:`\Sigma^{-} \in R^{C\times C}` be the estimates of the covariance matrices of the two conditions.
CSP analysis is given by the simultaneous diagonalization of the two covariance matrices

.. math::       W^{T}\Sigma^{+}W = \lambda^{+}
   :label: diagonalize_p
.. math::       W^{T}\Sigma^{-}W = \lambda^{-}
   :label: diagonalize_n

where :math:`\lambda^{C}` is a diagonal matrix whose entries are the eigenvalues of the following generalized eigenvalue problem

.. math::      \Sigma^{+}w = \lambda \Sigma^{-}w
   :label: eigen_problem

Large entries in the diagonal matrix corresponds to a spatial filter which gives high variance in one class but low variance in the other. Thus, the filter facilitates discrimination between the two classes.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_csp_eeg.py`
    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_csp_space.py`

.. topic:: Spotlight:

    The winning entry of the Grasp-and-lift EEG competition in Kaggle uses the CSP implementation in MNE. It was featured as a `script of the week`_.


Source Power Comodulation (SPoC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Source Power Comodulation (SPoC) [1]_ allows to identify the composition of orthogonal spatial filters that maximally correlate with a continuous target.

SPoC can be seen as an extension of the CSP where the target is driven by a continuous variable rather than a discrete variable. Typical applications include extraction of motor patterns using EMG power or audio patterns using sound envelope.

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_spoc_CMC.py`

xDAWN
^^^^^
Xdawn is a spatial filtering method designed to improve the signal to signal + noise ratio (SSNR) of the ERP responses. Xdawn was originally  designed for P300 evoked potential by enhancing the target response with respect to the non-target response. The implementation in MNE-Python is a generalization to any type of ERP.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_preprocessing_plot_xdawn_denoising.py`
    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_xdawn_eeg.py`

Effect-matched spatial filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The result is a spatial filter at each time point and a corresponding time course. Intuitively, the result gives the similarity between the filter at each time point and the data vector (sensors) at that time point.

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_decoding_plot_ems_filtering.py`

Patterns vs. filters
^^^^^^^^^^^^^^^^^^^^

When interpreting the components of the CSP, it is often more intuitive to think about how :math:`x(t)` is composed of the different CSP components :math:`x_{CSP}(t)`. In other words, we can rewrite Equation :eq:`csp` as follows:

.. math::       x(t) = (W^{-1})^{T}x_{CSP}(t)
   :label: patterns

The columns of the matrix :math:`(W^{-1})^T` are called spatial patterns. This is also called the mixing matrix. The example :ref:`sphx_glr_auto_examples_decoding_plot_linear_model_patterns.py` demonstrates the difference between patterns and filters.

Plotting a pattern is as simple as doing::

    >>> info = epochs.info
    >>> model.plot_patterns(info)  # model is an instantiation of an estimator described in this section

.. image:: ../../_images/sphx_glr_plot_linear_model_patterns_001.png
   :align: center
   :height: 100 px

To plot the corresponding filter, you can do::

    >>> model.plot_filters(info)

.. image:: ../../_images/sphx_glr_plot_linear_model_patterns_002.png
   :align: center
   :height: 100 px

Sensor-space decoding
=====================

Decoding over time
^^^^^^^^^^^^^^^^^^

This strategy consists in fitting a multivariate predictive model on each
time instant and evaluating its performance at the same instant on new
epochs. The :class:`mne.decoding.SlidingEstimator` will take as input a
pair of features :math:`X` and targets :math:`y`, where :math:`X` has
more than 2 dimensions. For decoding over time the data :math:`X`
is the epochs data of shape n_epochs x n_channels x n_times. As the
last dimension of :math:`X` is the time an estimator will be fit
on every time instant.

This approach is analogous to SlidingEstimator-based approaches in fMRI,
where here we are interested in when one can discriminate experimental
conditions and therefore figure out when the effect of interest happens.

When working with linear models as estimators, this approach boils
down to estimating a discriminative spatial filter for each time instant.

.. image:: ../../_images/sphx_glr_plot_decoding_sensors_001.png
   :align: center
   :width: 400px

To generate this plot see our tutorial :ref:`sphx_glr_auto_tutorials_plot_sensors_decoding.py`.

Temporal Generalization
^^^^^^^^^^^^^^^^^^^^^^^

Temporal Generalization is an extension of the decoding over time approach.
It consists in evaluating whether the model estimated at a particular
time instant accurately predicts any other time instant. It is analogous to
transferring a trained model to a distinct learning problem, where the problems
correspond to decoding the patterns of brain activity recorded at distinct time
instants.

The object to for Temporal Generalization is
:class:`mne.decoding.GeneralizingEstimator`. It expects as input :math:`X` and
:math:`y` (similarly to :class:`mne.decoding.SlidingEstimator`) but, when generate
predictions from each model for all time instants. The class
:class:`mne.decoding.GeneralizingEstimator` is generic and will treat the last
dimension as the one to be used for generalization testing. For convenience,
here, we refer to it different tasks. If :math:`X` corresponds to epochs data
then the last dimension is time.

.. image:: ../../_images/sphx_glr_plot_decoding_time_generalization_001.png
   :align: center
   :width: 400px

To generate this plot see our tutorial :ref:`sphx_glr_auto_tutorials_plot_sensors_decoding.py`.

Source-space decoding
=====================

Source space decoding is also possible, but because the number of features can be much larger than in the sensor space, univariate feature selection using ANOVA f-test (or some other metric) can be done to reduce the feature dimension. Interpreting decoding results might be easier in source space as compared to sensor space.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_decoding_plot_decoding_spatio_temporal_source.py`

.. _script of the week: http://blog.kaggle.com/2015/08/12/july-2015-scripts-of-the-week/

References
==========

.. [1] Dahne, S., Meinecke, F. C., Haufe, S., Hohne, J., Tangermann, M., Muller, K. R., & Nikulin, V. V. (2014). SPoC: a novel framework for relating the amplitude of neuronal oscillations to behaviorally relevant parameters. NeuroImage, 86, 111-122.
