.. _faq:

==========================
Frequently Asked Questions
==========================

.. contents:: Contents
   :local:


Inverse Solution
================

How should I regularize the covariance matrix
---------------------------------------------

The MNE manual suggests regularizing the noise covariance matrix (see
:ref:`CBBHEGAB`). In MNE-Python, this is done using the regularization methods
described in [1, 2, 3, 4]. For this the 'auto' option can be used. With this
option cross-validation will be used to learn the optimal regularization.

    >>> import numpy as np
    >>> import mne
    >>> epochs = mne.read_epochs(epochs_path)
    >>> cov = mne.compute_covariance(epochs, tmax=0., method='auto')

The noise covariance matrix can then be evaluated by how well it whitens the data.
Under the assumption that the baseline does not contain a systematic signal
(time-locked to the event of interest), the whitened baseline signal should be
follow a multivariate Gaussian distribution, i.e.,
whitened baseline signals should be between -1.96 and 1.96.
Based on the same reasoning, the expected value for the global field power (GFP)
is 1 (calculation of the GFP should take into account the true degrees of
freedom, e.g. ``ddof=3`` with 2 active SSP vectors)::

    >>> evoked = epochs.average()
    >>> evoked.plot_white(cov)

This plot displays both, the whitened evoked signals for each channels and
the whitened GFP. The numbers in the GFP panel represent the estimated rank of
the data, which amounts to the effective degrees of freedom by which the
sqaured sum across sensors is devided when computing the whitened GFP [1].
The whitened GFP also helps detecting spurious late components which are the
consequence of over- or under-regularization.

Note that if data have been processed using signal space separation (SSS) [5],
gradiometers and magnetometers will be displayed jointly because both are
reconstructed from the same SSS basis vectors with the same numerical rank.
This also implies that both sensor types are not any longer linearly independent.

These methods for evaluation can be used to assess model violations. Additional
introductory materials cen be found [here](https://speakerdeck.com/dengemann/eeg-sensor-covariance-using-cross-validation).

For expert use cases or debugging the alternative estimators can also be compareded::

    >>> covs = mne.compute_covariance(epochs, tmax=0., method='auto', return_estimators=True)
    >>> evoked = epochs.average()
    >>> evoked.plot_white(covs)

This will plot the whitened evoked for the optimal estimator and display the GFPs
for all estimators as seprate lines in the related panel.

References
----------
[1] Engemann D. and Gramfort A. (2015) Automated model selection in
    covariance estimation and spatial whitening of MEG and EEG signals,
    vol. 108, 328-342, NeuroImage.
[2] Ledoit, O., Wolf, M., (2004). A well-conditioned estimator for
    large-dimensional covariance matrices. Journal of Multivariate
    Analysis 88 (2), 365 - 411.
[3] Tipping, M. E., Bishop, C. M., (1999). Probabilistic principal
    component analysis. Journal of the Royal Statistical Society: Series
    B (Statistical Methodology) 61 (3), 611 - 622.
[4] Barber, D., (2012). Bayesian reasoning and machine learning.
    Cambridge University Press., Algorithm 21.1
[5] Taulu, S., Simola, J., Kajola, M., 2005. Applications of the signal space
    separation method. IEEE Trans. Signal Proc. 53, 3359–3372.
