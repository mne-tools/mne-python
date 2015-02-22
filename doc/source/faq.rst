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

Tha manual suggests regularizing the noise covariance matrix (see
:ref:`CBBHEGAB`). In mne-Python, this is done using :func:`mne.cov.regularize`,
as in::

    >>> import numpy as np
    >>> import mne
    >>> epochs = mne.read_epochs(epochs_path)
    >>> cov = mne.compute_covariance(ep, tmax=0.)
    >>> cov = mne.cov.regularize(cov, ep.info, mag=0.05, grad=0.05, eeg=0.1)

The noise covariance matrix can be evaluated by how well it whitens the data.
Under the assumption that the baseline does not contain a systematic signal,
the whitened baseline signal should be follow a multivariate Gaussian
distribution, i.e., whitened baseline signals should be between -1.96 and
1.96::

    >>> evoked = epochs.average()
    >>> whitened_evoked = mne.whiten_evoked(evoked, cov, range(len(cov.ch_names)))
    >>> whitened_evoked.plot()

Based on the same reasoning, the expected value for the global field power (GFP)
is 1 (calculation of the GFP should take into account the true degrees of
freedom, e.g. ``ddof=3`` with 2 active SSP vectors)::

    >>> gfp = whitened_evoked.data.std(0, ddof=1)
    >>> baseline = whitened_evoked.times < 0
    >>> print gfp[baseline].mean()  # should be close to 1
    1.0201640312647

A plot of the global field power of the whitenend evoked over time can also be
illustrative::

>>> import matplotlib.pyplot as plt
>>> plt.plot(evoked.times, gfp)
>>> plt.show()

These methods for evaluation can be used to choose optimal regularization
parameters (parameters can be chosen separately for each sensor type using the
``mag``, ``grad`` and ``eeg`` parameters in :func:`mne.cov.regularize`).
