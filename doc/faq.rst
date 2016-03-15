.. include:: links.inc

.. _faq:

==========================
Frequently Asked Questions
==========================

.. contents:: Contents
   :local:


General MNE-Python issues
=========================

Help! I can't get Python and MNE-Python working!
------------------------------------------------

Check out our section on how to get Anaconda up and running over at the
:ref:`getting started page <install_interpreter>`.

I'm not sure how to do *X* analysis step with my *Y* data...
------------------------------------------------------------

Knowing "the right thing" to do with EEG and MEG data is challenging.
We use the `MNE mailing list`_ to discuss
how to deal with different bits of data. It's worth searching the archives
to see if there have been relevant discussions before.

I think I found a bug, what do I do?
------------------------------------

Please report any problems you find while using MNE-Python
`issue tracker <https://github.com/mne-tools/mne-python/issues/>`_.
Try :ref:`using the latest master version <installing_master>` to
see if the problem persists before reporting the bug, as it may have
been fixed since the latest release.

It is helpful to include system information with bug reports, so it can be
useful to include the output of the :func:`mne.sys_info` command when
reporting a bug, which should look something like this::

    >>> import mne
    >>> mne.sys_info()  # doctest:+SKIP
    Platform:      Linux-4.2.0-27-generic-x86_64-with-debian-jessie-sid
    Python:        2.7.11 |Continuum Analytics, Inc.| (default, Dec  6 2015, 18:08:32)  [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
    Executable:    /home/larsoner/miniconda/bin/python

    mne:           0.12.dev0
    numpy:         1.10.2 {lapack=mkl_lapack95_lp64, blas=mkl_intel_lp64}
    scipy:         0.16.1
    matplotlib:    1.5.1

    sklearn:       Not found
    nibabel:       Not found
    nitime:        Not found
    mayavi:        Not found
    nose:          1.3.7
    pandas:        Not found
    pycuda:        Not found
    skcuda:        Not found


Inverse Solution
================

How should I regularize the covariance matrix?
----------------------------------------------

The estimated covariance can be numerically
unstable and tends to induce correlations between estimated source amplitudes
and the number of samples available. The MNE manual therefore suggests to regularize the noise covariance matrix (see
:ref:`CBBHEGAB`), especially if only few samples are available. Unfortunately
it is not easy to tell the effective number of samples, hence, to chose the appropriate regularization.
In MNE-Python, regularization is done using advanced regularization methods
described in [1]_. For this the 'auto' option can be used. With this
option cross-validation will be used to learn the optimal regularization::

    >>> import mne
    >>> epochs = mne.read_epochs(epochs_path) # doctest: +SKIP
    >>> cov = mne.compute_covariance(epochs, tmax=0., method='auto') # doctest: +SKIP

This procedure evaluates the noise covariance quantitatively by how well it whitens the data using the
negative log-likelihood of unseen data. The final result can also be visually inspected.
Under the assumption that the baseline does not contain a systematic signal
(time-locked to the event of interest), the whitened baseline signal should be
follow a multivariate Gaussian distribution, i.e.,
whitened baseline signals should be between -1.96 and 1.96 at a given time sample.
Based on the same reasoning, the expected value for the global field power (GFP)
is 1 (calculation of the GFP should take into account the true degrees of
freedom, e.g. ``ddof=3`` with 2 active SSP vectors)::

    >>> evoked = epochs.average() # doctest: +SKIP
    >>> evoked.plot_white(cov) # doctest: +SKIP

This plot displays both, the whitened evoked signals for each channels and
the whitened GFP. The numbers in the GFP panel represent the estimated rank of
the data, which amounts to the effective degrees of freedom by which the
squared sum across sensors is divided when computing the whitened GFP.
The whitened GFP also helps detecting spurious late evoked components which
can be the consequence of over- or under-regularization.

Note that if data have been processed using signal space separation (SSS) [2]_,
gradiometers and magnetometers will be displayed jointly because both are
reconstructed from the same SSS basis vectors with the same numerical rank.
This also implies that both sensor types are not any longer linearly independent.

These methods for evaluation can be used to assess model violations. Additional
introductory materials can be found `here <https://speakerdeck.com/dengemann/eeg-sensor-covariance-using-cross-validation>`_.

For expert use cases or debugging the alternative estimators can also be compared::

    >>> covs = mne.compute_covariance(epochs, tmax=0., method='auto', return_estimators=True) # doctest: +SKIP
    >>> evoked = epochs.average() # doctest: +SKIP
    >>> evoked.plot_white(covs) # doctest: +SKIP

This will plot the whitened evoked for the optimal estimator and display the GFPs
for all estimators as separate lines in the related panel.

References
----------

.. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
    covariance estimation and spatial whitening of MEG and EEG signals,
    vol. 108, 328-342, NeuroImage.

.. [2] Taulu, S., Simola, J., Kajola, M., 2005. Applications of the signal space
    separation method. IEEE Trans. Signal Proc. 53, 3359–3372.
