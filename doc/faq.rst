:orphan:

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

I still can't get it to work!
-----------------------------
For analysis talk, join the `MNE mailing list`_. File specific feature
requests or bug reports `on GitHub <https://github.com/mne-tools/mne-python/issues/>`_.
You can also chat with developers `on Gitter <https://gitter.im/mne-tools/mne-python>`_.

.. _cite:

How do I cite MNE?
------------------
If you use the implementations provided by the MNE software in your research,
you should cite:

    - A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, `MNE software for processing MEG and EEG data <http://www.ncbi.nlm.nih.gov/pubmed/24161808>`_, NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119, `[DOI] <http://dx.doi.org/10.1016/j.neuroimage.2013.10.027>`__

If you use the Python code you should cite as well:

    - A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, `MEG and EEG data analysis with MNE-Python <http://journal.frontiersin.org/article/10.3389/fnins.2013.00267/abstract>`_, Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X, `[DOI] <http://dx.doi.org/10.3389/fnins.2013.00267>`__

To cite specific versions of the software, you can use the DOIs provided by
`Zenodo <https://zenodo.org/search?ln=en&p=mne-python>`_.

You should as well cite the related method papers, some of which are listed in :ref:`ch_reading`.

I'm not sure how to do *X* analysis step with my *Y* data...
------------------------------------------------------------

Knowing "the right thing" to do with EEG and MEG data is challenging.
We use the `MNE mailing list`_ to discuss
how to deal with different bits of data. It's worth searching the archives
to see if there have been relevant discussions before.

I think I found a bug, what do I do?
------------------------------------

Please report any problems you find while using MNE-Python to the
`GitHub issues page`_.
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


Why is it dangerous to "pickle" my MNE-Python objects and data for later use?
-----------------------------------------------------------------------------
`Pickling <https://docs.python.org/3/library/pickle.html>`_ data and
MNE-Python objects for later use can be tempting due to its simplicity
and generality, but it is usually not the best option. Pickling is not
designed for stable persistence, and it is likely that you will not be
able to read your data in the not-too-distant future. For details, see:

- http://www.benfrederickson.com/dont-pickle-your-data/
- http://stackoverflow.com/questions/21752259/python-why-pickle

MNE-Python is designed to provide its own file saving formats
(often based on the FIF standard) for its objects usually via a
``save`` method or ``write_*`` method, e.g. :func:`mne.io.Raw.save`,
:func:`mne.Epochs.save`, :func:`mne.write_evokeds`,
:func:`mne.SourceEstimate.save`. If you have some data that you
want to save but can't figure out how, shoot an email to the
`MNE mailing list`_ or post it to the `GitHub issues page`_.

If you want to write your own data to disk (e.g., subject behavioral
scores), we strongly recommend using `h5io <https://github.com/h5io/h5io>`_,
which is based on the
`HDF5 format <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ and
h5py_, to save data in a fast, future-compatible, standard format.


I downloaded a dataset once, but MNE-Python is asking to download it again. Why?
--------------------------------------------------------------------------------
The default location for the MNE-sample data is ``~/mne_data``.
If you downloaded data and an example asks you whether to download it again,
make sure the data reside in the examples directory and that you run the
script from its current directory:

.. code-block:: bash

  $ cd examples/preprocessing

Then in Python you can do::

  In [1]: %run plot_find_ecg_artifacts.py

See :ref:`datasets` for a list of all available datasets and some advanced
configuration options, e.g. to specify a custom location for storing the
datasets.

.. _faq_cpu:

A function uses multiple CPU cores even though I didn't tell it to. Why?
------------------------------------------------------------------------
Ordinarily in MNE-python the ``parallel`` module is used to deploy multiple
cores via the ``n_jobs`` variable. However, functions like
:func:`mne.preprocessing.maxwell_filter` that use :mod:`scipy.linalg` do not have an
``n_jobs`` flag but may still use multiple cores. This is because :mod:`scipy.linalg`
is built with linear algebra libraries that natively support multithreading:

* `OpenBLAS <http://www.openblas.net/>`_
* `Intel Math Kernel Library (MKL) <https://software.intel.com/en-us/intel-mkl>`_,
  which uses `OpenMP <http://www.openmp.org/>`_

To control how many cores are used for linear-algebra-heavy functions like
:func:`mne.preprocessing.maxwell_filter`, you can set the
``OMP_NUM_THREADS`` or ``OPENBLAS_NUM_THREADS`` environment variable to the
desired number of cores for MKL or OpenBLAS, respectively. This can be done
before running Python, or inside Python you can achieve the same effect by,
e.g.::

    >>> import os
    >>> num_cpu = '4' # Set as a string
    >>> os.environ['OMP_NUM_THREADS'] = num_cpu

This must be done *before* running linear algebra functions; subsequent
changes in the same Python session will have no effect.


Resampling and decimating data
==============================

What are all these options for resampling, decimating, and binning data?
------------------------------------------------------------------------

There are many functions in MNE-Python for changing the effective sampling
rate of data. We'll discuss some major ones here, with some of their
implications:

- :func:`mne.io.Raw.resample` is used to resample (typically downsample) raw
  data. Resampling is the two-step process of applying a low-pass FIR filter
  and subselecting samples from the data.

  Using this function to resample data before forming :class:`mne.Epochs`
  for final analysis is generally discouraged because doing so effectively
  loses precision of (and jitters) the event timings, see
  `this gist <https://gist.github.com/Eric89GXL/01642cb3789992fbca59>`_ as
  a demonstration. However, resampling raw data can be useful for
  (at least):

    - Computing projectors in low- or band-passed data
    - Exploring data

- :func:`mne.preprocessing.ICA.fit` decimates data without low-passing,
  but is only used for fitting a statistical model to the data.

- :func:`mne.Epochs.decimate`, which does the same thing as the
  ``decim`` parameter in the :class:`mne.Epochs` constructor, sub-selects every
  :math:`N^{th}` sample before and after each event. This should only be
  used when the raw data have been sufficiently low-passed e.g. by
  :func:`mne.io.Raw.filter` to avoid aliasing artifacts.

- :func:`mne.Epochs.resample`, :func:`mne.Evoked.resample`, and
  :func:`mne.SourceEstimate.resample` all resample data.
  This process avoids potential aliasing artifacts because the
  resampling process applies a low-pass filter. However, this filtering
  introduces edge artifacts. Edge artifacts also exist when using
  :func:`mne.io.Raw.resample`, but there the edge artifacts are constrained
  to two times: the start and end of the recording. With these three methods,
  edge artifacts are introduced to the start and end of every epoch
  of data (or the start and end of the :class:`mne.Evoked` or
  :class:`mne.SourceEstimate` data), which often has a more pronounced
  effect on the data.

- :func:`mne.SourceEstimate.bin` can be used to decimate, with or without
  "binning" (averaging across data points). This is equivalent to applying
  a moving-average (boxcar) filter to the data and decimating. A boxcar in
  time is a `sinc <https://en.wikipedia.org/wiki/Sinc_function>`_ in
  frequency, so this acts as a simplistic, non-ideal low-pass filter;
  this will reduce but not eliminate aliasing if data were not sufficiently
  low-passed. In the case where the "filter" or bin-width is a single sample
  (i.e., an impulse) this operation simplifies to decimation without filtering.

Resampling raw data is taking forever! What do I do?
----------------------------------------------------

:func:`mne.io.Raw.resample` was significantly sped up for version 0.12 by
using the parameter ``npad=='auto'``. Try it, it might help!

If you have an NVIDIA GPU you could also try using :ref:`CUDA`, which can
sometimes speed up filtering and resampling operations by an order of
magnitude.


Inverse Solution
================

How should I regularize the covariance matrix?
----------------------------------------------

The estimated covariance can be numerically
unstable and tends to induce correlations between estimated source amplitudes
and the number of samples available. The MNE manual therefore suggests to regularize the noise covariance matrix (see
:ref:`cov_regularization`), especially if only few samples are available. Unfortunately
it is not easy to tell the effective number of samples, hence, to choose the appropriate regularization.
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

Morphing data
=============

Should I morph my source estimates using ``morph`` or ``morph_precomputed``?
----------------------------------------------------------------------------
The two functions :func:`mne.SourceEstimate.morph` and
:func:`mne.SourceEstimate.morph_precomputed` perform the same operation:
taking surface-based source space data from one subject and
morphing it to another using a smoothing procedure. However, they can
take different amounts of time to perform the computation.

To use :func:`mne.SourceEstimate.morph_precomputed`, you must first
precompute a morphing matrix with :func:`mne.compute_morph_matrix` which
can take some time, but then the actual morphing operation carried out by
:func:`mne.SourceEstimate.morph_precomputed` is very fast, even for
:class:`mne.SourceEstimate` objects with many time points. The method
:func:`mne.SourceEstimate.morph`, by contrast, smooths the data by operating
directly on the data, which can be **very slow** with many time points.
If there are thousands of time points, then
:func:`mne.SourceEstimate.morph_precomputed` will be much faster; if there
are a few time points, then :func:`mne.SourceEstimate.morph` will be faster.
For data sizes in between, we advise testing to determine which is best,
although some developers choose to always use
:func:`mne.SourceEstimate.morph_precomputed` since it will rarely take
a long time.

References
----------

.. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
       covariance estimation and spatial whitening of MEG and EEG signals,
       vol. 108, 328-342, NeuroImage.

.. [2] Taulu, S., Simola, J., Kajola, M., 2005. Applications of the signal
       space separation method. IEEE Trans. Signal Proc. 53, 3359–3372.
