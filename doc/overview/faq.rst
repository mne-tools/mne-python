.. include:: ../links.inc

.. _faq:

================================
Frequently Asked Questions (FAQ)
================================

.. contents:: Page contents
   :local:

.. highlight:: python

General MNE-Python issues
=========================


Help! I can't get Python and MNE-Python working!
------------------------------------------------

Check out our section on how to get Anaconda up and running over at the
:ref:`getting started page <install_python_and_mne_python>`.


I still can't get it to work!
-----------------------------

See :ref:`help`.


I can't get Mayavi/3D plotting to work under Windows
----------------------------------------------------

If Mayavi plotting in Jupyter Notebooks doesn't work well, using the IPython
magic ``%gui qt`` after importing MNE/Mayavi/PySurfer should `help
<https://github.com/ipython/ipython/issues/10384>`_.

.. code-block:: ipython

   from mayavi import mlab
   %gui qt

Python runs on macOS extremely slow even on simple commands!
------------------------------------------------------------

Python uses some backends that interfere with the macOS energy saver when
using an IDE such as Spyder or PyCharm. To test it, import ``time`` and run::

    start = time.time(); time.sleep(0.0005); print(time.time() - start)

If it takes several seconds you can either:

- Install the module ``appnope`` and run in your script::

      import appnope
      appnope.nope()

- Change the configuration defaults by running in your terminal:

  .. code-block:: console

      $ defaults write org.python.python NSAppSleepDisabled -bool YES


How do I cite MNE?
------------------

See :ref:`cite`.


I'm not sure how to do *X* analysis step with my *Y* data...
------------------------------------------------------------

Knowing "the right thing" to do with EEG and MEG data is challenging. We use
the `MNE mailing list`_ to discuss analysis strategies for different kinds of
data. It's worth searching the archives to see if there have been relevant
discussions in the past, but don't hesitate to ask a new question if the answer
isn't out there already.


I think I found a bug, what do I do?
------------------------------------

When you encounter an error message or unexpected results, it can be hard to
tell whether it happened because of a bug in MNE-Python, a mistake in user
code, a corrupted data file, or irregularities in the data itself. Your first
step when asking for help should be the `MNE mailing list`_ or the
`MNE Gitter channel`_, not GitHub. This bears repeating: *the GitHub issue
tracker is not for usage help* — it is for software bugs, feature requests, and
improvements to documentation. If you open an issue that contains only a usage
question, we will close the issue and direct you to the mailing list or Gitter
channel. If you're pretty sure the problem you've encountered is a software bug
(not bad data or user error):

- Make sure you're using `the most current version`_. You can check it locally
  at a shell prompt with:

  .. code-block:: console

      $ mne sys_info

  which will also give you version info about important MNE-Python
  dependencies.

- If you're already on the most current version, if possible try using
  :ref:`the latest development version <installing_master>`, as the bug may
  have been fixed already since the latest release. If you can't try the latest
  development version, search the GitHub issues page to see if the problem has
  already been reported and/or fixed.

- Try to replicate the problem with one of the :ref:`MNE sample datasets
  <datasets>`. If you can't replicate it with a built-in dataset, provide a
  link to a small, anonymized portion of your data that does yield the error.

If the problem persists, `open a new issue
<https://github.com/mne-tools/mne-python/issues/new?template=bug_report.md>`__
and include the *smallest possible* code sample that replicates the error
you're seeing. Paste the code sample into the issue, with a line containing
three backticks (\`\`\`) above and below the lines of code. This
`minimal working example`_ should be self-contained, which means that
MNE-Python contributors should be able to copy and paste the provided snippet
and replicate the bug on their own computers.

If you post to the `mailing list
<https://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis>__
instead, a `GitHub Public Gist <https://gist.github.com>`_ for the code sample
is recommended; if you use the
[Gitter channel](https://gitter.im/mne-tools/mne-python) the three backticks
(\`\`\`) trick works there too.


Why is it dangerous to "pickle" my MNE-Python objects and data for later use?
-----------------------------------------------------------------------------

`Pickling <https://docs.python.org/3/library/pickle.html>`_ data and MNE-Python
objects for later use can be tempting due to its simplicity and generality, but
it is usually not the best option. Pickling is not designed for stable
persistence, and it is likely that you will not be able to read your data in
the not-too-distant future. For details, see:

- http://www.benfrederickson.com/dont-pickle-your-data/
- https://stackoverflow.com/questions/21752259/python-why-pickle

MNE-Python is designed to provide its own file saving formats (often based on
the FIF standard) for its objects usually via a ``save`` method or ``write_*``
method, e.g. :func:`mne.io.Raw.save`, :func:`mne.Epochs.save`,
:func:`mne.write_evokeds`, :func:`mne.SourceEstimate.save`. If you have some
data that you want to save but can't figure out how, shoot an email to the `MNE
mailing list`_ or post it to the `GitHub issues page`_.

If you want to write your own data to disk (e.g., subject behavioral scores),
we strongly recommend using `h5io <https://github.com/h5io/h5io>`_, which is
based on the `HDF5 format
<https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ and h5py_, to save
data in a fast, future-compatible, standard format.


I downloaded a dataset once, but MNE-Python is asking to download it again. Why?
--------------------------------------------------------------------------------

The default location for the MNE-sample data is ``~/mne_data``. If you
downloaded data and an example asks you whether to download it again, make sure
the data reside in the examples directory and that you run the script from its
current directory:

.. code-block:: console

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
:func:`mne.preprocessing.maxwell_filter` that use :mod:`scipy.linalg` do not
have an ``n_jobs`` flag but may still use multiple cores. This is because
:mod:`scipy.linalg` is built with linear algebra libraries that natively
support multithreading:

- `OpenBLAS <http://www.openblas.net/>`_
- `Intel Math Kernel Library (MKL) <https://software.intel.com/en-us/mkl>`_,
  which uses `OpenMP <https://www.openmp.org/>`_

To control how many cores are used for linear-algebra-heavy functions like
:func:`mne.preprocessing.maxwell_filter`, you can set the ``OMP_NUM_THREADS``
or ``OPENBLAS_NUM_THREADS`` environment variable to the desired number of cores
for MKL or OpenBLAS, respectively. This can be done before running Python, or
inside Python you can achieve the same effect by, e.g.::

    >>> import os
    >>> num_cpu = '4' # Set as a string
    >>> os.environ['OMP_NUM_THREADS'] = num_cpu

This must be done *before* running linear algebra functions; subsequent
changes in the same Python session will have no effect.


I have a mystery FIF file, how do I read it?
--------------------------------------------

The :func:`mne.what` function can be called on any :file:`.fif` file to
identify the kind of data contained in the file. This will help you determine
whether to use :func:`mne.read_cov`, :func:`mne.read_epochs`,
:func:`mne.read_evokeds`, etc. There is also a corresponding command line tool
:ref:`mne what`:

.. code-block:: console

    $ mne what sample_audvis_eog-eve.fif
    events


Resampling and decimating data
==============================

What are all these options for resampling, decimating, and binning data?
------------------------------------------------------------------------

There are many functions in MNE-Python for changing the effective sampling rate
of data. We'll discuss some major ones here, with some of their implications:

- :func:`mne.io.Raw.resample` is used to resample (typically downsample) raw
  data. Resampling is the two-step process of applying a low-pass FIR filter
  and subselecting samples from the data.

  Using this function to resample data before forming :class:`mne.Epochs`
  for final analysis is generally discouraged because doing so effectively
  loses precision of (and jitters) the event timings, see
  `this gist <https://gist.github.com/larsoner/01642cb3789992fbca59>`_ as
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

:func:`mne.io.Raw.resample` has a parameter ``npad=='auto'``. This is the
default, but if you've changed it you could try changing it back to ``'auto'``,
it might help.

If you have an NVIDIA GPU you could also try using :ref:`CUDA`, which can
sometimes speed up filtering and resampling operations by an order of
magnitude.


Forward and Inverse Solution
============================


How should I regularize the covariance matrix?
----------------------------------------------

The estimated covariance can be numerically unstable and tends to induce
correlations between estimated source amplitudes and the number of samples
available. It is thus suggested to regularize the noise covariance
matrix (see :ref:`cov_regularization_math`), especially if only few samples
are available. Unfortunately it is not easy to tell the effective number of
samples, hence, to choose the appropriate regularization. In MNE-Python,
regularization is done using advanced regularization methods described in
:footcite:`EngemannGramfort2015`. For this the 'auto' option can be used. With
this option cross-validation will be used to learn the optimal regularization::

    >>> import mne
    >>> epochs = mne.read_epochs(epochs_path) # doctest: +SKIP
    >>> cov = mne.compute_covariance(epochs, tmax=0., method='auto') # doctest: +SKIP

This procedure evaluates the noise covariance quantitatively by how well it
whitens the data using the negative log-likelihood of unseen data. The final
result can also be visually inspected. Under the assumption that the baseline
does not contain a systematic signal (time-locked to the event of interest),
the whitened baseline signal should be follow a multivariate Gaussian
distribution, i.e., whitened baseline signals should be between -1.96 and 1.96
at a given time sample. Based on the same reasoning, the expected value for the
:term:`Global Field Power (GFP) <GFP>` is 1 (calculation of the :term:`GFP`
should take into account the true degrees of freedom, e.g. ``ddof=3`` with 2
active SSP vectors)::

    >>> evoked = epochs.average() # doctest: +SKIP
    >>> evoked.plot_white(cov) # doctest: +SKIP

This plot displays both, the whitened evoked signals for each channels and the
whitened :term:`GFP`. The numbers in the :term:`GFP` panel represent the
estimated rank of the data, which amounts to the effective degrees of freedom
by which the squared sum across sensors is divided when computing the whitened
:term:`GFP`. The whitened :term:`GFP` also helps detecting spurious late evoked
components which can be the consequence of over- or under-regularization.

Note that if data have been processed using signal space separation (SSS)
:footcite:`TauluEtAl2005`, gradiometers and magnetometers will be displayed
jointly because both are reconstructed from the same SSS basis vectors with the
same numerical rank. This also implies that both sensor types are not any
longer linearly independent.

These methods for evaluation can be used to assess model violations. Additional
introductory materials can be found `here
<https://speakerdeck.com/dengemann/eeg-sensor-covariance-using-cross-validation>`_.

For expert use cases or debugging the alternative estimators can also be
compared::

    >>> covs = mne.compute_covariance(epochs, tmax=0., method='auto', return_estimators=True) # doctest: +SKIP
    >>> evoked = epochs.average() # doctest: +SKIP
    >>> evoked.plot_white(covs) # doctest: +SKIP

This will plot the whitened evoked for the optimal estimator and display the
:term:`GFPs <GFP>` for all estimators as separate lines in the related panel.


.. _faq_watershed_bem_meshes:

My watershed BEM meshes look incorrect
--------------------------------------

After using :ref:`mne watershed_bem` or :func:`mne.bem.make_watershed_bem`
you might find that the BEM meshes for the brain, inner skull, outer skull,
and/or scalp surfaces do not look correct in :func:`mne.viz.plot_alignment`
and :func:`mne.viz.plot_bem`.

MNE relies on FreeSurfer's mri_watershed_ to compute the BEM meshes.
Freesurfer's watershed bem strategy is to:

1. Compute the outer skin (scalp) surface
2. Shrink outer skin inward make the "outer skull"
3. Compute brain surface
4. Expand brain surface outward to make the "inner skull"

A common problem is to see:

    the surface inner skull is not completely inside surface outer skull

When looking at the meshes, the inner skull surface (expanded brain surface)
will have defects, and these defects will protrude into the outer skull surface
(shrunken scalp surface). In these cases, you can try (in rough ascending
order of difficulty):

.. highlight:: console

1. Changing the ``--preflood`` / ``-p`` parameter in
   :ref:`mne watershed_bem`.
2. Changing the ``--atlas`` and ``--gcaatlas`` options of
   :ref:`mne watershed_bem`.
3. Manually editing the meshes (see `this tutorial
   <https://github.com/ezemikulan/blender_freesurfer>`__.
4. Manually running mri_watershed_ with various FreeSurfer flags (e.g.,
   ``-less`` to fix the output).
5. Going farther back in your Freesurfer pipeline to fix the problem.
   In particular, ``mri/brainmask.mgz`` could be incorrectly generated by the
   autorecon1_ step and contain some dura and/or skull within the brain mask.
   You can check by using freeview_ or some other MRI-viewing tool.

   - Consult the Freesurfer docs on `fixing errors
     <https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/TroubleshootingDataV6.0#Fixingerrors>`__.
   - Try tweaking the mri_normalize_ parameters `via xopts
     <https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg20991.html>`__,
     e.g.::

         $ mri_normalize -mprage -b 20 -n 5

   - Try `manually setting the control points and/or using -gentle
     <https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg11658.html>`__.
   - Examine the talairach transformation to see if it's not quite right,
     and if it's not, `adjust it manually
     <https://surfer.nmr.mgh.harvard.edu/fswiki/Edits>`__.
   - Search the `FreeSurfer listserv`_ for other ideas

   It can be helpful to run ``recon_all -autorecon1 -xopts xopts.txt`` in a
   clean directory first to see if this fixes everything, and, if not, then
   resorting to manual control point setting and/or talairach adjustment.
   Once everything looks good at the end of ``-autorecon1``, you can then run
   :ref:`mne watershed_bem` to see if the output is good. Once it is
   (and once brainmask.mgz is correct), you can then proceed with
   ``recon_all -autorecon2`` and ``recon_all -autorecon3`` to effectively
   complete all ``recon_all`` steps.

.. highlight:: python


References
----------

.. footbibliography::

.. LINKS

.. _`the most current version`: https://github.com/mne-tools/mne-python/releases/latest
.. _`minimal working example`: https://en.wikipedia.org/wiki/Minimal_Working_Example
.. _mri_watershed: http://freesurfer.net/fswiki/mri_watershed
.. _mri_normalize: https://surfer.nmr.mgh.harvard.edu/fswiki/mri_normalize
.. _freeview: https://surfer.nmr.mgh.harvard.edu/fswiki/FreeviewGuide/FreeviewIntroduction
.. _`FreeSurfer listserv`: https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/
.. _autorecon1: https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllDevTable
