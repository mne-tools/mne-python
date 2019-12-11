.. include:: ../links.inc

Before you install
==================

.. contents::
   :local:
   :depth: 1

Overview of the MNE tools suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- MNE-C was the initial stage of this project,
  providing a set of interrelated command-line and GUI programs focused on
  computing cortically constrained Minimum Norm Estimates from MEG and EEG
  data. These tools were written in C by Matti Hämäläinen, and are
  documented `here <MNE-C manual_>`_.

- :doc:`MNE-Python <../python_reference>` reimplements the functionality of
  MNE-C, and extends considerably the analysis and visualization capabilities.
  MNE-Python is collaboratively developed and has more than 150 contributors.

- The :ref:`mne_matlab` provides a MATLAB interface to the .fif file
  format and other MNE data structures, and provides example MATLAB
  implementations of some of the core analysis functionality of MNE-C. It is
  distributed alongside MNE-C, and can also be downloaded from the `MNE-MATLAB
  git repository`_.

- :doc:`MNE-CPP <../mne_cpp>` provides core MNE functionality implemented in
  C++ and is primarily intended for embedded and real-time applications.

There is also a growing ecosystem of other Python packages that work alongside
MNE-Python, including packages for:

.. sidebar:: Something missing?

    If you know of a package that is related but not listed here, feel free to
    :ref:`make a pull request <contributing>` to add it to this list.

- a graphical user interface for MNE-Python (`MNELAB`_)
- easily importing MEG data from the Human Connectome Project for
  use with MNE-Python (`MNE-HCP`_)
- managing MNE projects so that they comply with the `Brain
  Imaging Data Structure`_ specification (`MNE-BIDS`_)
- automatic bad channel detection and interpolation (`autoreject`_)
- convolutional sparse dictionary learning and waveform shape estimation
  (`alphaCSC`_)
- independent component analysis (ICA) with good performance on real data
  (`PICARD`_)
- phase-amplitude coupling (`pactools`_)
- representational similarity analysis (`rsa`_)
- microstate analysis (`microstate`_)
- connectivity analysis using dynamic imaging of coherent sources (DICS)
  (`conpy`_)
- general-purpose statistical analysis of M/EEG data (`eelbrain`_)
- post-hoc modification of linear models (`posthoc`_)
- a python implementation of the Preprocessing Pipeline (PREP) for EEG data
  (`pyprep`_)


What should I install?
^^^^^^^^^^^^^^^^^^^^^^

If you intend only to perform ERP, ERF, or other sensor-level analyses,
:doc:`MNE-Python <mne_python>` is all you need. If you prefer to work with
shell scripts and the Unix command line, or prefer MATLAB over Python, probably
all you need is :doc:`MNE-C <mne_c>` — the MNE MATLAB toolbox is distributed
with it — although note that the C tools and the MATLAB toolbox are less
actively developed than the MNE-Python module, and hence are considerably less
feature-complete.

If you want to transform sensor recordings into estimates of localized brain
activity, you will most likely also need :doc:`FreeSurfer <freesurfer>` to
convert structural MRI scans into models of the scalp, inner/outer skull, and
cortical surfaces (specifically, for command-line functions
:ref:`gen_mne_flash_bem`, :ref:`gen_mne_watershed_bem`, and
:ref:`gen_mne_make_scalp_surfaces`).


Getting help
^^^^^^^^^^^^

Help with installation is available through the `MNE mailing list`_ and
`MNE gitter channel`_. See the :ref:`help` page for more information.


**Next:** :doc:`mne_python`


.. LINKS:

.. _MNELAB: https://github.com/cbrnr/mnelab
.. _autoreject: https://autoreject.github.io/
.. _alphaCSC: https://alphacsc.github.io/
.. _picard: https://pierreablin.github.io/picard/
.. _pactools: https://pactools.github.io/
.. _rsa: https://github.com/wmvanvliet/rsa
.. _microstate: https://github.com/wmvanvliet/mne_microstates
.. _conpy: https://aaltoimaginglanguage.github.io/conpy/
.. _eelbrain: https://eelbrain.readthedocs.io/en/stable/index.html
.. _posthoc: https://users.aalto.fi/~vanvlm1/posthoc/python/
.. _pyprep: https://github.com/sappelhoff/pyprep
