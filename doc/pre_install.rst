.. include:: links.inc

.. _preinstall:

Before you install
==================

.. contents::
   :local:
   :depth: 1

Overview of the MNE tools suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The project started with what is now called :ref:`MNE-C <c_reference>` — a set
of interrelated command-line and GUI programs focused on computing cortically
constrained Minimum Norm Estimates from MEG and EEG data. These tools were
written in C by Matti Hämäläinen.

:ref:`MNE-python <api_reference>` reimplements the functionality of MNE-C,
and extends considerably the analysis and visualization capabilities.
MNE-python is collaboratively developed and has more than 150 contributors.

The :ref:`ch_matlab` provides a MATLAB interface to the .fif file format and
other MNE data structures, and provides example MATLAB implementations of
some of the core analysis functionality of MNE-C. It is distributed alongside
MNE-C, and can also be downloaded from its
`git repository <mne-matlab-git>`_.

:ref:`mne_cpp` provides core MNE functionality implemented in C++ and is
primarily intended for embedded and real-time applications.

There are also python tools for easily importing MEG data from the Human
Connectome Project for use with MNE-python (`MNE-HCP`_), and tools for
managing MNE projects so that they comply with the Brain Imaging Data Structure
specification (`MNE-BIDS`_).

What should I install?
^^^^^^^^^^^^^^^^^^^^^^

If you intend only to perform ERP, ERF, or other sensor-level analyses,
:doc:`MNE-python <install_mne_python>` is all you need. If you
prefer MATLAB over python, probably all you need is :doc:`MNE-C
<install_mne_c>` — the MNE MATLAB toolbox is distributed with it.

If you want to transform sensor recordings into estimates of localized brain
activity, you will most likely need:

- :ref:`FreeSurfer <install_freesurfer>` (to convert structural MRI scans into
  models of the scalp, inner/outer skull, and cortical surfaces)

- :ref:`MNE-C <install_mne_c>` (for constructing and solving a boundary-element
  model of tissue conductance, and for aligning coordinate frames between the
  structural MRI and the digitizations of M/EEG sensor locations)

- :ref:`MNE-python <install_python_and_mne_python>` (for everything else)

Getting help
^^^^^^^^^^^^

There are three main channels for obtaining help with MNE software tools.

The `MNE mailing list`_ and `MNE gitter channel`_ are a good
place to start for both troubleshooting and general questions.  If you want to
request new features or if you're confident that you have found a bug, please
create a new issue on the `GitHub issues page`_. When reporting bugs,
please try to replicate the bug with the MNE-python sample data, and make every
effort to simplify your example script to only the elements necessary to
replicate the bug.

**Next:** :doc:`install_freesurfer`
