.. include:: links.inc

.. _preinstall:

Before you install
==================

.. contents::
   :local:
   :depth: 1

Overview of the MNE tools suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`MNE-C <manual/c_reference>` was the initial stage of this project,
  providing a set of interrelated command-line and GUI programs focused on
  computing cortically constrained Minimum Norm Estimates from MEG and EEG
  data. These tools were written in C by Matti Hämäläinen.

- :doc:`MNE-python <python_reference>` reimplements the functionality of MNE-C,
  and extends considerably the analysis and visualization capabilities.
  MNE-python is collaboratively developed and has more than 150 contributors.

- The :doc:`manual/matlab` provides a MATLAB interface to the .fif file format
  and other MNE data structures, and provides example MATLAB implementations of
  some of the core analysis functionality of MNE-C. It is distributed alongside
  MNE-C, and can also be downloaded from the `MNE-MATLAB git repository`_.

- :doc:`MNE-CPP <mne_cpp>` provides core MNE functionality implemented in C++
  and is primarily intended for embedded and real-time applications.

There are also Python tools for easily importing MEG data from the Human
Connectome Project for use with MNE-python (`MNE-HCP`_), and tools for
managing MNE projects so that they comply with the Brain Imaging Data Structure
specification (`MNE-BIDS`_).

What should I install?
^^^^^^^^^^^^^^^^^^^^^^

If you intend only to perform ERP, ERF, or other sensor-level analyses,
:doc:`MNE-python <install_mne_python>` is all you need. If you
prefer MATLAB over Python, probably all you need is :doc:`MNE-C
<install_mne_c>` — the MNE MATLAB toolbox is distributed with it.

If you want to transform sensor recordings into estimates of localized brain
activity, you will most likely need:

- :doc:`FreeSurfer <install_freesurfer>` to convert structural MRI scans into
  models of the scalp, inner/outer skull, and cortical surfaces

- :doc:`MNE-C <install_mne_c>` for constructing and solving a boundary-element
  model of tissue conductance, and for aligning coordinate frames between the
  structural MRI and the digitizations of M/EEG sensor locations

- :doc:`MNE-python <install_mne_python>` can be used for everything else

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

**Next:** :doc:`install_mne_python`
