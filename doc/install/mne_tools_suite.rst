Overview of the MNE tools suite
===============================

MNE-Python is an open-source Python module for processing, analysis, and
visualization of functional neuroimaging data (EEG, MEG, sEEG, ECoG, and
fNIRS). There are several related or interoperable software packages that you
may also want to install, depending on your analysis needs.

Related software
^^^^^^^^^^^^^^^^

- MNE-C was the initial stage of this project,
  providing a set of interrelated command-line and GUI programs focused on
  computing cortically constrained Minimum Norm Estimates from MEG and EEG
  data. These tools were written in C by Matti Hämäläinen, and are
  documented `here <MNE-C manual_>`_. See :ref:`install_mne_c` for installation
  instructions.

- MNE-Python reimplements the functionality of MNE-C, extends considerably the
  analysis and visualization capabilities, and adds support for additional data
  types like functional near-infrared spectroscopy (fNIRS). MNE-Python is
  collaboratively developed and has more than 200 contributors.

- `MNE-MATLAB`_ provides a MATLAB interface to the .fif
  file format and other MNE data structures, and provides example MATLAB
  implementations of some of the core analysis functionality of MNE-C. It is
  distributed alongside MNE-C, and can also be downloaded from the `MNE-MATLAB`_ GitHub repository.

- :ref:`MNE-CPP <mne_cpp>` provides core MNE functionality implemented in
  C++ and is primarily intended for embedded and real-time applications.

There is also a growing ecosystem of other Python packages that work alongside
MNE-Python, including packages for:

.. note:: Something missing?
    :class: sidebar

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
- automatic labeling of ICA components (`MNE-ICAlabel`_)
- phase-amplitude coupling (`pactools`_)
- representational similarity analysis (`rsa`_)
- microstate analysis (`microstate`_)
- connectivity analysis using dynamic imaging of coherent sources (DICS)
  (`conpy`_)
- other connectivity algorithms (`MNE-Connectivity`_)
- general-purpose statistical analysis of M/EEG data (`eelbrain`_)
- post-hoc modification of linear models (`posthoc`_)
- a python implementation of the Preprocessing Pipeline (PREP) for EEG data
  (`pyprep`_)
- automatic multi-dipole localization and uncertainty quantification with
  the Bayesian algorithm SESAME (`sesameeg`_)
- GLM and group level analysis of near-infrared spectroscopy data (`MNE-NIRS`_)
- high-level EEG Python library for all kinds of EEG inverse solutions (`invertmeeg`_)
- All-Resolutions Inference (ARI) for statistically valid circular inference
  and effect localization (`MNE-ARI`_)
- real-time analysis (`MNE-Realtime`_)
- non-parametric sequential analyses and adaptive sample size determination (`niseq`_)

What should I install?
^^^^^^^^^^^^^^^^^^^^^^

If you intend only to perform ERP, ERF, or other sensor-level analyses,
:ref:`MNE-Python <standard-instructions>` is all you need. If you prefer to
work with
shell scripts and the Unix command line, or prefer MATLAB over Python, probably
all you need is :doc:`MNE-C <mne_c>` — the MNE MATLAB toolbox is distributed
with it — although note that the C tools and the MATLAB toolbox are less
actively developed than the MNE-Python module, and hence are considerably less
feature-complete.

If you want to transform sensor recordings into estimates of localized brain
activity, you will need MNE-Python, plus :doc:`FreeSurfer <freesurfer>` to
convert structural MRI scans into models of the scalp, inner/outer skull, and
cortical surfaces (specifically, for command-line functions
:ref:`mne flash_bem`, :ref:`mne watershed_bem`, and
:ref:`mne make_scalp_surfaces`).


Getting help
^^^^^^^^^^^^

Help with installation is available through the `MNE Forum`_. See the
:ref:`help` page for more information.


.. LINKS:

.. _MNELAB: https://github.com/cbrnr/mnelab
.. _autoreject: https://autoreject.github.io/
.. _alphaCSC: https://alphacsc.github.io/
.. _picard: https://pierreablin.github.io/picard/
.. _pactools: https://pactools.github.io/
.. _rsa: https://github.com/wmvanvliet/mne-rsa
.. _microstate: https://github.com/wmvanvliet/mne_microstates
.. _conpy: https://aaltoimaginglanguage.github.io/conpy/
.. _eelbrain: https://eelbrain.readthedocs.io/en/stable/index.html
.. _posthoc: https://users.aalto.fi/~vanvlm1/posthoc/python/
.. _pyprep: https://github.com/sappelhoff/pyprep
.. _sesameeg: https://pybees.github.io/sesameeg
.. _invertmeeg: https://github.com/LukeTheHecker/invert
.. _MNE-ARI: https://github.com/john-veillette/mne_ari
.. _niseq: https://github.com/john-veillette/niseq

.. include:: ../links.inc
