# -*- coding: utf-8 -*-
"""
.. _loading-raw-tutorial:

Loading raw data
================

This tutorial covers the basics of loading EEG/MEG data into Python. We'll
start by importing the Python modules we need:
"""

import os
import mne

###############################################################################
#
# Supported data formats
# ^^^^^^^^^^^^^^^^^^^^^^
#
# When MNE-Python loads sensor data, the data are stored in a Python object of
# type :class:`mne.io.Raw`. Specialized loading functions are provided for the
# raw data file formats from a variety of equipment manufacturers. All raw data
# input/output functions in MNE-Python are found in :mod:`mne.io` and start
# with *read_raw_*; see the documentation for each function for more info on
# reading specific file types.
#
# As seen in the table below, there are also a few formats defined by other
# neuroimaging analysis software packages that are supported (EEGLAB,
# FieldTrip). Like the equipment-specific loading functions, these will also
# return an object of class :class:`~mne.io.Raw`.
#
# =========   =================  =========  ===================================
# Data type   File format        Extension  MNE-Python function
# =========   =================  =========  ===================================
# MEG         Artemis123         .bin       :func:`mne.io.read_raw_artemis123`
#
# MEG         4-D Neuroimaging   .dir       :func:`mne.io.read_raw_bti` / BTI
#
# MEG         CTF                .dir       :func:`mne.io.read_raw_ctf`
#
# MEG         Elekta Neuromag    .fif       :func:`mne.io.read_raw_fif`
#
# MEG         KIT                .sqd       :func:`mne.io.read_raw_kit`
#
# MEG and     FieldTrip          .mat       :func:`mne.io.read_raw_fieldtrip`
# EEG
#
# EEG         Brainvision        .vhdr      :func:`mne.io.read_raw_brainvision`
#
# EEG         Biosemi data       .bdf       :func:`mne.io.read_raw_bdf` format
#
# EEG         Neuroscan CNT      .cnt       :func:`mne.io.read_raw_cnt`
#
# EEG         European data      .edf       :func:`mne.io.read_raw_edf` format
#
# EEG         EEGLAB             .set       :func:`mne.io.read_raw_eeglab`
#
# EEG         EGI simple         .egi       :func:`mne.io.read_raw_egi` binary
#
# EEG         EGI MFF format     .mff       :func:`mne.io.read_raw_egi`
#
# EEG         eXimia             .nxe       :func:`mne.io.read_raw_eximia`
#
# EEG         General data       .gdf       :func:`mne.io.read_raw_gdf` format
#
# EEG         Nicolet            .data      :func:`mne.io.read_raw_nicolet`
# =========   =================  =========  ===================================
#
# .. note:: MNE-Python is aware of the measurement units used by each
#     manufacturer, and will always convert data into a common internal
#     representation. See :ref:`the section on internal representation <units>`
#     for more information.
#
#
# Loading example data
# ^^^^^^^^^^^^^^^^^^^^
#
# MNE-Python provides several example datasets that can be downloaded with just
# a few lines of code. Functions for downloading example datasets are in the
# :mod:`mne.datasets` submodule; here we'll use
# :func:`mne.datasets.sample.data_path` to download a dataset called
# ``"sample"``, which contains EEG and MEG data from one subject performing an
# audiovisual experiment, along with structural MRI scans for that subject.
# :func:`~mne.datasets.sample.data_path` will return the folder location where
# it put the downloaded dataset; you can navigate there with your file browser
# if you want to examine the files yourself.

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')

###############################################################################
# Now we can load the sample data; since it has a ``.fif`` extension we'll use
# :func:`~mne.io.read_raw_fif`.  This will return a :class:`~mne.io.Raw`
# object, which we'll store in a variable called ``raw``.

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

###############################################################################
# Notice that :func:`~mne.io.read_raw_fif` also takes a ``preload`` parameter,
# which determines whether the data will be copied into RAM or not.  Some
# operations (such as filtering) require that the data be preloaded, but it is
# possible use ``preload=False`` and then copy raw data into memory later using
# the :meth:`~mne.io.Raw.load_data` method.
#
# You can get a glimpse of the basic details of a :class:`~mne.io.Raw` object
# by printing it:

print(raw)

###############################################################################
# To extract more detailed information, see the tutorial on
# :ref:`querying-raw-tutorial`.
#
# .. note::
#
#     There are ``data_path`` functions for several example datasets in
#     MNE-Python (e.g., :func:`mne.datasets.kiloword.data_path`,
#     :func:`mne.datasets.spm_face.data_path`, etc). All of them will check the
#     default download location first to see if the dataset is already on your
#     computer, and only download it if necessary.  The default download
#     location is also configurable; see the documentation of any of the
#     ``data_path`` functions for more information.
