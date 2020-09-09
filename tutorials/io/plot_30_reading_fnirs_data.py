# -*- coding: utf-8 -*-
r"""
.. _tut-importing-fnirs-data:

=================================
Importing data from fNIRS devices
=================================

MNE includes various functions and utilities for reading NIRS
data and optode locations.

.. contents:: Page contents
   :local:
   :depth: 2


.. _import-nirx:

NIRx (directory)
================================

NIRx recordings can be read in using :func:`mne.io.read_raw_nirx`.
The NIRx device stores data directly to a directory with multiple file types,
MNE extracts the appropriate information from each file.


.. _import-snirf:

SNIRF (.snirf)
================================

Data stored in the SNIRF format can be read in
using :func:`mne.io.read_raw_snirf`.

.. warning:: The SNIRF format has provisions for many different types of NIRS
             recordings. MNE currently only supports continuous wave data
             stored in the .snirf format.


Storing of optode locations
===========================

NIRs devices consist of light sources and light detectors.
A channel is formed by source-detector pairs.
MNE stores the location of the channels, sources, and detectors.


.. warning:: Information about device light wavelength is stored in
             channel names. Manual modification of channel names is not
             recommended.

"""  # noqa:E501
