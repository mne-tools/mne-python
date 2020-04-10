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

.. warning:: Information about device light wavelength is stored in
             channel names. Manual modification of channel names is not
             recommended.


Storing of optode locations
===========================

NIRs devices consist of light sources and light detectors.
A channel is formed by source-detector pairs.
MNE stores the location of the channels, sources, and detectors.

"""  # noqa:E501
