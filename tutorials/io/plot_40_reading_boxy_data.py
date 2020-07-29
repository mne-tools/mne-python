# -*- coding: utf-8 -*-
r"""
.. _tut-importing-boxy-data:

=========================================================
Importing data from BOXY software and ISS Imagent devices
=========================================================

MNE includes various functions and utilities for reading optical imaging
data and optode locations.

.. contents:: Page contents
   :local:
   :depth: 2


.. _import-boxy:

BOXY (directory)
================================

BOXY recordings can be read in using :func:`mne.io.read_raw_boxy`.
The BOXY software and Imagent devices store data in a single .txt file
containing DC, AC, and Phase information for each source and detector
combination. Recording settings, such as the number of sources/detectors, and
the sampling rate of the recording, are also saved at the beginning of this
file. MNE will extract the raw DC, AC, and Phase data, along with the recording
settings.

"""  # noqa:E501
