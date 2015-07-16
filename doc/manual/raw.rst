.. _ch_raw:

Reading raw data
################

.. contents:: Contents
   :local:
   :depth: 2

MNE-Python offers a unified format for reading raw data. Once a raw file has
been read in using any of the following methods, the returned `mne.io.Raw`-type
object can be processed using any of the MNE-Python tools.


Elekta NeuroMag (.fif)
---------------------------

Raw FIF files can be loaded using :func:`mne.io.read_raw_fif`.

.. note::
    If the data were recorded with MaxShield on and have not been processed
    with MaxFilter, they may need to be loaded with
    ``mne.io.read_raw_fif(..., allow_maxshield=True)``.


Brainvision (.vhdr)
-------------------
Brainvision EEG files can be read in using :func:`mne.io.read_raw_brainvision`.


4D Neuroimaging MagnesWH3600 (.pdf)
-----------------------------------
Data from MagnesWH3600 systems can be read with :func:`mne.io.read_raw_bdf`.


European data format (.edf)
---------------------------
EDF files can be read in using :func:`mne.io.read_raw_edf`.


Biosemi data format (.bdf)
--------------------------
BDF files can be read in using :func:`mne.io.read_raw_edf`.


EGI simple binary (.egi)
------------------------
EGI simple binary files can be read in using :func:`mne.io.read_raw_egi`.


KIT data (.sqd)
---------------
KIT files can be read in using :func:`mne.io.read_raw_kit`.


Arrays (from memory)
--------------------
Arbitrary (e.g., simulated or manually read in) raw data can be constructed
from memory by making use of :class:`mne.io.RawArray` and :func:`mne.io.create_info`.
