# -*- coding: utf-8 -*-
r"""
.. _tut-imorting-eeg-data:

===============================
Importing data from EEG devices
===============================

MNE includes various functions and utilities for reading EEG
data and electrode locations.

.. contents:: Page contents
   :local:
   :depth: 2


.. _import-bv:

BrainVision (.vhdr, .vmrk, .eeg)
================================

The BrainVision file format consists of three separate files:

1. A text header file (``.vhdr``) containing meta data
2. A text marker file (``.vmrk``) containing information about events in the
   data
3. A binary data file (``.eeg``) containing the voltage values of the EEG

Both text files are based on the
`Microsoft Windows INI format <https://en.wikipedia.org/wiki/INI_file>`_
consisting of:

* sections marked as ``[square brackets]``
* comments marked as ``; comment``
* key-value pairs marked as ``key=value``

A documentation for core BrainVision file format is provided by Brain Products.
You can view the specification hosted on the
`Brain Products website <https://www.brainproducts.com/productdetails.php?id=21&tab=5>`_

BrainVision EEG files can be read in using :func:`mne.io.read_raw_brainvision`
with the ``.vhdr`` header file as an input.

.. warning:: Renaming BrainVision files can be problematic due to their
             multifile structure. See this
             `example <https://mne.tools/mne-bids/stable/auto_examples/rename_brainvision_files.html#sphx-glr-auto-examples-rename-brainvision-files-py>`_
             for an instruction.

.. note:: For *writing* BrainVision files, you can use the Python package
          `pybv <https://pypi.org/project/pybv/>`_.

.. _import-edf:

European data format (.edf)
===========================

EDF and EDF+ files can be read using :func:`mne.io.read_raw_edf`.

`EDF (European Data Format) <http://www.edfplus.info/specs/edf.html>`_ and
`EDF+ <http://www.edfplus.info/specs/edfplus.html>`_ are 16-bit formats.

The EDF+ files may contain an annotation channel which can be used to store
trigger information. These annotations are available in ``raw.annotations``.

Saving EDF files is not supported natively yet. `This gist
<https://gist.github.com/skjerns/bc660ef59dca0dbd53f00ed38c42f6be>`__
can be used to save any mne.io.Raw into EDF/EDF+/BDF/BDF+.


.. _import-biosemi:

BioSemi data format (.bdf)
==========================

The `BDF format <http://www.biosemi.com/faq/file_format.htm>`_ is a 24-bit
variant of the EDF format used by EEG systems manufactured by BioSemi. It can
be imported with :func:`mne.io.read_raw_bdf`.

BioSemi amplifiers do not perform "common mode noise rejection" automatically.
The signals in the EEG file are the voltages between each electrode and CMS
active electrode, which still contain some CM noise (50 Hz, ADC reference
noise, etc., see `the BioSemi FAQ <https://www.biosemi.com/faq/cms&drl.htm>`__
for further detail).
Thus, it is advisable to choose a reference (e.g., a single channel like Cz,
average of linked mastoids, average of all electrodes, etc.) on import of
BioSemi data to avoid losing signal information. The data can be re-referenced
later after cleaning if desired.

.. warning:: The data samples in a BDF file are represented in a 3-byte
             (24-bit) format. Since 3-byte raw data buffers are not presently
             supported in the fif format these data will be changed to 4-byte
             integers in the conversion.


.. _import-gdf:

General data format (.gdf)
==========================

GDF files can be read in using :func:`mne.io.read_raw_gdf`.

`GDF (General Data Format) <https://arxiv.org/abs/cs/0608052>`_ is a flexible
format for biomedical signals that overcomes some of the limitations of the
EDF format. The original specification (GDF v1) includes a binary header
and uses an event table. An updated specification (GDF v2) was released in
2011 and adds fields for additional subject-specific information (gender,
age, etc.) and allows storing several physical units and other properties.
Both specifications are supported in MNE.


.. _import-cnt:

Neuroscan CNT data format (.cnt)
================================

CNT files can be read in using :func:`mne.io.read_raw_cnt`.
The channel locations can be read from a montage or the file header. If read
from the header, the data channels (channels that are not assigned to EOG, ECG,
EMG or misc) are fit to a sphere and assigned a z-value accordingly. If a
non-data channel does not fit to the sphere, it is assigned a z-value of 0.

.. warning::
    Reading channel locations from the file header may be dangerous, as the
    x_coord and y_coord in ELECTLOC section of the header do not necessarily
    translate to absolute locations. Furthermore, EEG-electrode locations that
    do not fit to a sphere will distort the layout when computing the z-values.
    If you are not sure about the channel locations in the header, use of a
    montage is encouraged.


.. _import-egi:

EGI simple binary (.egi)
========================

EGI simple binary files can be read in using :func:`mne.io.read_raw_egi`.
The EGI raw files are simple binary files with a header and can be exported
from using the EGI Netstation acquisition software.


.. _import-mff:

EGI MFF (.mff)
==============
These files can also be read with :func:`mne.io.read_raw_egi`.


.. _import-set:

EEGLAB set files (.set)
=======================

EEGLAB .set files can be read in using :func:`mne.io.read_raw_eeglab`
and :func:`mne.read_epochs_eeglab`.


.. _import-nicolet:

Nicolet (.data)
===============
These files can be read with :func:`mne.io.read_raw_nicolet`.


.. _import-nxe:

eXimia EEG data (.nxe)
======================

EEG data from the Nexstim eXimia system can be read in using the
:func:`mne.io.read_raw_eximia` function.


.. _import-persyst:

Persyst EEG data (.lay, .dat)
=============================

EEG data from the Persyst system can be read in using the
:func:`mne.io.read_raw_persyst` function.

Note that not all the subject metadata may be properly read in
due to the fact that Persyst changes its specification
sometimes from version to version. Please submit an issue, or
pull request if you encounter a problem.

Nihon Kohden EEG data (.EEG, .21E, .PNT, .LOG)
==============================================

EEG data from the Nihon Kohden (NK) system can be read using the
:func:`mne.io.read_raw_nihon` function.

Files with the following extensions will be read:

- The ``.EEG`` file contains the actual raw EEG data.
- The ``.PNT`` file contains the metadata related to the recording, such
  as the measurement date.
- The ``.LOG`` file contains annotations for the recording.
- The ``.21E`` file contains the channel and electrode
  recording system information.

Reading ``.11D``, ``.CMT``, ``.CN2``, and ``.EDF`` files is currently not
supported.

Note that not all the subject metadata may be properly read in
due to the fact that NK changes the specification
sometimes from version to version. Please submit an issue, or
pull request if you encounter a problem.


Setting EEG references
======================

The preferred method for applying an EEG reference in MNE is
:func:`mne.set_eeg_reference`, or equivalent instance methods like
:meth:`raw.set_eeg_reference() <mne.io.Raw.set_eeg_reference>`. By default,
the data are assumed to already be properly referenced. See
:ref:`tut-set-eeg-ref` for more information.

Reading electrode locations and head shapes for EEG recordings
==============================================================

Some EEG formats (EGI, EDF/EDF+, BDF) neither contain electrode location
information nor head shape digitization information. Therefore, this
information has to be provided separately. For that purpose all raw instances
have a :meth:`mne.io.Raw.set_montage` method to set electrode locations.

When using the locations of the fiducial points the digitization data
are converted to the MEG head coordinate system employed in the
MNE software, see :ref:`coordinate_systems`.
"""  # noqa:E501
