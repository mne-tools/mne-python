# -*- coding: utf-8 -*-
r"""
.. _tut-imorting-eeg-data:

===============================
Importing data from EEG devices
===============================

MNE includes various functions and utilities for reading EEG data and electrode
locations.

.. _import-bv:

BrainVision (.vhdr, .vmrk, .eeg)
================================

The BrainVision file format consists of three separate files:

1. A text header file (``.vhdr``) containing meta data.
2. A text marker file (``.vmrk``) containing information about events in the
   data.
3. A binary data file (``.eeg``) containing the voltage values of the EEG.

Both text files are based on the `INI format <https://en.wikipedia.org/wiki/INI_file>`_
consisting of

* sections marked as ``[square brackets]``,
* comments marked as ``; comment``,
* and key-value pairs marked as ``key=value``.

Brain Products provides documentation for their core BrainVision file format.
The format specification is hosted on the
`Brain Products website <https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/>`_.

BrainVision EEG files can be read using :func:`mne.io.read_raw_brainvision`,
passing the ``.vhdr`` header file as the argument.

.. warning:: Renaming BrainVision files can be problematic due to their
             multi-file structure. See this
             `example <https://mne.tools/mne-bids/stable/auto_examples/rename_brainvision_files.html#sphx-glr-auto-examples-rename-brainvision-files-py>`_
             for instructions.

.. note:: For *writing* BrainVision files, take a look at the :py:mod:`mne.export`
          module, which used the `pybv <https://pypi.org/project/pybv/>`_ Python
          package.


.. _import-edf:

European data format (.edf)
===========================

`EDF <http://www.edfplus.info/specs/edf.html>`_ and
`EDF+ <http://www.edfplus.info/specs/edfplus.html>`_ files can be read using
:func:`mne.io.read_raw_edf`. Both variants are 16-bit formats.

EDF+ files may contain annotation channels which can be used to store trigger
and event information. These annotations are available in ``raw.annotations``.

Writing EDF files is not supported natively yet. `This gist
<https://gist.github.com/skjerns/bc660ef59dca0dbd53f00ed38c42f6be>`__ or
`MNELAB <https://github.com/cbrnr/mnelab>`_ (both of which use
`pyedflib <https://github.com/holgern/pyedflib>`_ under the hood) can be used
to export any :class:`mne.io.Raw` object to EDF/EDF+/BDF/BDF+.


.. _import-biosemi:

BioSemi data format (.bdf)
==========================

The `BDF format <http://www.biosemi.com/faq/file_format.htm>`_ is a 24-bit
variant of the EDF format used by EEG systems manufactured by BioSemi. It can
be imported with :func:`mne.io.read_raw_bdf`.

BioSemi amplifiers do not perform "common mode noise rejection" automatically.
The signals in the EEG file are the voltages between each electrode and the CMS
active electrode, which still contain some CM noise (50 Hz, ADC reference
noise, etc.). The `BioSemi FAQ <https://www.biosemi.com/faq/cms&drl.htm>`__
provides more details on this topic.
Therefore, it is advisable to choose a reference (e.g., a single channel like Cz,
average of linked mastoids, average of all electrodes, etc.) after importing
BioSemi data to avoid losing signal information. The data can be re-referenced
later after cleaning if desired.

.. warning:: Data samples in a BDF file are represented in a 3-byte
             (24-bit) format. Since 3-byte raw data buffers are not presently
             supported in the FIF format, these data will be changed to 4-byte
             integers in the conversion.


.. _import-gdf:

General data format (.gdf)
==========================

GDF files can be read using :func:`mne.io.read_raw_gdf`.

`GDF (General Data Format) <https://arxiv.org/abs/cs/0608052>`_ is a flexible
format for biomedical signals that overcomes some of the limitations of the
EDF format. The original specification (GDF v1) includes a binary header
and uses an event table. An updated specification (GDF v2) was released in
2011 and adds fields for additional subject-specific information (gender,
age, etc.) and allows storing several physical units and other properties.
Both specifications are supported by MNE.


.. _import-cnt:

Neuroscan CNT (.cnt)
====================

CNT files can be read using :func:`mne.io.read_raw_cnt`.
Channel locations can be read from a montage or the file header. If read
from the header, the data channels (channels that are not assigned to EOG, ECG,
EMG or MISC) are fit to a sphere and assigned a z-value accordingly. If a
non-data channel does not fit to the sphere, it is assigned a z-value of 0.

.. warning::
    Reading channel locations from the file header may be dangerous, as the
    x_coord and y_coord in the ELECTLOC section of the header do not necessarily
    translate to absolute locations. Furthermore, EEG electrode locations that
    do not fit to a sphere will distort the layout when computing the z-values.
    If you are not sure about the channel locations in the header, using a
    montage is encouraged.


.. _import-egi:

EGI simple binary (.egi)
========================

EGI simple binary files can be read using :func:`mne.io.read_raw_egi`.
EGI raw files are simple binary files with a header and can be exported by the
EGI Netstation acquisition software.


.. _import-mff:

EGI MFF (.mff)
==============

EGI MFF files can be read with :func:`mne.io.read_raw_egi`.


.. _import-set:

EEGLAB files (.set, .fdt)
=========================

EEGLAB .set files (which sometimes come with a separate .fdt file) can be read
using :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab`.


.. _import-nicolet:

Nicolet (.data)
===============

These files can be read with :func:`mne.io.read_raw_nicolet`.


.. _import-nxe:

eXimia EEG data (.nxe)
======================

EEG data from the Nexstim eXimia system can be read with
:func:`mne.io.read_raw_eximia`.


.. _import-persyst:

Persyst EEG data (.lay, .dat)
=============================

EEG data from the Persyst system can be read with
:func:`mne.io.read_raw_persyst`.

Note that subject metadata may not be properly imported because Persyst
sometimes changes its specification from version to version. Please let us know
if you encounter a problem.


Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)
==============================================

EEG data from the Nihon Kohden (NK) system can be read using the
:func:`mne.io.read_raw_nihon` function.

Files with the following extensions will be read:

- The ``.eeg`` file contains the actual raw EEG data.
- The ``.pnt`` file contains metadata related to the recording such as the
  measurement date.
- The ``.log`` file contains annotations for the recording.
- The ``.21e`` file contains channel and electrode information.

Reading ``.11d``, ``.cmt``, ``.cn2``, and ``.edf`` files is currently not
supported.

Note that not all subject metadata may be properly read because NK changes the
specification sometimes from version to version. Please let us know if you
encounter a problem.


XDF data (.xdf, .xdfz)
======================

MNE-Python does not support loading
`XDF <https://github.com/sccn/xdf/wiki/Specifications>`_ files out of the box,
because the inherent flexibility of the XDF format makes it difficult to
provide a one-size-fits-all function. For example, XDF supports signals from
various modalities recorded with different sampling rates. However, it is
relatively straightforward to import only a specific stream (such as EEG
signals) using the `pyxdf <https://github.com/xdf-modules/pyxdf>`_ package.
See :ref:`ex-read-xdf` for a simple example.

A more sophisticated version, which supports selection of specific streams as
well as converting marker streams into annotations, is available in
`MNELAB <https://github.com/cbrnr/mnelab>`_. If you want to use this
functionality in a script, MNELAB records its history (View - History), which
contains all commands required to load an XDF file after successfully loading
that file with the graphical user interface.


Setting EEG references
======================

The preferred method for applying an EEG reference in MNE is
:func:`mne.set_eeg_reference`, or equivalent instance methods like
:meth:`raw.set_eeg_reference() <mne.io.Raw.set_eeg_reference>`. By default,
the data are assumed to already be properly referenced. See
:ref:`tut-set-eeg-ref` for more information.


Reading electrode locations and head shapes for EEG recordings
==============================================================

Some EEG formats (e.g., EGI, EDF/EDF+, BDF) contain neither electrode locations
nor head shape digitization information. Therefore, this information has to be
provided separately. For that purpose, all raw instances have a
:meth:`mne.io.Raw.set_montage` method to set electrode locations.

When using locations of fiducial points, the digitization data are converted to
the MEG head coordinate system employed in the MNE software, see
:ref:`coordinate_systems`.
"""  # noqa:E501

# %%
