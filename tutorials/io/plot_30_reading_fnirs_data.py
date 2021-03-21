# -*- coding: utf-8 -*-
r"""
.. _tut-importing-fnirs-data:

=================================
Importing data from fNIRS devices
=================================

MNE includes various functions and utilities for reading NIRS
data and optode locations.

.. _import-nirx:

NIRx (directory)
================================

NIRx recordings can be read in using :func:`mne.io.read_raw_nirx`.
The NIRx device stores data directly to a directory with multiple file types,
MNE extracts the appropriate information from each file.
MNE only supports NIRx files recorded with NIRStar version 15.0 and above.


.. _import-snirf:

SNIRF (.snirf)
================================

Data stored in the SNIRF format can be read in
using :func:`mne.io.read_raw_snirf`.

.. warning:: The SNIRF format has provisions for many different types of NIRS
             recordings. MNE currently only supports continuous wave data
             stored in the .snirf format.


.. _import-boxy:

BOXY (.txt)
===========

BOXY recordings can be read in using :func:`mne.io.read_raw_boxy`.
The BOXY software and ISS Imagent I and II devices are frequency domain
systems that store data in a single ``.txt`` file containing what they call
(with MNE's name for that type of data in parens):

- DC
    All light collected by the detector (``fnirs_cw_amplitude``)
- AC
    High-frequency modulated light intensity (``fnirs_fd_ac_amplitude``)
- Phase
    Phase of the modulated light (``fnirs_fd_phase``)

DC data is stored as the type ``fnirs_cw_amplitude`` because it
collects both the modulated and any unmodulated light, and hence is analogous
to what is collected by continuous wave systems such as NIRx. This helps with
conformance to SNIRF standard types.

These raw data files can be saved by the acquisition devices as parsed or
unparsed ``.txt`` files, which affects how the data in the file is organised.
MNE will read either file type and extract the raw DC, AC, and Phase data.
If triggers are sent using the ``digaux`` port of the recording hardware, MNE
will also read the ``digaux`` data and create annotations for any triggers.


Loading legacy data in csv or tsv format
========================================

.. warning:: This method is not supported. You should convert your data
             to the `SNIRF <https://github.com/fNIRS/snirf>`_
             format using the tools provided by the Society
             for functional Near-Infrared Spectroscopy, and then load it
             using :func:`mne.io.read_raw_snirf`.

             This method will only work for data that has already been
             converted to oxyhaemoglobin and deoxyhaemoglobin. It will not work
             with raw intensity data or optical density data.

Many legacy fNIRS measurements are stored in csv and tsv formats.
These formats are not officially supported in MNE as there is no
standardisation of the file format -
the naming and ordering of channels, the type and scaling of data, and
specification of sensor positions varies between each vendor.
Instead, we suggest that data is converted to the format approved by the
Society for functional Near-Infrared Spectroscopy called
`SNIRF <https://github.com/fNIRS/snirf>`_,
they provide a number of tools to convert your legacy
data to the SNIRF format.
However, due to the prevalence of these legacy files we provide
a template example of how you may read data in t/csv formats.
"""  # noqa:E501

import numpy as np
import pandas as pd
import mne

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# First, we generate an example csv file which will then be loaded in to MNE.
# This step would be skipped if you have actual data you wish to load.
# We simulate 16 channels with 100 samples of data and save this to a file
# called fnirs.csv.

pd.DataFrame(np.random.normal(size=(16, 100))).to_csv("fnirs.csv")


###############################################################################
#
# .. warning:: You must ensure that the channel naming structure follows
#              the MNE format of S#_D# type.
#              The channels must be ordered in pairs haemoglobin pairs,
#              such that for a single channel all the types are in subsequent
#              indices. The type order must be hbo then hbr.
#              The data below is already in the correct order and may be
#              used as a template for how data must be stored.
#              If the order that your data is stored is different to the
#              mandatory formatting, then you must first read the data with
#              channel naming according to the data structure, then reorder
#              the channels to match the required format.
#
# Next, we will load the example csv file.
# The metadata must be specified manually as the csv file does not contain
# information about channel names, types, sample rate etc.

data = pd.read_csv('fnirs.csv')

# In MNE the naming of channels MUST follow this structure of
# `S#_D# type` where # is replaced
# by the appropriate source and detector number and type is
# either hbo or hbr.

ch_names = ['S1_D1 hbo', 'S1_D1 hbr', 'S2_D1 hbo', 'S2_D1 hbr',
            'S3_D1 hbo', 'S3_D1 hbr', 'S4_D1 hbo', 'S4_D1 hbr',
            'S5_D2 hbo', 'S5_D2 hbr', 'S6_D2 hbo', 'S6_D2 hbr',
            'S7_D2 hbo', 'S7_D2 hbr', 'S8_D2 hbo', 'S8_D2 hbr']
ch_types = ['hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr']
sfreq = 10.  # Hz


###############################################################################
# Finally, the data can be converted in to an MNE data structure.
# The metadata above is used to create an :class:`mne.Info` data structure,
# and this is combined with the data to create
# an MNE :class:`~mne.io.Raw` object. For more details on the info structure
# see :ref:`tut-info-class`, and for additional details on how continuous
# data is stored in MNE see :ref:`tut-raw-class`.
# For a more extensive description of how to create MNE data structures from
# raw array data see :ref:`tut_creating_data_structures`.

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
raw = mne.io.RawArray(data, info, verbose=True)


###############################################################################
# Applying standard sensor locations to imported data
# ---------------------------------------------------
#
# Having information about optode locations may assist in your analysis.
# Beyond the general benefits this provides
# (e.g. creating regions of interest, etc),
# this is particularly important for fNIRS as information about the
# distance between optodes is required to convert the optical density data
# in to an estimate of the haemoglobin concentrations.
# MNE provides methods to load standard sensor configurations (montages) from
# some vendors, and this is demonstrated below.
# Some handy tutorials for understanding sensor locations, coordinate systems,
# and how to store and view this information in MNE are:
# :ref:`tut-sensor-locations`, :ref:`plot_source_alignment`, and
# :ref:`ex-eeg-scalp`.
#
# Below is an example of how to load the optode positions for an Artinis
# OctaMon device. However, many fNIRS researchers use custom optode montages,
# in this case you can generate your own .elc file (see `example file
# <https://github.com/mne-tools/mne-python/blob/main/mne/channels/data
# /montages/standard_1020.elc>`_) and load that instead.

raw.set_montage('artinis-octamon')

# View the position of optodes in 2D to confirm the positions are correct.
raw.plot_sensors()


###############################################################################
# To validate the positions were loaded correctly it is also possible
# to view the location of the sources (red), detectors (black),
# and channel (white lines and orange dots) locations in a 3D representation.
# The ficiduals are marked in blue, green and red.
# See :ref:`plot_source_alignment` for more details.

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(raw.info, show_axes=True,
                             subject='fsaverage', coord_frame='mri',
                             trans='fsaverage', surfaces=['brain', 'head'],
                             fnirs=['channels', 'pairs',
                                    'sources', 'detectors'],
                             dig=True, mri_fiducials=True,
                             subjects_dir=subjects_dir, fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90, distance=0.4,
                    focalpoint=(0., -0.01, 0.02))


###############################################################################
# Storing of optode locations
# ===========================
#
# fNIRS devices consist of light sources and light detectors.
# A channel is formed by source-detector pairs.
# MNE stores the location of the channels, sources, and detectors.
#
#
# .. warning:: Information about device light wavelength is stored in
#              channel names. Manual modification of channel names is not
#              recommended.
