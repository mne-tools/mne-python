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


"""  # noqa:E501

###############################################################################
# Loading legacy data in csv or tsv format
# ========================================
#
# Many legacy fNIRS measurements are stored in csv and tsv formats.
# These formats are not officially supported in MNE as there is no
# standardisation of the file format -
# the naming and ordering of channels, the type and scaling of data, and
# specification of sensor positions varies between each vendor.
# Instead, we suggest that data is converted to the format approved by the
# Society for functional near-infrared spectroscopy called
# [SNIRF](https://github.com/fNIRS/snirf), the society provides converters
# to translate your data to SNIRF.
# However, due to the prevalence of these legacy files we provide
# a template example of how you may read data in t/csv formats.

import numpy as np
import pandas as pd
import mne


###############################################################################
# First, we generate an example csv file.
# This is only required for this example, this step would be skipped
# if you have actual data you wish to load.
# We simulate 16 channels with 100 samples of data and save this to a file
# called `fnirs.csv`.

pd.DataFrame(np.random.normal(size=(16, 100))).to_csv("fnirs.csv")


###############################################################################
# Next, we will load the example csv file.
# The metadata must be specified manually as the csv file does not contain
# information about channel names, types, sample rate etc.

data = pd.read_csv('fnirs.csv')

# In MNE the naming of channels MUST follow this structure of
# `S#_D# type` or `S#_D# wavelength`, where # is replaced by the appropriate
# source and detector number.
ch_names = ['D1_S1 hbo', 'D1_S1 hbr', 'D1_S2 hbo', 'D1_S2 hbr',
            'D1_S3 hbo', 'D1_S3 hbr', 'D1_S4 hbo', 'D1_S4 hbr',
            'D2_S5 hbo', 'D2_S5 hbr', 'D2_S6 hbo', 'D2_S6 hbr',
            'D2_S7 hbo', 'D2_S7 hbr', 'D2_S8 hbo', 'D2_S8 hbr']

ch_types = ['hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr']
sfreq = 10.  # Hz


###############################################################################
# Finally, the data can be converted in to a MNE data structure.
# The metadata above is used to create an :class:`~mne.info` structure,
# and this is combined with the data to create
# an MNE :class:`~mne.io.Raw` object, for more details on how continuous
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
# such as creating regions of interest,
# this is particularly important for fNIRS as information about the
# distance between optodes is required to convert the optical density data
# in to an estimate of the haemoglobin concentrations.
# MNE provides methods to load standard sensor configurations (montages) from
# some vendors, which is demonstrated below.
# However, many fNIRS researchers use custom optode montages, in this case
# you can generate your own `.elc` file
# (see [example file](https://github.com/mne-tools/mne-python/blob/main/mne/channels/data/montages/standard_1020.elc))
# and load that instead.
# Below is an example of how to load the optode positions for a Artinis Octomon
# device.

raw.set_montage('artinis-octamon')
# To load a custom montage use:
# raw.set_montage('/path/to/custom/montage.elc')

# View the position of optodes in 2D to confirm the positions are correct.
raw.plot_sensors()


###############################################################################
# It is also possible to view the location of the sources (red),
# detectors (black), and channel (white lines and orange dots) locations
# in a 3D representation to validate the positions were loaded correctly.

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)

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
# NIRS devices consist of light sources and light detectors.
# A channel is formed by source-detector pairs.
# MNE stores the location of the channels, sources, and detectors.
#
#
# .. warning:: Information about device light wavelength is stored in
#              channel names. Manual modification of channel names is not
#              recommended.
