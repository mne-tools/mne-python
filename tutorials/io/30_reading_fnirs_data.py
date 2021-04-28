# -*- coding: utf-8 -*-
r"""
.. _tut-importing-fnirs-data:

=================================
Importing data from fNIRS devices
=================================

MNE includes various functions and utilities for reading NIRS
data and optode locations.

fNIRS devices consist of light sources and light detectors. A channel is formed
by source-detector pairs. MNE stores the location of the channels, sources, and
detectors.

.. warning:: Information about device light wavelength is stored in channel
             names. Manual modification of channel names is not recommended.

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


Loading legacy data in CSV or TSV format
========================================

.. warning:: This method is not supported and users are discoraged to use it.
             You should convert your data to the
             `SNIRF <https://github.com/fNIRS/snirf>`_ format using the tools
             provided by the Society for functional Near-Infrared Spectroscopy,
             and then load it using :func:`mne.io.read_raw_snirf`.

fNIRS measurements can have a non-standardised format that is not supported by
MNE and cannot be converted easily into SNIRF. This legacy data is often in CSV
or TSV format, we show here a way to load it even though it is not officially
supported by MNE due to the lack of standardisation of the file format (the
naming and ordering of channels, the type and scaling of data, and
specification of sensor positions varies between each vendor). You will likely
have to adapt this depending on the system from which your CSV originated.
"""  # noqa:E501

import numpy as np
import pandas as pd
import mne

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# First, we generate an example CSV file which will then be loaded in to MNE.
# This step would be skipped if you have actual data you wish to load.
# We simulate 16 channels with 100 samples of data and save this to a file
# called fnirs.csv.

pd.DataFrame(np.random.normal(size=(16, 100))).to_csv("fnirs.csv")


###############################################################################
#
# .. warning:: The channels must be ordered in haemoglobin pairs, such that for
#              a single channel all the types are in subsequent indices. The
#              type order must be 'hbo' then 'hbr'.
#              The data below is already in the correct order and may be
#              used as a template for how data must be stored.
#              If the order that your data is stored is different to the
#              mandatory formatting, then you must first read the data with
#              channel naming according to the data structure, then reorder
#              the channels to match the required format.
#
# Next, we will load the example CSV file.

data = pd.read_csv('fnirs.csv')


###############################################################################
# Then, the metadata must be specified manually as the CSV file does not
# contain information about channel names, types, sample rate etc.
#
# .. warning:: In MNE the naming of channels MUST follow the structure of
#              ``S#_D# type`` where # is replaced by the appropriate source and
#              detector numbers and type is either ``hbo``, ``hbr`` or the
#              wavelength.

ch_names = ['S1_D1 hbo', 'S1_D1 hbr', 'S2_D1 hbo', 'S2_D1 hbr',
            'S3_D1 hbo', 'S3_D1 hbr', 'S4_D1 hbo', 'S4_D1 hbr',
            'S5_D2 hbo', 'S5_D2 hbr', 'S6_D2 hbo', 'S6_D2 hbr',
            'S7_D2 hbo', 'S7_D2 hbr', 'S8_D2 hbo', 'S8_D2 hbr']
ch_types = ['hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr',
            'hbo', 'hbr', 'hbo', 'hbr']
sfreq = 10.  # in Hz


###############################################################################
# Finally, the data can be converted in to an MNE data structure.
# The metadata above is used to create an :class:`mne.Info` data structure,
# and this is combined with the data to create an MNE :class:`~mne.io.Raw`
# object. For more details on the info structure see :ref:`tut-info-class`, and
# for additional details on how continuous data is stored in MNE see
# :ref:`tut-raw-class`.
# For a more extensive description of how to create MNE data structures from
# raw array data see :ref:`tut_creating_data_structures`.

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
raw = mne.io.RawArray(data, info, verbose=True)


###############################################################################
# Applying standard sensor locations to imported data
# ---------------------------------------------------
#
# Having information about optode locations may assist in your analysis.
# Beyond the general benefits this provides (e.g. creating regions of interest,
# etc), this is may be particularly important for fNIRS as information about
# the optode locations is required to convert the optical density data in to an
# estimate of the haemoglobin concentrations.
# MNE provides methods to load standard sensor configurations (montages) from
# some vendors, and this is demonstrated below.
# Some handy tutorials for understanding sensor locations, coordinate systems,
# and how to store and view this information in MNE are:
# :ref:`tut-sensor-locations`, :ref:`plot_source_alignment`, and
# :ref:`ex-eeg-on-scalp`.
#
# Below is an example of how to load the optode positions for an Artinis
# OctaMon device.
#
# .. note:: It is also possible to create a custom montage from a file for
#           fNIRS with :func:`mne.channels.read_custom_montage` by setting
#           ``coord_frame`` to ``'mri'``.

montage = mne.channels.make_standard_montage('artinis-octamon')
raw.set_montage(montage)

# View the position of optodes in 2D to confirm the positions are correct.
raw.plot_sensors()


###############################################################################
# To validate the positions were loaded correctly it is also possible to view
# the location of the sources (red), detectors (black), and channels (white
# lines and orange dots) in a 3D representation.
# The ficiduals are marked in blue, green and red.
# See :ref:`plot_source_alignment` for more details.

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

trans = mne.channels.compute_native_head_t(montage)

fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(
    raw.info, trans=trans, subject='fsaverage', subjects_dir=subjects_dir,
    surfaces=['brain', 'head'], coord_frame='mri', dig=True, show_axes=True,
    fnirs=['channels', 'pairs', 'sources', 'detectors'], fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=90, elevation=90, distance=0.5,
                    focalpoint=(0., -0.01, 0.02))
