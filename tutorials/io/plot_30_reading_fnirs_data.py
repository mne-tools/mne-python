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


Storing of optode locations
===========================

NIRs devices consist of light sources and light detectors.
A channel is formed by source-detector pairs.
MNE stores the location of the channels, sources, and detectors.


.. warning:: Information about device light wavelength is stored in
             channel names. Manual modification of channel names is not
             recommended.

"""  # noqa:E501




###############################################################################
# Fake some data
# --------------
#
# Here we just create some fake data with the correct names for an
# artinis octamon system

import mne
from mne.channels import make_standard_montage
from mne.channels.tests.test_standard_montage import _simulate_artinis_octamon

raw_intensity = _simulate_artinis_octamon()


###############################################################################
# Next we load the montage
# -------------------------------------------
#
# And apply montage to data

montage = make_standard_montage("artinis-octamon")
raw_intensity.set_montage(montage)


###############################################################################
# View location of sensors over brain surface
# -------------------------------------------
#
# Here we validate

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)

fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(raw_intensity.info, show_axes=True,
                             dig=True, mri_fiducials=True,
                             subject='fsaverage', coord_frame='mri',
                             trans='fsaverage', surfaces=['brain', 'head'],
                             fnirs=['channels', 'pairs',
                                    'sources', 'detectors'],
                             subjects_dir=subjects_dir, fig=fig)

mne.viz.set_3d_view(figure=fig, azimuth=70, elevation=100, distance=0.4,
                    focalpoint=(0., -0.01, 0.02))


###############################################################################
# TODO Can compare trans to the internal fsaverage-trans.fif as a test:
trans = mne.channels.compute_native_head_t(montage)

# TODO Also works just with:
raw_intensity.set_montage('artinis-octamon')
