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

.. warning:: Manual modification of channel names and info fields is not
             recommended. fNIRS data is encoded in MNE using a combination
             of the channel name and the info structure. Additionally, there
             is an expected order of channels that must be maintained. Any
             change to the expected structure will cause an error.

             Channels names must follow the structure
             ``S#_D# value``
             where S# is the source number (e.g. S12 for source 12),
             D# is the detector number (e.g. D4 for detector 4),
             and the value is either the light wavelength if the data type is
             raw intensity or optical density, or the chromophore type if the
             data is hbo or hbr. For example ``S1_D2 760`` is valid as is
             ``S11_D1 850`` and ``S1_D2 hbo`` and ``S1_D2 hbr``. However,
             these examples are not valid ``D1_S2 hbo``, ``S1_D2_760``.

             Channels with the same source-detector pairing must be stored
             in consecutive order. For example
             ``["S11_D2 hbo", "S11_D2 hbr", "S1_D2 hbo", "S11_D2 hbr"]``
             is acceptable, but
             ``["S11_D2 hbo", "S1_D2 hbo", "S11_D2 hbr", "S11_D2 hbr"]``
             is not. Further, the order of type must be maintained for all
             fNIRS channels so the following is not valid
             ``["S11_D2 hbo", "S11_D2 hbr", "S1_D2 hbr", "S11_D2 hbo"]``.

             For raw amplitude measurements and for optical density data
             the wavelength information must be stored in
             ``info["chs"][ii]["loc"][9]``
             and it must match the channel name. For example the channel
             ``S11_D2 760`` must have the value 760 stored in the loc[9] field.

"""  # noqa:E501
