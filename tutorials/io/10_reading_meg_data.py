# -*- coding: utf-8 -*-
r"""
.. _tut-imorting-meg-data:

===============================
Importing data from MEG devices
===============================

This section describes how to read data for various MEG manufacturers.

.. _import-neuromag:

MEGIN/Elekta Neuromag VectorView and TRIUX (.fif)
=================================================

Neuromag Raw FIF files can be loaded using :func:`mne.io.read_raw_fif`.

If the data were recorded with MaxShield on and have not been processed
with MaxFilter, they may need to be loaded with
``mne.io.read_raw_fif(..., allow_maxshield=True)``.


.. _import-artemis:

Artemis123 (.bin)
=================
MEG data from the Artemis123 system can be read with\
:func:`mne.io.read_raw_artemis123`.


.. _import-bti:

4-D Neuroimaging / BTI data (dir)
=================================

MNE-Python provides :func:`mne.io.read_raw_bti` to read and convert 4D / BTI
data. This reader function will by default replace the original channel names,
typically composed of the letter ``A`` and the channel number with Neuromag.
To import the data, the following input files are mandatory:

- A data file (typically c,rfDC)
  containing the recorded MEG time series.

- A hs_file
  containing the digitizer data.

- A config file
  containing acquisition information and metadata.

By default :func:`mne.io.read_raw_bti` assumes that these three files are located
in the same folder.

.. note:: While reading the reference or compensation channels,
          the compensation weights are currently not processed.
          As a result, the :class:`mne.io.Raw` object and the corresponding fif
          file does not include information about the compensation channels
          and the weights to be applied to realize software gradient
          compensation. If the data are saved in the Magnes system are already
          compensated, there will be a small error in the forward calculations,
          whose significance has not been evaluated carefully at this time.


.. _import-ctf:

CTF data (dir)
==============

The function :func:`mne.io.read_raw_ctf` can be used to read CTF data.

CTF Polhemus data
-----------------

The function :func:`mne.channels.read_dig_polhemus_isotrak` can be used to read
Polhemus data.

Applying software gradient compensation
---------------------------------------

Since the software gradient compensation employed in CTF
systems is a reversible operation, it is possible to change the
compensation status of CTF data in the data files as desired. This
section contains information about the technical details of the
compensation procedure and a description of
:func:`mne.io.Raw.apply_gradient_compensation`.

The raw instances returned by :func:`mne.io.read_raw_ctf` contain several
compensation matrices which are employed to suppress external disturbances
with help of the reference channel data. The reference sensors are
located further away from the brain than the helmet sensors and
are thus measuring mainly the external disturbances rather than magnetic
fields originating in the brain. Most often, a compensation matrix
corresponding to a scheme nicknamed *Third-order gradient
compensation* is employed.

Let us assume that the data contain :math:`n_1` MEG
sensor channels, :math:`n_2` reference sensor
channels, and :math:`n_3` other channels.
The data from all channels can be concatenated into a single vector

.. math::    x = [x_1^T x_2^T x_3^T]^T\ ,

where :math:`x_1`, :math:`x_2`,
and :math:`x_3` are the data vectors corresponding
to the MEG sensor channels, reference sensor channels, and other
channels, respectively. The data before and after compensation,
denoted here by :math:`x_{(0)}` and :math:`x_{(k)}`, respectively,
are related by

.. math::    x_{(k)} = M_{(k)} x_{(0)}\ ,

where the composite compensation matrix is

.. math::    M_{(k)} = \begin{bmatrix}
                I_{n_1} & C_{(k)} & 0 \\
                0 & I_{n_2} & 0 \\
                0 & 0 & I_{n_3}
                \end{bmatrix}\ .

In the above, :math:`C_{(k)}` is a :math:`n_1` by :math:`n_2` compensation
data matrix corresponding to compensation "grade" :math:`k`.
It is easy to see that

.. math::    M_{(k)}^{-1} = \begin{bmatrix}
                I_{n_1} & -C_{(k)} & 0 \\
                0 & I_{n_2} & 0 \\
                0 & 0 & I_{n_3}
                \end{bmatrix}\ .

To convert from compensation grade :math:`k` to :math:`p` one
can simply multiply the inverse of one compensate compensation matrix
by another and apply the product to the data:

.. math::    x_{(k)} = M_{(k)} M_{(p)}^{-1} x_{(p)}\ .

This operation is performed by :meth:`mne.io.Raw.apply_gradient_compensation`.


.. _import-kit:

Ricoh/KIT MEG system data (.con/.sqd)
=====================================

MNE-Python includes the :func:`mne.io.read_raw_kit` and
:func:`mne.read_epochs_kit` to read and convert Ricoh/KIT MEG data.

.. admonition:: Channel naming
    :class: sidebar warning

    In MNE 0.21 This reader function will by default replace the original channel names,
    which typically with index starting with zero, with ones with an index starting
    with one. In 0.22 it will use native names when possible. Use the
    ``standardize_names`` argument to control this behavior.

To import continuous data, only the input .sqd or .con file is needed. For
epochs, an Nx3 matrix containing the event number/corresponding trigger value
in the third column is needed.

The following input files are optional:

- A KIT marker file (mrk file) or an array-like containing the locations of
  the HPI coils in the MEG device coordinate system.
  These data are used together with the elp file to establish the coordinate
  transformation between the head and device coordinate systems.

- A Polhemus points file (elp file) or an array-like
  containing the locations of the fiducials and the head-position
  indicator (HPI) coils. These data are usually given in the Polhemus
  head coordinate system.

- A Polhemus head shape data file (hsp file) or an array-like
  containing locations of additional points from the head surface.
  These points must be given in the same coordinate system as that
  used for the elp file.

Modern Ricoh systems may encode this information it the file itself, in which
case ``mrk``, ``elp``, and ``hsp`` can all be ``None`` and the data will be
read from the file itself.

.. note::
   The output fif file will use the Neuromag head coordinate system convention,
   see :ref:`coordinate_systems`. A coordinate transformation between the
   Polhemus head coordinates and the Neuromag head coordinates is included.

By default, KIT-157 systems assume the first 157 channels are the MEG channels,
the next 3 channels are the reference compensation channels, and channels 160
onwards are designated as miscellaneous input channels (MISC 001, MISC 002,
etc.).
By default, KIT-208 systems assume the first 208 channels are the MEG channels,
the next 16 channels are the reference compensation channels, and channels 224
onwards are designated as miscellaneous input channels (MISC 001, MISC 002,
etc.).

In addition, it is possible to synthesize the digital trigger channel (STI 014)
from available analog trigger channel data by specifying the following
parameters:

- A list of trigger channels (stim) or default triggers with order: '<' | '>'
  Channel-value correspondence when converting KIT trigger channels to a
  Neuromag-style stim channel. By default, we assume the first eight
  miscellaneous channels are trigger channels. For '<', the largest values are
  assigned to the first channel (little endian; default). For '>', the largest
  values are assigned to the last channel (big endian). Can also be specified
  as a list of trigger channel indexes.
- The trigger channel slope (slope) : '+' | '-'
  How to interpret values on KIT trigger channels when synthesizing a
  Neuromag-style stim channel. With '+', a positive slope (low-to-high)
  is interpreted as an event. With '-', a negative slope (high-to-low)
  is interpreted as an event.
- A stimulus threshold (stimthresh) : float
  The threshold level for accepting voltage changes in KIT trigger
  channels as a trigger event.

The synthesized trigger channel data value at sample :math:`k` will
be:

.. math::    s(k) = \sum_{p = 1}^n {t_p(k) 2^{p - 1}}\ ,

where :math:`t_p(k)` are the thresholded
from the input channel data d_p(k):

.. math::    t_p(k) = \Bigg\{ \begin{array}{l}
                 0 \text{  if  } d_p(k) \leq t\\
                 1 \text{  if  } d_p(k) > t
             \end{array}\ .

The threshold value :math:`t` can
be adjusted with the ``stimthresh`` parameter.


.. _import-fieldtrip:

FieldTrip MEG/EEG data (.mat)
=============================

MNE-Python includes :func:`mne.io.read_raw_fieldtrip`, :func:`mne.read_epochs_fieldtrip` and :func:`mne.read_evoked_fieldtrip` to read data coming from FieldTrip.

The data is imported directly from a ``.mat`` file.

The ``info`` parameter can be explicitly set to ``None``. The import functions will still work but:

#. All channel locations will be in head coordinates.
#. Channel orientations cannot be guaranteed to be accurate.
#. All channel types will be set to generic types.

This is probably fine for anything that does not need that information, but if you intent to do things like interpolation of missing channels, source analysis or look at the RMS pairs of planar gradiometers, you most likely run into problems.

It is **highly recommended** to provide the ``info`` parameter as well. The ``info`` dictionary can be extracted by loading the original raw data file with the corresponding MNE-Python functions::

    original_data = mne.io.read_raw_fiff('original_data.fif', preload=False)
    original_info = original_data.info
    data_from_ft = mne.read_evoked_fieldtrip('evoked_data.mat', original_info)

The imported data can have less channels than the original data. Only the information for the present ones is extracted from the ``info`` dictionary.

As of version 0.17, importing FieldTrip data has been tested on a variety of systems with the following results:

+----------+-------------------+-------------------+-------------------+
| System   | Read Raw Data     | Read Epoched Data | Read Evoked Data  |
+==========+===================+===================+===================+
| BTI      | Works             | Untested          | Untested          |
+----------+-------------------+-------------------+-------------------+
| CNT      | Data imported as  | Data imported as  | Data imported as  |
|          | microvolts.       | microvolts.       | microvolts.       |
|          | Otherwise fine.   | Otherwise fine.   | Otherwise fine.   |
+----------+-------------------+-------------------+-------------------+
| CTF      | Works             | Works             | Works             |
+----------+-------------------+-------------------+-------------------+
| EGI      | Mostly Ok. Data   | Mostly Ok. Data   | Mostly Ok. Data   |
|          | imported as       | imported as       | imported as       |
|          | microvolts.       | microvolts.       | microvolts.       |
|          | FieldTrip does    | FieldTrip does    | FieldTrip does    |
|          | not apply         | not apply         | not apply         |
|          | calibration.      | calibration.      | calibration.      |
+----------+-------------------+-------------------+-------------------+
| KIT      | Does not work.    | Does not work.    | Does not work.    |
|          | Channel names are | Channel names are | Channel names are |
|          | different in      | different in      | different in      |
|          | MNE-Python and    | MNE-Python and    | MNE-Python and    |
|          | FieldTrip.        | FieldTrip.        | FieldTrip.        |
+----------+-------------------+-------------------+-------------------+
| Neuromag | Works             | Works             | Works             |
+----------+-------------------+-------------------+-------------------+
| eximia   | Works             | Untested          | Untested          |
+----------+-------------------+-------------------+-------------------+

Creating MNE data structures from arbitrary data (from memory)
==============================================================

Arbitrary (e.g., simulated or manually read in) raw data can be constructed
from memory by making use of :class:`mne.io.RawArray`, :class:`mne.EpochsArray`
or :class:`mne.EvokedArray` in combination with :func:`mne.create_info`.

This functionality is illustrated in :ref:`tut-creating-data-structures`.
Using 3rd party
libraries such as `NEO <https://github.com/NeuralEnsemble/python-neo>`__ in
combination with these functions abundant electrophysiological file formats can
be easily loaded into MNE.
"""  # noqa:E501

# %%
