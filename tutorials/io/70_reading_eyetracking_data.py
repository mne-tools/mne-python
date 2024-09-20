# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

r"""
.. _tut-importing-eyetracking-data:

=======================================
Importing Data from Eyetracking devices
=======================================

Eyetracking devices record a persons point of gaze, usually in relation to a
screen. Typically, gaze position (also referred to as eye or pupil position)
and pupil size are recorded as separate channels. This section describes how to
read data from supported eyetracking manufacturers.

MNE-Python provides functions for reading eyetracking data. When possible,
MNE-Python will internally convert and store eyetracking data according to an
SI unit (for example radians for position data, and meters for pupil size).

.. note:: If you have eye tracking data in a format that MNE does not support
          yet, you can try reading it using other tools and create an MNE
          object from a numpy array. Then you can use
          :func:`mne.preprocessing.eyetracking.set_channel_types_eyetrack`
          to assign the correct eyetrack channel types.

.. seealso:: Some MNE functions may not be available to eyetracking and other
             physiological data, because MNE does not consider them to be data
             channels. See the :ref:`glossary` for more information.

.. _import-eyelink_asc:

SR Research (Eyelink) (.asc)
============================

.. note:: MNE-Python currently only supports reading Eyelink eyetracking data
          stored in the ASCII (.asc) format.

Eyelink recordings are stored in the Eyelink Data Format (EDF; .edf), which are
binary files and thus relatively complex to support. To make the data in EDF
files accessible, Eyelink provides the application EDF2ASC, which converts EDF
files to a plain text ASCII format (.asc). These files can be imported
into MNE using :func:`mne.io.read_raw_eyelink`.

.. note:: The Eyelink Data Format (EDF), should not be confused
          with the European Data Format, the common EEG data format that also
          uses the .edf extension.

Supported measurement types from Eyelink files include eye position, pupil
size, saccadic velocity, resolution, and head position (for recordings
collected in remote mode). Eyelink files often report ocular events (blinks,
saccades, and fixations), MNE will store these events as `mne.Annotations`.
Blink annotation descriptions will be ``'BAD_blink'``. For more information
on the various measurement types that can be present in Eyelink files. read below.

Eye Position Data
-----------------

Eyelink samples can report eye position data in pixels, units of visual
degrees, or as raw pupil coordinates. Samples are written as (x, y) coordinate
pairs (or two pairs for binocular data). The type of position data present in
an ASCII file will be detected automatically by MNE. The three types of
position data are explained below.

Gaze
^^^^
Gaze position data report the estimated (x, y) pixel coordinates of the
participants's gaze on the stimulus screen, compensating for head position
changes and distance from  the screen. This datatype may be preferable if you
are interested in knowing where the participant was looking at on the stimulus
screen. The default (0, 0) location for Eyelink systems is at the top left of
the screen.

This may be best demonstrated with an example. In the file plotted below,
eyetracking data was recorded while the participant read text on a display.
In this file, as the participant read the each line from left to right, the
x-coordinate increased. When the participant moved their gaze down to read a
new line, the y-coordinate *increased*, which is why the ``ypos_right`` channel
in the plot below increases over time (for example, at about 4-seconds, and
at about 8-seconds).

.. seealso::

    :ref:`tut-eyetrack`
"""

# %%
import mne

# %%
fpath = mne.datasets.misc.data_path() / "eyetracking" / "eyelink"
fname = fpath / "px_textpage_ws.asc"
raw = mne.io.read_raw_eyelink(fname, create_annotations=["blinks"])
cal = mne.preprocessing.eyetracking.read_eyelink_calibration(
    fname,
    screen_distance=0.7,
    screen_size=(0.53, 0.3),
    screen_resolution=(1920, 1080),
)[0]
mne.preprocessing.eyetracking.convert_units(raw, calibration=cal, to="radians")

# %%
# Visualizing the data
# ^^^^^^^^^^^^^^^^^^^^

# %%
cal.plot()

# %%
custom_scalings = dict(pupil=1e3)
raw.pick(picks="eyetrack").plot(scalings=custom_scalings)

# %%
# Note that we passed a custom `dict` to the ``'scalings'`` argument of
# `mne.io.Raw.plot`. This is because MNE expects the data to be in SI units
# (radians for eyegaze data, and meters for pupil size data), but we did not convert
# the pupil size data in this example.

# %%
# Head-Referenced Eye Angle (HREF)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# HREF position data measures eye rotation angles relative to the head. It does
# not take into account changes in subject head position and angle, or distance
# from the stimulus screen. This datatype might be preferable for analyses
# that are interested in eye movement velocities and amplitudes, or for
# simultaneous and EEG/MEG eyetracking recordings where eye position data are
# used to identify EOG artifacts.
#
# HREF coordinates are stored in the ASCII file as integer values, with 260 or
# more units per visual degree, however MNE will convert and store these
# coordinates in radians. The (0, 0) point of HREF data is arbitrary, as the
# relationship between the screen position and the coordinates changes as the
# subject's head moves.
#
# Below is the same text reading recording that we plotted above, except a new
# ASCII file was generated, this time using HREF eye position data.


# %%
fpath = mne.datasets.misc.data_path() / "eyetracking" / "eyelink"
fname_href = fpath / "HREF_textpage_ws.asc"
raw = mne.io.read_raw_eyelink(fname_href, create_annotations=["blinks"])
custom_scalings = dict(pupil=1e3)
raw.pick(picks="eyetrack").plot(scalings=custom_scalings)

# %%
# Pupil Position
# ^^^^^^^^^^^^^^
#
# Pupil position data contains (x, y) coordinate pairs from the eye camera.
# It has not been converted to pixels (gaze) or eye angles (HREF). Most use
# cases do not require this data type, and caution should be taken when
# analyzing raw pupil positions. Note that when plotting data from a
# ``Raw`` object containing raw pupil position data, the plot scalings
# will likely be incorrect. You can pass custom scalings into the ``scalings``
# parameter of `mne.io.Raw.plot` so that the signals are legible when plotting.

# %%
# .. warning:: If a calibration was not performed prior to data collection, the
#              EyeLink system cannot convert raw pupil position data to pixels
#              (gaze) or eye angle (HREF).

# %%
# Pupil Size Data
# ---------------
# Pupil size is measured by the EyeLink system at up to 500 samples per second.
# It may be reported as pupil *area*, or pupil *diameter* (i.e. the diameter
# of a circle/ellipse model fit to the pupil area).
# Which of these datatypes you get is specified by your recording- and/or your
# EDF2ASC settings. The pupil size data is not calibrated and reported in
# arbitrary units. Typical pupil *area* data range between 800 to 2000 units,
# with a precision of 1 unit, while pupil *diameter* data range between
# 1800-3000 units.
#
# Velocity, resolution, and head position data
# --------------------------------------------
# Eyelink files can produce data on saccadic velocity, resolution, and head
# position for each sample in the file. MNE will read in these data if they are
# present in the file, but will label their channel types as ``'misc'``.
#
# .. warning:: Eyelink's EDF2ASC API allows for modification of the data
#              and format that is converted to ASCII. However, MNE-Python
#              assumes a specific structure, which the default parameters of
#              EDF2ASC follow. ASCII files should be tab-deliminted, and both
#              Samples and Events should be output. If the data were recorded
#              at 2000Hz, timestamps should be floating point numbers. Manual
#              modification of ASCII conversion via EDF2ASC is not recommended.
