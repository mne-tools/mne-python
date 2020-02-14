"""
.. _tut-head-pos:

================================================
Extracting and visualizing subject head movement
================================================

Continuous head movement can be encoded during MEG recordings by use of
HPI coils that continuously emit sinusoidal signals. These signals can then be
extracted from the recording and used to estimate head position as a function
of time. Here we show an example of how to do this, and how to visualize
the result.

HPI frequencies
---------------

First let's load a short bit of raw data where the subject intentionally moved
their head during the recording. Its power spectral density shows five peaks
(most clearly visible in the gradiometers) corresponding to the HPI coil
frequencies, plus other peaks related to power line interference (60 Hz and
harmonics).
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op

import mne

print(__doc__)

data_path = op.join(mne.datasets.testing.data_path(verbose=True), 'SSS')
fname_raw = op.join(data_path, 'test_move_anon_raw.fif')
raw = mne.io.read_raw_fif(fname_raw, allow_maxshield='yes').load_data()
raw.plot_psd()

###############################################################################
# Estimating continuous head position
# -----------------------------------
#
# First, let's extract the HPI coil amplitudes as a function of time:

chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)

###############################################################################
# Second, let's compute time-varying HPI coil locations from these:

chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)

###############################################################################
# Lastly, compute head positions from the coil locations:

head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)

###############################################################################
# Note that these can then be written to disk or read from disk with
# :func:`mne.chpi.write_head_pos` and :func:`mne.chpi.read_head_pos`,
# respectively.
#
# Visualizing continuous head position
# ------------------------------------
#
# We can plot as traces, which is especially useful for long recordings:

# sphinx_gallery_thumbnail_number = 2

mne.viz.plot_head_positions(head_pos, mode='traces')

###############################################################################
# Or we can visualize them as a continuous field (with the vectors pointing
# in the head-upward direction):

mne.viz.plot_head_positions(head_pos, mode='field')

###############################################################################
# These head positions can then be used with
# :func:`mne.preprocessing.maxwell_filter` to compensate for movement,
# or with :func:`mne.preprocessing.annotate_movement` to mark segments as
# bad that deviate too much from the average head position.
