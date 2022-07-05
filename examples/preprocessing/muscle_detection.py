# -*- coding: utf-8 -*-
"""
.. _ex-muscle-artifacts:

=========================
Annotate muscle artifacts
=========================

Muscle contractions produce high frequency activity that can mask brain signal
of interest. Muscle artifacts can be produced when clenching the jaw,
swallowing, or twitching a cranial muscle. Muscle artifacts are most
noticeable in the range of 110-140 Hz.

This example uses :func:`~mne.preprocessing.annotate_muscle_zscore` to annotate
segments where muscle activity is likely present. This is done by band-pass
filtering the data in the 110-140 Hz range. Then, the envelope is taken using
the hilbert analytical signal to only consider the absolute amplitude and not
the phase of the high frequency signal. The envelope is z-scored and summed
across channels and divided by the square root of the number of channels.
Because muscle artifacts last several hundred milliseconds, a low-pass filter
is applied on the averaged z-scores at 4 Hz, to remove transient peaks.
Segments above a set threshold are annotated as ``BAD_muscle``. In addition,
the ``min_length_good`` parameter determines the cutoff for whether short
spans of "good data" in between muscle artifacts are included in the
surrounding "BAD" annotation.

"""
# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import numpy as np
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing import annotate_muscle_zscore


# Load data
data_path = bst_auditory.data_path()
raw_fname = data_path / 'MEG' / 'bst_auditory' / 'S01_AEF_20131218_01.ds'

raw = read_raw_ctf(raw_fname, preload=False)

raw.crop(130, 160).load_data()  # just use a fraction of data for speed here
raw.resample(300, npad="auto")

# %%
# Notch filter the data:
#
# .. note::
#     If line noise is present, you should perform notch-filtering *before*
#     detecting muscle artifacts. See :ref:`tut-section-line-noise` for an
#     example.

raw.notch_filter([60, 120])

# %%

# The threshold is data dependent, check the optimal threshold by plotting
# ``scores_muscle``.
threshold_muscle = 5  # z-score
# Choose one channel type, if there are axial gradiometers and magnetometers,
# select magnetometers as they are more sensitive to muscle activity.
annot_muscle, scores_muscle = annotate_muscle_zscore(
    raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[110, 140])

# %%
# Plot muscle z-scores across recording
# --------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(raw.times, scores_muscle)
ax.axhline(y=threshold_muscle, color='r')
ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
# %%
# View the annotations
# --------------------------------------------------------------------------
order = np.arange(144, 164)
raw.set_annotations(annot_muscle)
raw.plot(start=5, duration=20, order=order)
