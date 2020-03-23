"""
===========================
Annotate muscle artifacts
===========================

Muscle contractions produce high frequency activity that can mask brain signal
of interest. Muscle artifacts can be produced when clenching the jaw,
swallowing, or twitching a head muscle. Muscle artifacts are most notable
in the range of 110-140 Hz.

This example uses :func:`~mne.preprocessing.annotate_muscle_zscore` to annotate
segments where muscle activity is likely present. This is done by band-pass
filtering the data in the 110-140 Hz range. Then, the envelope is taken using
the hilbert analytical signal to only consider the absolute amplitude and not
the phase of the high frequency signal. The envelope is z-scored and summed
across channels and divedid by the square root of the channel numer. To remove
noisy transient peaks (muscle artifacts last several hundred milliseconds),
the channel averaged z-scored is low-pass filtered to 4 Hz. Segments above a
set threshold are annotated as ``BAD_muscle``. In addition, ``min_length_good``
parameter determines the cutoff for whether short spans of "good data" in
between muscle artifacts are included in the surrounding "BAD" annotation.

"""
# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)

import os.path as op
import matplotlib.pyplot as plt
import numpy as np
from mne import pick_types
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing import annotate_muscle_zscore


# Load data
data_path = bst_auditory.data_path()
data_path_MEG = op.join(data_path, 'MEG')
subject = 'bst_auditory'
raw_fname = op.join(data_path_MEG, subject, 'S01_AEF_20131218_01.ds')

raw = read_raw_ctf(raw_fname, preload=False)

raw.crop(130, 160).load_data()  # just use a fraction of data for speed here
raw.resample(300, npad="auto")

# The inputted raw data has to be from a single channel type as there might be
# different channel type magnitudes.
# If the MEG has axial gradiometers and magnetometers, select magnetometers as
# they are more sensitive to muscle activity
picks = pick_types(raw.info, meg=True, ref_meg=False)

###############################################################################
raw.notch_filter([50, 100])
"""
.. note::
    If line noise is present, you should perform notch-filtering *before*
    detecting muscle artifacts. See :ref:`tut-section-line-noise` for an
    example.
"""

# The threshold is data dependent, check the optimal threshold by plotting
# ``threshold_muscle``
threshold_muscle = 4  # z-score
out = annotate_muscle_zscore(raw, picks=picks, threshold=threshold_muscle,
                             min_length_good=0.2, filter_freq=[90, 120])
annot_muscle, scores_muscle = out

###############################################################################
# Plot movement z-scores across recording
# --------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(raw.times, scores_muscle)
ax.axhline(y=threshold_muscle, color='r')
ax.set(xlabel='time, s.', ylabel='zscore', title='Muscle activity')
###############################################################################
# Plot raw with annotated muscles
# --------------------------------------------------------------------------
order = np.arange(220, 240)
raw.set_annotations(annot_muscle)
raw.plot(duration=20, order=order)
