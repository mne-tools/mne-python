"""
===============================
Visualize subject head movement
===============================

Show how subjects move as a function of time.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op

import mne

print(__doc__)

data_path = op.join(mne.datasets.testing.data_path(verbose=True), 'SSS')

pos = mne.chpi.read_head_pos(op.join(data_path, 'test_move_anon_raw.pos'))

###############################################################################
# Visualize the subject head movements as traces:

mne.viz.plot_head_positions(pos, mode='traces')

###############################################################################
# Or we can visualize them as a continuous field (with the vectors pointing
# in the head-upward direction):

mne.viz.plot_head_positions(pos, mode='field')
