"""
==============================================
Reading a source space from a forward operator
==============================================

This example visualizes a source space mesh used by a forward operator.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


import os.path as op

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')

patch_stats = True  # include high resolution source space
src = mne.read_source_spaces(fname, patch_stats=patch_stats)

# Plot the 3D source space (high sampling)
src.plot(subjects_dir=subjects_dir)
