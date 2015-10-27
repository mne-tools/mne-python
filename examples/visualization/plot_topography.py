"""
=================================
Plot topographies for MEG sensors
=================================

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

from mne import read_evokeds
from mne.viz import plot_evoked_topo
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

###############################################################################
# Show topography
title = 'MNE sample data (condition : %s)' % evoked.comment
plot_evoked_topo(evoked, title=title)
plt.show()
