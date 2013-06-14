"""
=================================
Plot topographies for MEG sensors
=================================

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl

from mne import fiff
from mne.viz import plot_topo
from mne.datasets import sample
data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
evoked = fiff.read_evoked(fname, setno=0, baseline=(None, 0))

###############################################################################
# Show topography
title = 'MNE sample data (condition : %s)' % evoked.comment
plot_topo(evoked, title=title)
pl.show()
