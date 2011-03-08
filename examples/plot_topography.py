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
from mne.layouts import Layout
from mne.viz import plot_topo
from mne.datasets import sample
data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
evoked = fiff.read_evoked(fname, setno=0, baseline=(None, 0))

layout = Layout('Vectorview-all')

###############################################################################
# Show topography
plot_topo(evoked, layout)
title = 'MNE sample data (condition : %s)' % evoked.comment
pl.figtext(0.03, 0.93, title, color='w', fontsize=18)
pl.show()
