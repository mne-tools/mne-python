"""
=================================
Plot topographies for MEG sensors
=================================

"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import pylab as pl

from mne import fiff
from mne.layouts import Layout
from mne.viz import plot_topo

fname = os.environ['MNE_SAMPLE_DATASET_PATH']
fname += '/MEG/sample/sample_audvis-ave.fif'

# Reading
data = fiff.read_evoked(fname, setno=0, baseline=(None, 0))

layout = Layout('Vectorview-all')

###############################################################################
# Show topography
plot_topo(data, layout)
title = 'MNE sample data (condition : %s)' % data['evoked']['comment']
pl.figtext(0.03, 0.93, title, color='w', fontsize=18)
pl.show()