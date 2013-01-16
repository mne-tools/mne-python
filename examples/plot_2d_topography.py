"""
==================================
View 2D topography of evoked data
==================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
evoked = fiff.Evoked(fname, setno=0, baseline=(None, 0), proj=True)

###############################################################################
# Show result

import pylab as pl
import matplotlib.delaunay as delaunay
from mne.layouts import read_layout

def plot_topo2d(evoked, time, ch_type='mag', layout=None,
                colorbar=True, show=True):
    if layout is None:
        layout = read_layout('Vectorview-%s' % ch_type)

    picks = fiff.pick_types(evoked.info, meg=ch_type,
                    exclude=evoked.info['bads'])  # Pick channels to view
    pos = np.array([layout.pos[layout.names.index(evoked.ch_names[k])] for k in picks])

    pl.figure(facecolor='w')
    pl.axis('off')
    pl.gca().set_aspect('equal')
    pl.xticks(())
    pl.yticks(())

    data = evoked.data[picks, np.where(evoked.times > time)[0][0]]

    triang = delaunay.Triangulation(pos[:,0], pos[:,1])
    interp = triang.linear_interpolator(data)
    resolution = 100
    x = np.linspace(pos[:, 0].min(), pos[:, 0].max(), resolution)
    y = np.linspace(pos[:, 1].min(), pos[:, 1].max(), resolution)
    xi, yi = np.meshgrid(x, y)

    data_im = interp[yi.min():yi.max():complex(0,yi.shape[0]),
                     xi.min():xi.max():complex(0,xi.shape[1])]
    data_im = np.ma.masked_array(data_im, data_im == np.nan)

    pl.imshow(data_im, origin='lower')

    if colorbar:
        pl.colorbar()

    if show:
        pl.show()

plot_topo2d(evoked, ch_type='mag', time=0.1, show=True)
# plot_topo2d(evoked, ch_type='grad', time=0.1, show=True)
