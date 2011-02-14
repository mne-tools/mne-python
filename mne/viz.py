"""Functions to plot M/EEG data e.g. topographies
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import pylab as pl


def plot_topo(data, layout):
    """Plot 2D topographies
    """
    ch_names = data['info']['ch_names']
    times = data['evoked']['times']
    epochs = data['evoked']['epochs']

    pl.rcParams['axes.edgecolor'] = 'w'
    pl.figure(facecolor='k')
    for name in layout.names:
        if name in ch_names:
            idx = ch_names.index(name)
            ax = pl.axes(layout.pos[idx], axisbg='k')
            ax.plot(times, epochs[idx,:], 'w')
            pl.xticks([], ())
            pl.yticks([], ())

    pl.show()
    pl.rcParams['axes.edgecolor'] = 'k'

