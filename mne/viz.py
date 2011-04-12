"""Functions to plot M/EEG data e.g. topographies
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import pylab as pl
from .fiff.pick import channel_type


def plot_topo(evoked, layout):
    """Plot 2D topographies
    """
    ch_names = evoked.info['ch_names']
    times = evoked.times
    data = evoked.data

    pl.rcParams['axes.edgecolor'] = 'w'
    pl.figure(facecolor='k')
    for name in layout.names:
        if name in ch_names:
            idx = ch_names.index(name)
            ax = pl.axes(layout.pos[idx], axisbg='k')
            ax.plot(times, data[idx,:], 'w')
            pl.xticks([], ())
            pl.yticks([], ())

    pl.rcParams['axes.edgecolor'] = 'k'


def plot_evoked(evoked, picks=None, unit=True, show=True):
    """Plot evoked data

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    picks : None | array-like of int
        The indices of channels to plot. If None show all.
    unit : bool
        Scale plot with channel (SI) unit.
    show : bool
        Call pylab.show() as the end or not.
    """
    pl.clf()
    if picks is None:
        picks = range(evoked.info['nchan'])
    types = [channel_type(evoked.info, idx) for idx in picks]
    n_channel_types = 0
    channel_types = []
    for t in ['eeg', 'grad', 'mag']:
        if t in types:
            n_channel_types += 1
            channel_types.append(t)

    counter = 1
    times = 1e3 * evoked.times # time in miliseconds
    for t, scaling, name, ch_unit in zip(['eeg', 'grad', 'mag'],
                           [1e6, 1e13, 1e15],
                           ['EEG', 'Gradiometers', 'Magnetometers'],
                           ['uV', 'fT/cm', 'fT']):
        if unit is False:
            scaling = 1.0
            ch_unit = 'NA' # no unit
        idx = [picks[i] for i in range(len(picks)) if types[i] is t]
        if len(idx) > 0:
            pl.subplot(n_channel_types, 1, counter)
            pl.plot(times, scaling*evoked.data[idx,:].T)
            pl.title(name)
            pl.xlabel('time (ms)')
            counter += 1
            pl.ylabel('data (%s)' % ch_unit)

    pl.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
    if show:
        pl.show()
