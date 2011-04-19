"""Functions to plot M/EEG data e.g. topographies
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import numpy as np
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
            ax.plot(times, data[idx, :], 'w')
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
    times = 1e3 * evoked.times  # time in miliseconds
    for t, scaling, name, ch_unit in zip(['eeg', 'grad', 'mag'],
                           [1e6, 1e13, 1e15],
                           ['EEG', 'Gradiometers', 'Magnetometers'],
                           ['uV', 'fT/cm', 'fT']):
        if unit is False:
            scaling = 1.0
            ch_unit = 'NA'  # no unit
        idx = [picks[i] for i in range(len(picks)) if types[i] is t]
        if len(idx) > 0:
            pl.subplot(n_channel_types, 1, counter)
            pl.plot(times, scaling * evoked.data[idx, :].T)
            pl.title(name)
            pl.xlabel('time (ms)')
            counter += 1
            pl.ylabel('data (%s)' % ch_unit)

    pl.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
    if show:
        pl.show()


def plot_sources(src, data, text=None, n_smooth=200, colorbar=True,
                 cmap="jet"):
    """Source space data
    """
    from enthought.mayavi import mlab
    from enthought.tvtk.api import tvtk
    lh_points = src[0]['rr']
    rh_points = src[1]['rr']
    # lh_faces = src[0]['tris']
    # rh_faces = src[1]['tris']
    lh_faces = src[0]['use_tris']
    rh_faces = src[1]['use_tris']
    points = np.r_[lh_points, rh_points]
    points *= 200
    faces = np.r_[lh_faces, lh_points.shape[0] + rh_faces]

    lh_idx = np.where(src[0]['inuse'])[0]
    rh_idx = np.where(src[1]['inuse'])[0]
    use_idx = np.r_[lh_idx, lh_points.shape[0] + rh_idx]

    points = points[use_idx]
    faces = np.searchsorted(use_idx, faces)

    mlab.test_quiver3d()
    mlab.clf()
    mlab.options.offscreen = True
    f = mlab.figure(512, bgcolor=(.05, 0, .1), size=(800, 800))
    mlab.clf()
    f.scene.disable_render = True
    surface_mesh = mlab.pipeline.triangular_mesh_source(points[:, 0],
                                    points[:, 1], points[:, 2], faces,
                                    scalars=data)
    smooth_ = tvtk.SmoothPolyDataFilter(number_of_iterations=n_smooth,
                                        relaxation_factor=0.18,
                                        feature_angle=70,
                                        feature_edge_smoothing=False,
                                        boundary_smoothing=False,
                                        convergence=0.)
    surface_mesh_smooth = mlab.pipeline.user_defined(surface_mesh,
                                                     filter=smooth_)
    mlab.pipeline.surface(surface_mesh_smooth, colormap=cmap)
    bar = mlab.scalarbar()
    if text is not None:
        mlab.text(0.7, 0.9, text, width=0.2)
    if not colorbar:
        bar.visible = False
