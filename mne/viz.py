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
        idx = [picks[i] for i in range(len(picks)) if types[i] == t]
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


def plot_source_estimate(src, stc, n_smooth=200, cmap='jet'):
    """Plot source estimates
    """
    from enthought.tvtk.api import tvtk
    from enthought.traits.api import HasTraits, Range, Instance, \
                                     on_trait_change
    from enthought.traits.ui.api import View, Item, Group

    from enthought.mayavi.core.api import PipelineBase
    from enthought.mayavi.core.ui.api import MayaviScene, SceneEditor, \
                    MlabSceneModel

    class SurfaceViewer(HasTraits):
        n_times = Range(0, 100, 0, )

        scene = Instance(MlabSceneModel, ())
        surf = Instance(PipelineBase)
        text = Instance(PipelineBase)

        def __init__(self, src, data, times, n_smooth=20, cmap='jet'):
            super(SurfaceViewer, self).__init__()
            self.src = src
            self.data = data
            self.times = times
            self.n_smooth = n_smooth
            self.cmap = cmap

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

            self.points = points[use_idx]
            self.faces = np.searchsorted(use_idx, faces)

        # When the scene is activated, or when the parameters are changed, we
        # update the plot.
        @on_trait_change('n_times,scene.activated')
        def update_plot(self):
            idx = int(self.n_times * len(self.times) / 100)
            t = self.times[idx]
            d = self.data[:, idx].astype(np.float)  # 8bits for mayavi
            points = self.points
            faces = self.faces
            info_time = "%d ms" % (1e3 * t)
            if self.surf is None:
                surface_mesh = self.scene.mlab.pipeline.triangular_mesh_source(
                                    points[:, 0], points[:, 1], points[:, 2],
                                    faces, scalars=d)
                smooth_ = tvtk.SmoothPolyDataFilter(
                                    number_of_iterations=self.n_smooth,
                                    relaxation_factor=0.18,
                                    feature_angle=70,
                                    feature_edge_smoothing=False,
                                    boundary_smoothing=False,
                                    convergence=0.)
                surface_mesh_smooth = self.scene.mlab.pipeline.user_defined(
                                                surface_mesh, filter=smooth_)
                self.surf = self.scene.mlab.pipeline.surface(
                                    surface_mesh_smooth, colormap=self.cmap)

                self.scene.mlab.colorbar()
                self.text = self.scene.mlab.text(0.7, 0.9, info_time,
                                                 width=0.2)
                self.scene.background = (.05, 0, .1)
            else:
                self.surf.mlab_source.set(scalars=d)
                self.text.set(text=info_time)

        # The layout of the dialog created
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                         height=800, width=800, show_label=False),
                    Group('_', 'n_times',),
                    resizable=True,)

    viewer = SurfaceViewer(src, stc.data, stc.times, n_smooth=200)
    viewer.configure_traits()
    return viewer
