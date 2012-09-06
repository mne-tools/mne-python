"""Functions to plot M/EEG data e.g. topographies
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: Simplified BSD

from itertools import cycle
import copy
import numpy as np
from scipy import linalg
from mne.baseline import rescale

# XXX : don't import pylab here or you will break the doc

from .fiff.pick import channel_type, pick_types
from .fiff.proj import make_projector, activate_proj

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#473C8B', '#458B74',
          '#CD7F32', '#FF4040', '#ADFF2F', '#8E2323', '#FF1493']


def plot_topo(evoked, layout):
    """Plot 2D topographies
    """
    ch_names = evoked.info['ch_names']
    times = evoked.times
    data = evoked.data

    import pylab as pl
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


def plot_evoked(evoked, picks=None, unit=True, show=True,
                ylim=None, proj=False, xlim='tight'):
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
    ylim : dict
        ylim for plots. e.g. ylim = dict(eeg=[-200e-6, 200e6])
        Valid keys are eeg, mag, grad
    xlim : 'tight' | tuple | None
        xlim for plots.
    proj : bool
        If true SSP projections are applied before display.
    """
    import pylab as pl
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
            D = scaling * evoked.data[idx, :]
            if proj:
                projs = activate_proj(evoked.info['projs'])
                this_ch_names = [evoked.ch_names[k] for k in idx]
                P, ncomp, _ = make_projector(projs, this_ch_names)
                D = np.dot(P, D)

            pl.subplot(n_channel_types, 1, counter)
            pl.plot(times, D.T)
            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                pl.xlim(xlim)
            if ylim is not None and t in ylim:
                pl.ylim(ylim[t])
            pl.title(name + ' (%d channels)' % len(D))
            pl.xlabel('time (ms)')
            counter += 1
            pl.ylabel('data (%s)' % ch_unit)

    pl.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
    try:
        pl.tight_layout()
    except:
        pass
    if show:
        pl.show()


def _prepare_tfr_plot(epochs, tfr, ch_name, freq, baseline, mode, decim):
    """Helper function: plot time frequency representation for single channel

    Parameters
    ----------
    epochs : epochsnobject
        epochs object from mne.Epochs
    tfr : 3D-array
        results from induced_power, either power or phase_lock
    ch_name : string
        channel name
    freq : array-like
        frequencies of interest as passed to induced_power
    baseline: tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode: 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    decim : integer
        steps for leaving out timeslices

    Returns
    =======
    tfr_ch : 2D array
        Time frequency representation of power for the selected channel
    extent : tuple
        Plotting parameters -- tmin, tmax, lower and upper frequency
    """
    if ch_name not in epochs.info['ch_names']:
        raise ValueError('"%s" is not a valid channel name' % ch_name)
    # Milli seconds
    times = epochs.times
    times = 1e3 * times
    tmin = 1e3 * epochs.tmin
    tmax = 1e3 * epochs.tmax
    ch_idx = epochs.info["ch_names"].index(ch_name)
    # channel selcetion
    tfr_ch = tfr[ch_idx:(ch_idx + 1)].copy()
    # baseline corrections
    tfr_ch = rescale(tfr_ch, times, baseline, mode)
    # plot_data = 20 * np.log10(tfr[0]) if dB else tfr_ch[0]
    extent = (tmin, tmax, freq[0], freq[-1])

    return tfr_ch[0], extent


def _plot_topo_imshow(epochs, tfr, freq, layout, baseline, mode, decim):
    """ Helper function: plot time frequency representations on sensor layout

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs used to generate the power
    tfr : 3D-array
        return value from time_frequency.induced_power, wither power
        or phase_lock
    freq : array-like
        frequencies of interest as passed to induced_power
    layout: instance of Layout
        System specific sensor positions
    baseline: tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode: 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    decim : integer
        steps for leaving out timeslices

    Returns
    -------
    fig : matplotlib figure
        images of induced power at sensor locartions
    """

    import pylab as pl
    ch_names = epochs.info['ch_names']
    pl.rcParams['axes.edgecolor'] = 'w'
    fig = pl.figure(facecolor='k')
    for idx, name in enumerate(layout.names):
        if name in ch_names:
            ax = pl.axes(layout.pos[idx], axisbg='k')
            # tfr, ch_name, epochs, is_power, freq,
            tfr_data, extent = _prepare_tfr_plot(epochs, tfr, name, freq,
                                                 baseline, mode, decim)
            ax.imshow(tfr_data, extent=extent, aspect="auto", origin="lower")
            pl.xticks([], ())
            pl.yticks([], ())
    pl.rcParams['axes.edgecolor'] = 'k'

    return fig


def plot_topo_power(epochs, power, freq, layout, baseline=None, mode='mean',
                    decim=1):
    """Plot induced power on sensor layout

    Parameters
    ----------

    epochs : instance of Epochs
        The epochs used to generate the power
    power : 3D-array
        first return value from mne.time_frequency.induced_power
    freq : array-like
        frequencies of interest as passed to induced_power
    layout: instance of Layout
        System specific sensor positions
    baseline: tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode: 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    decim : integer
        steps for leaving out timeslices

    Returns
    -------
    fig : matplotlib figure
        images of induced power at sensor locartions

    """
    if not baseline:
        baseline = epochs.baseline

    # name = 'Induced Power'
    ret = _plot_topo_imshow(epochs, power, freq, layout, decim=decim,
                            baseline=baseline, mode=mode)

    return ret


def plot_topo_phase_lock(epochs, phase_lock, freq, layout, baseline=None,
                         mode='mean', decim=1):
    """Plot phase locking values on sensor layout

    Parameters
    ----------
    phase_lock : 3D-array
        Phase locking value, second return value from
        mne.time_frequency.induced_power
    epochs : instance of Epochs
        The epochs used to generate the phase locking value
    freq : array-like
        frequencies of interest as passed to induced_power
    baseline: tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode: 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviatio of power during baseline after substracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    decim : integer
        steps for leaving out timeslices

    Returns
    -------
    fig : matplotlib figure
        Phase lock images at sensor locartions

    """
    if not baseline:
        baseline = epochs.baseline

    # name = 'Phase Locking Value'
    ret = _plot_topo_imshow(epochs, phase_lock, freq, layout, decim=decim,
                            baseline=baseline, mode=mode)

    return ret


def plot_sparse_source_estimates(src, stcs, colors=None, linewidth=2,
                                 fontsize=18, bgcolor=(.05, 0, .1), opacity=0.2,
                                 brain_color=(0.7, ) * 3, show=True,
                                 high_resolution=False, fig_name=None,
                                 fig_number=None, labels=None,
                                 modes=['cone', 'sphere'],
                                 scale_factors=[1, 0.6],
                                 **kwargs):
    """Plot source estimates obtained with sparse solver

    Active dipoles are represented in a "Glass" brain.
    If the same source is active in multiple source estimates it is
    displayed with a sphere otherwise with a cone in 3D.

    Parameters
    ----------
    src: dict
        The source space
    stcs: instance of SourceEstimate or list of instances of SourceEstimate
        The source estimates (up to 3)
    colors: list
        List of colors
    linewidth: int
        Line width in 2D plot
    fontsize: int
        Font size
    bgcolor: tuple of length 3
        Background color in 3D
    opacity: float in [0, 1]
        Opacity of brain mesh
    brain_color: tuple of length 3
        Brain color
    show: bool
        Show figures if True
    fig_name:
        Mayavi figure name
    fig_number:
        Pylab figure number
    labels: ndarray or list of ndarrays
        Labels to show sources in clusters. Sources with the same
        label and the waveforms within each cluster are presented in
        the same color. labels should be a list of ndarrays when
        stcs is a list ie. one label for each stc.
    kwargs: kwargs
        kwargs pass to mlab.triangular_mesh
    """
    if not isinstance(stcs, list):
        stcs = [stcs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    if colors is None:
        colors = COLORS

    linestyles = ['-', '--', ':']

    # Show 3D
    lh_points = src[0]['rr']
    rh_points = src[1]['rr']
    points = np.r_[lh_points, rh_points]

    lh_normals = src[0]['nn']
    rh_normals = src[1]['nn']
    normals = np.r_[lh_normals, rh_normals]

    if high_resolution:
        use_lh_faces = src[0]['tris']
        use_rh_faces = src[1]['tris']
    else:
        use_lh_faces = src[0]['use_tris']
        use_rh_faces = src[1]['use_tris']

    use_faces = np.r_[use_lh_faces, lh_points.shape[0] + use_rh_faces]

    points *= 170

    vertnos = [np.r_[stc.lh_vertno, lh_points.shape[0] + stc.rh_vertno]
               for stc in stcs]
    unique_vertnos = np.unique(np.concatenate(vertnos).ravel())

    try:
        from mayavi import mlab
    except ImportError:
        from enthought.mayavi import mlab

    from matplotlib.colors import ColorConverter
    color_converter = ColorConverter()

    f = mlab.figure(figure=fig_name, bgcolor=bgcolor, size=(600, 600))
    mlab.clf()
    f.scene.disable_render = True
    surface = mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
                                   use_faces, color=brain_color,
                                   opacity=opacity, **kwargs)

    import pylab as pl
    # Show time courses
    pl.figure(fig_number)
    pl.clf()

    colors = cycle(colors)

    print "Total number of active sources: %d" % len(unique_vertnos)

    if labels is not None:
        colors = [colors.next() for _ in
                  range(np.unique(np.concatenate(labels).ravel()).size)]

    for v in unique_vertnos:
        # get indices of stcs it belongs to
        ind = [k for k, vertno in enumerate(vertnos) if v in vertno]
        is_common = len(ind) > 1

        if labels is None:
            c = colors.next()
        else:
            # if vertex is in different stcs than take label from first one
            c = colors[labels[ind[0]][vertnos[ind[0]] == v]]

        mode = modes[1] if is_common else modes[0]
        scale_factor = scale_factors[1] if is_common else scale_factors[0]
        x, y, z = points[v]
        nx, ny, nz = normals[v]
        mlab.quiver3d(x, y, z, nx, ny, nz, color=color_converter.to_rgb(c),
                      mode=mode, scale_factor=scale_factor)

        for k in ind:
            vertno = vertnos[k]
            mask = (vertno == v)
            assert np.sum(mask) == 1
            linestyle = linestyles[k]
            pl.plot(1e3 * stc.times, 1e9 * stcs[k].data[mask].ravel(), c=c,
                    linewidth=linewidth, linestyle=linestyle)

    pl.xlabel('Time (ms)', fontsize=18)
    pl.ylabel('Source amplitude (nAm)', fontsize=18)

    if fig_name is not None:
        pl.title(fig_name)

    if show:
        pl.show()

    surface.actor.property.backface_culling = True
    surface.actor.property.shading = True

    return surface


def plot_cov(cov, info, exclude=[], colorbar=True, proj=False, show_svd=True,
             show=True):
    """Plot Covariance data

    Parameters
    ----------
    cov : instance of Covariance
        The covariance matrix
    info: dict
        Measurement info
    exclude : list of string
        List of channels to exclude. If empty do not exclude any channel.
    colorbar : bool
        Show colorbar or not
    proj : bool
        Apply projections or not
    show : bool
        Call pylab.show() as the end or not.
    show_svd : bool
        Plot also singular values of the noise covariance for each sensor type.
        We show square roots ie. standard deviations.
    """
    ch_names = [n for n in cov.ch_names if not n in exclude]
    ch_idx = [cov.ch_names.index(n) for n in ch_names]
    info_ch_names = info['ch_names']
    sel_eeg = pick_types(info, meg=False, eeg=True, exclude=exclude)
    sel_mag = pick_types(info, meg='mag', eeg=False, exclude=exclude)
    sel_grad = pick_types(info, meg='grad', eeg=False, exclude=exclude)
    idx_eeg = [ch_names.index(info_ch_names[c])
               for c in sel_eeg if info_ch_names[c] in ch_names]
    idx_mag = [ch_names.index(info_ch_names[c])
               for c in sel_mag if info_ch_names[c] in ch_names]
    idx_grad = [ch_names.index(info_ch_names[c])
                for c in sel_grad if info_ch_names[c] in ch_names]

    idx_names = [(idx_eeg, 'EEG covariance', 'uV', 1e6),
                 (idx_grad, 'Gradiometers', 'fT/cm', 1e13),
                 (idx_mag, 'Magnetometers', 'fT', 1e15)]
    idx_names = [(idx, name, unit, scaling)
                 for idx, name, unit, scaling in idx_names if len(idx) > 0]

    C = cov.data[ch_idx][:, ch_idx]

    if proj:
        projs = copy.deepcopy(info['projs'])

        #   Activate the projection items
        for p in projs:
            p['active'] = True

        P, ncomp, _ = make_projector(projs, ch_names)
        if ncomp > 0:
            print '    Created an SSP operator (subspace dimension = %d)' % \
                                                                        ncomp
            C = np.dot(P, np.dot(C, P.T))
        else:
            print '    The projection vectors do not apply to these channels.'

    import pylab as pl

    pl.figure(figsize=(2.5 * len(idx_names), 2.7))
    for k, (idx, name, _, _) in enumerate(idx_names):
        pl.subplot(1, len(idx_names), k + 1)
        pl.imshow(C[idx][:, idx], interpolation="nearest")
        pl.title(name)
    pl.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
    try:
        pl.tight_layout()  # XXX : recent pylab feature
    except:
        pass

    if show_svd:
        pl.figure()
        for k, (idx, name, unit, scaling) in enumerate(idx_names):
            _, s, _ = linalg.svd(C[idx][:, idx])
            pl.subplot(1, len(idx_names), k + 1)
            pl.ylabel('Noise std (%s)' % unit)
            pl.xlabel('Eigenvalue index')
            pl.semilogy(np.sqrt(s) * scaling)
            pl.title(name)
        try:
            pl.tight_layout()  # XXX : recent pylab feature
        except:
            pass

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


def plot_ica_panel(sources, start=None, stop=None, n_components=None,
                   source_idx=None, ncol=3, nrow=10):
    """Create panel plots of ICA sources

    Parameters
    ----------
    sources : ndarray
        sources as drawn from ica.get_sources
    start : int
        x-axis start index. If None from the beginning.
    stop : int
        x-axis stop index. If None to the end.
    n_components : int
        number of components fitted
    source_idx : array-like
        indices for subsetting the sources
    ncol : int
        number of panel-columns
    nrow : int
        number of panel-rows

    Returns
    -------
    fig : instance of pyplot.Figure
    """
    import pylab as pl

    if n_components is None:
        n_components = len(sources)

    hangover = n_components % ncol
    nplots = nrow * ncol

    if source_idx is not None:
        sources = sources[source_idx]
    if source_idx is None:
        source_idx = np.arange(n_components)
    elif source_idx.shape > 30:
        print ('More sources selected than rows and cols specified.'
               'Showing the first %i sources.' % nplots)
        source_idx = np.arange(nplots)

    sources = sources[:, start:stop]
    ylims = sources.min(), sources.max()
    fig, panel_axes = pl.subplots(nrow, ncol, sharey=True, figsize=(9, 10))
    fig.suptitle('MEG signal decomposition'
                 ' -- %i components.' % n_components, size=16)

    pl.subplots_adjust(wspace=0.05, hspace=0.05)

    iter_plots = ((row, col) for row in range(nrow) for col in range(ncol))

    for idx, (row, col) in enumerate(iter_plots):
        xs = panel_axes[row, col]
        xs.grid(linestyle='-', color='gray', linewidth=.25)
        if idx < n_components:
            component = '[%i]' % idx
            xs.plot(sources[idx], linewidth=0.5, color='red')
            xs.text(0.05, .95, component,
                    transform=panel_axes[row, col].transAxes,
                    verticalalignment='top')
            pl.ylim(ylims)
        else:
            # Make extra subplots invisible
            pl.setp(xs, visible=False)

        xtl = xs.get_xticklabels()
        ytl = xs.get_yticklabels()
        if row < nrow - 2 or (row < nrow - 1 and
                              (hangover == 0 or col <= hangover - 1)):
            pl.setp(xtl, visible=False)
        if (col > 0) or (row % 2 == 1):
            pl.setp(ytl, visible=False)
        if (col == ncol - 1) and (row % 2 == 1):
            xs.yaxis.tick_right()

        pl.setp(xtl, rotation=90.)

    return fig
