"""Functions to plot M/EEG data e.g. topographies
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD
import os
import warnings
from itertools import cycle
from functools import partial
import copy
import inspect

import numpy as np
from scipy import linalg
from scipy import ndimage

import logging
logger = logging.getLogger('mne')
from warnings import warn


# XXX : don't import pylab here or you will break the doc
from .fixes import tril_indices
from .baseline import rescale
from .utils import deprecated, get_subjects_dir
from .fiff.pick import channel_type, pick_types
from .fiff.proj import make_projector, activate_proj
from . import verbose

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#473C8B', '#458B74',
          '#CD7F32', '#FF4040', '#ADFF2F', '#8E2323', '#FF1493']


def _clean_names(names):
    """ Remove white-space on topo matching

    Over the years, Neuromag systems employed inconsistent handling of
    white-space in layout names. This function handles different naming
    conventions and hence should be used in each topography-plot to
    warrant compatibility across systems.

    Usage
    -----
    Wrap this function around channel and layout names:
    ch_names = _clean_names(epochs.ch_names)

    for n in _clean_names(layout.names):
        if n in ch_names:
            # prepare plot

    """
    return [n.replace(' ', '') if ' ' in n else n for n in names]


def tight_layout(pad=1.2, h_pad=None, w_pad=None):
    """ Adjust subplot parameters to give specified padding.

    Note. For plotting please use this function instead of pl.tight_layout

    Parameters:
    -----------
    pad : float
        padding between the figure edge and the edges of subplots, as a
        fraction of the font-size.
    h_pad, w_pad : float
        padding (height/width) between edges of adjacent subplots.
        Defaults to `pad_inches`.
    """
    try:
        import pylab as pl
        pl.tight_layout(pad=1.2, h_pad=None, w_pad=None)
    except:
        msg = ('Matplotlib function \'tight_layout\'%s.'
               ' Skipping subpplot adjusment.')
        if not hasattr(pl, 'tight_layout'):
            case = ' is not available'
        else:
            case = ' seems corrupted'
        warn(msg % case)


def _plot_topo(info, times, show_func, layout, decim, vmin, vmax, colorbar,
               cmap, layout_scale, title=None, x_label=None, y_label=None):
    """Helper function to plot on sensor layout"""
    import pylab as pl
    orig_facecolor = pl.rcParams['axes.facecolor']
    orig_edgecolor = pl.rcParams['axes.edgecolor']
    try:
        if cmap is None:
            cmap = pl.cm.jet
        ch_names = _clean_names(info['ch_names'])
        pl.rcParams['axes.facecolor'] = 'k'
        fig = pl.figure(facecolor='k')
        pos = layout.pos.copy()
        tmin, tmax = times[0], times[-1]
        if colorbar:
            pos[:, :2] *= layout_scale
            pl.rcParams['axes.edgecolor'] = 'k'
            sm = pl.cm.ScalarMappable(cmap=cmap,
                                      norm=pl.normalize(vmin=vmin, vmax=vmax))
            sm.set_array(np.linspace(vmin, vmax))
            ax = pl.axes([0.015, 0.025, 1.05, .8], axisbg='k')
            cb = fig.colorbar(sm, ax=ax)
            cb_yticks = pl.getp(cb.ax.axes, 'yticklabels')
            pl.setp(cb_yticks, color='w')
        pl.rcParams['axes.edgecolor'] = 'w'
        for idx, name in enumerate(_clean_names(layout.names)):
            if name in ch_names:
                ax = pl.axes(pos[idx], axisbg='k')
                ch_idx = ch_names.index(name)
                # hack to inlcude channel idx and name, to use in callback
                ax.__dict__['_mne_ch_name'] = name
                ax.__dict__['_mne_ch_idx'] = ch_idx
                show_func(ax, ch_idx, tmin, tmax, vmin, vmax)
                pl.xticks([], ())
                pl.yticks([], ())

        # register callback
        callback = partial(_plot_topo_onpick, show_func=show_func, tmin=tmin,
                           tmax=tmax, vmin=vmin, vmax=vmax, colorbar=colorbar,
                           title=title, x_label=x_label, y_label=y_label)

        fig.canvas.mpl_connect('pick_event', callback)

        if title is not None:
            pl.figtext(0.03, 0.9, title, color='w', fontsize=19)

    finally:
        # Revert global pylab config
        pl.rcParams['axes.facecolor'] = orig_facecolor
        pl.rcParams['axes.edgecolor'] = orig_edgecolor

    return fig


def _plot_topo_onpick(event, show_func=None, tmin=None, tmax=None,
                      vmin=None, vmax=None, colorbar=False, title=None,
                      x_label=None, y_label=None):
    """Onpick callback that shows a single channel in a new figure"""

    # make sure that the swipe gesture in OS-X doesn't open many figures
    if event.mouseevent.inaxes is None or event.mouseevent.button != 1:
        return

    artist = event.artist
    try:
        import pylab as pl
        ch_idx = artist.axes._mne_ch_idx
        fig, ax = pl.subplots(1)
        ax.set_axis_bgcolor('k')
        show_func(pl, ch_idx, tmin, tmax, vmin, vmax)
        if colorbar:
            pl.colorbar()
        if title is not None:
            pl.title(title + ' ' + artist.axes._mne_ch_name)
        else:
            pl.title(artist.axes._mne_ch_name)
        if x_label is not None:
            pl.xlabel(x_label)
        if y_label is not None:
            pl.ylabel(y_label)
    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers, so we print
        # it here to know what went wrong
        print err
        raise err


def _imshow_tfr(ax, ch_idx, tmin, tmax, vmin, vmax, tfr=None, freq=None):
    """ Aux function to show time-freq map on topo """
    extent = (tmin, tmax, freq[0], freq[-1])
    ax.imshow(tfr[ch_idx], extent=extent, aspect="auto", origin="lower",
              vmin=vmin, vmax=vmax, picker=True)


def _plot_timeseries(ax, ch_idx, tmin, tmax, vmin, vmax, data, color, times):
    """ Aux function to show time series on topo """
    # use large tol for picker so we can click anywhere in the axes
    for data_, color in zip(data, color):
        ax.plot(times, data_[ch_idx], color, picker=1e9)


def plot_topo(evoked, layout, layout_scale=0.945, color=None, title=None):
    """Plot 2D topography of evoked responses.

    Clicking on the plot of an individual sensor opens a new figure showing
    the evoked response for the selected sensor.

    Parameters
    ----------
    evoked : list of Evoked | Evoked
        The evoked response to plot.
    layout : instance of Layout
        System specific sensor positions
    layout_scale: float
        Scaling factor for adjusting the relative size of the layout
        on the canvas
    color : list of color objects | color object | None
        Everything matplotlib accepts to specify colors. If not list-like,
        the color specified will be repeated. If None, colors are
        automatically drawn.
    title : str
        Title of the figure.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Images of evoked responses at sensor locations
    """
    if not type(evoked) in (tuple, list):
        evoked = [evoked]

    if type(color) in (tuple, list):
        if len(color) != len(evoked):
            raise ValueError('Lists of evoked objects and colors'
                             ' must have the same length')
    elif color is None:
        colors = ['w'] + COLORS
        stop = (slice(len(evoked)) if len(evoked) < len(colors)
                else slice(len(colors)))
        color = cycle(colors[stop])
        if len(evoked) > len(colors):
            warnings.warn('More evoked objects then colors available.'
                          'You should pass a list of unique colors.')
    else:
        color = cycle([color])

    times = evoked[0].times
    if not all([(e.times == times).all() for e in evoked]):
        raise ValueError('All evoked.times must be the same')

    ch_names = evoked[0].ch_names
    if not all([e.ch_names == ch_names for e in evoked]):
        raise ValueError('All evoked.picks must be the same')

    plot_fun = partial(_plot_timeseries, data=[e.data for e in evoked],
                       color=color, times=times)

    info = evoked[0].info
    fig = _plot_topo(info, times, plot_fun, layout,
                     decim=1, colorbar=False, vmin=0, vmax=0,
                     cmap=None, layout_scale=layout_scale, title=title,
                     x_label='Time (s)')
    return fig


def plot_topo_tfr(epochs, tfr, freq, layout, colorbar=True, vmin=None,
                  vmax=None, cmap=None, layout_scale=0.945, title=None):
    """Plot time-frequency data on sensor layout

    Clicking on the time-frequency map of an individual sensor opens a
    new figure showing the time-frequency map of the selected sensor.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs used to generate the power
    tfr : 3D-array shape=(n_sensors, n_freqs, n_times)
        The time-frequency data. Must have the same channels as Epochs.
    freq : array-like
        Frequencies of interest as passed to induced_power
    layout : instance of Layout
        System specific sensor positions
    colorbar : bool
        If true, colorbar will be added to the plot
    vmin : float
        Minimum value mapped to lowermost color
    vmax : float
        Minimum value mapped to upppermost color
    cmap : instance of matplotlib.pylab.colormap
        Colors to be mapped to the values
    layout_scale : float
        Scaling factor for adjusting the relative size of the layout
        on the canvas
    title : str
        Title of the figure.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Images of time-frequency data at sensor locations
    """

    if vmin is None:
        vmin = tfr.min()
    if vmax is None:
        vmax = tfr.max()

    tfr_imshow = partial(_imshow_tfr, tfr=tfr.copy(), freq=freq)

    fig = _plot_topo(epochs.info, epochs.times, tfr_imshow, layout,
                     decim=1, colorbar=colorbar, vmin=vmin, vmax=vmax,
                     cmap=cmap, layout_scale=layout_scale, title=title,
                     x_label='Time (s)', y_label='Frequency (Hz)')
    return fig


def plot_topo_power(epochs, power, freq, layout, baseline=None, mode='mean',
                    decim=1, colorbar=True, vmin=None, vmax=None, cmap=None,
                    layout_scale=0.945, dB=True, title=None):
    """Plot induced power on sensor layout

    Clicking on the induced power map of an individual sensor opens a
    new figure showing the induced power map of the selected sensor.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs used to generate the power
    power : 3D-array
        First return value from mne.time_frequency.induced_power
    freq : array-like
        Frequencies of interest as passed to induced_power
    layout : instance of Layout
        System specific sensor positions
    baseline : tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode : 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or z-score (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    If None, baseline no correction will be performed.
    decim : integer
        Increment for selecting each nth time slice
    colorbar : bool
        If true, colorbar will be added to the plot
    vmin : float
        Minimum value mapped to lowermost color
    vmax : float
        Minimum value mapped to upppermost color
    cmap : instance of matplotlib.pylab.colormap
        Colors to be mapped to the values
    layout_scale : float
        Scaling factor for adjusting the relative size of the layout
        on the canvas
    dB : bool
        If True, log10 will be applied to the data.
    title : str
        Title of the figure.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Images of induced power at sensor locations
    """
    if mode is not None:
        if baseline is None:
            baseline = epochs.baseline
        times = epochs.times[::decim] * 1e3
        power = rescale(power.copy(), times, baseline, mode)
    if dB:
        power = 20 * np.log10(power)
    if vmin is None:
        vmin = power.min()
    if vmax is None:
        vmax = power.max()

    power_imshow = partial(_imshow_tfr, tfr=power.copy(), freq=freq)

    fig = _plot_topo(epochs.info, epochs.times, power_imshow, layout,
                     decim=decim, colorbar=colorbar, vmin=vmin, vmax=vmax,
                     cmap=cmap, layout_scale=layout_scale, title=title,
                     x_label='Time (s)', y_label='Frequency (Hz)')
    return fig


def plot_topo_phase_lock(epochs, phase, freq, layout, baseline=None,
                         mode='mean', decim=1, colorbar=True, vmin=None,
                         vmax=None, cmap=None, layout_scale=0.945,
                         title=None):
    """Plot phase locking values (PLV) on sensor layout

    Clicking on the PLV map of an individual sensor opens a new figure
    showing the PLV map of the selected sensor.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs used to generate the phase locking value
    phase_lock : 3D-array
        Phase locking value, second return value from
        mne.time_frequency.induced_power.
    freq : array-like
        Frequencies of interest as passed to induced_power
    layout : instance of Layout
        System specific sensor positions.
    baseline : tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode : 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent' | None
        Do baseline correction with ratio (phase is divided by mean
        phase during baseline) or z-score (phase is divided by standard
        deviation of phase during baseline after subtracting the mean,
        phase = [phase - mean(phase_baseline)] / std(phase_baseline)).
        If None, baseline no correction will be performed.
    decim : integer
        Increment for selecting each nth time slice
    colorbar : bool
        If true, colorbar will be added to the plot
    vmin : float
        Minimum value mapped to lowermost color
    vmax : float
        Minimum value mapped to upppermost color
    cmap : instance of matplotlib.pylab.colormap
        Colors to be mapped to the values
    layout_scale : float
        Scaling factor for adjusting the relative size of the layout
        on the canvas.
    title : str
        Title of the figure.
    Returns
    -------
    fig : Instance of matplotlib.figure.Figrue
        Phase lock images at sensor locations
    """
    if mode is not None:  # do baseline correction
        if baseline is None:
            baseline = epochs.baseline
        times = epochs.times[::decim] * 1e3
        phase = rescale(phase.copy(), times, baseline, mode)
    if vmin is None:
        vmin = phase.min()
    if vmax is None:
        vmax = phase.max()

    phase_imshow = partial(_imshow_tfr, tfr=phase.copy(), freq=freq)

    fig = _plot_topo(epochs.info, epochs.times, phase_imshow, layout,
                     decim=decim, colorbar=colorbar, vmin=vmin, vmax=vmax,
                     cmap=cmap, layout_scale=layout_scale, title=title,
                     x_label='Time (s)', y_label='Frequency (Hz)')

    return fig


def _erfimage_imshow(ax, ch_idx, tmin, tmax, vmin, vmax,
                     data=None, epochs=None, sigma=None,
                     order=None, scalings=None):
    """Aux function to plot erfimage on sensor topography"""

    this_data = data[:, ch_idx, :].copy()
    ch_type = channel_type(epochs.info, ch_idx)
    this_data *= scalings[ch_type]

    this_order = order
    if callable(order):
        this_order = order(epochs.times, this_data)

    if this_order is not None:
        this_data = this_data[this_order]

    this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

    ax.imshow(this_data, extent=[tmin, tmax, 0, len(data)], aspect='auto',
              origin='lower', vmin=vmin, vmax=vmax, picker=True)


def plot_topo_image_epochs(epochs, layout, sigma=0.3, vmin=None,
                           vmax=None, colorbar=True, order=None,
                           cmap=None, layout_scale=.95, title=None,
                           scalings=dict(eeg=1e6, grad=1e13, mag=1e15)):
    """Plot Event Related Potential / Fields image on topographies

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    layout: instance of Layout
        System specific sensor positions.
    sigma : float
        The standard deviation of the Gaussian smoothing to apply along
        the epoch axis to apply in the image.
    vmin : float
        The min value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers.
    vmax : float
        The max value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers.
    colorbar : bool
        Display or not a colorbar.
    order : None | array of int | callable
        If not None, order is used to reorder the epochs on the y-axis
        of the image. If it's an array of int it should be of length
        the number of good epochs. If it's a callable the arguments
        passed are the times vector and the data as 2d array
        (data.shape[1] == len(times)).
    cmap : instance of matplotlib.pylab.colormap
        Colors to be mapped to the values.
    layout_scale: float
        scaling factor for adjusting the relative size of the layout
        on the canvas.
    title : str
        Title of the figure.
    scalings : dict
        The scalings of the channel types to be applied for plotting.
    Returns
    -------
    fig : instacne fo matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """

    data = epochs.get_data()
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    erf_imshow = partial(_erfimage_imshow, scalings=scalings,
                         data=data, epochs=epochs, sigma=sigma)

    fig = _plot_topo(epochs.info, epochs.times, erf_imshow, layout, decim=1,
                     colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                     layout_scale=layout_scale, title=title,
                     x_label='Time (s)', y_label='Epoch')

    return fig


def plot_evoked(evoked, picks=None, unit=True, show=True, ylim=None,
                proj=False, xlim='tight', hline=None, units=dict(eeg='uV',
                grad='fT/cm', mag='fT'), scalings=dict(eeg=1e6, grad=1e13,
                mag=1e15), titles=dict(eeg='EEG', grad='Gradiometers',
                mag='Magnetometers'), axes=None):
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
    hline : list of floats | None
        The values at which show an horizontal line.
    units : dict
        The units of the channel types used for axes lables.
    scalings : dict
        The scalings of the channel types to be applied for plotting.
    titles : dict
        The titles associated with the channels.
    axes : instance of Axes | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
    """

    if scalings.keys() != units.keys() != titles.keys():
        raise ValueError('Scalings and units must have the same keys.')
    else:
        channel_types = sorted(scalings.keys())

    import pylab as pl
    if picks is None:
        picks = range(evoked.info['nchan'])
    types = [channel_type(evoked.info, idx) for idx in picks]
    n_channel_types = 0
    ch_types_used = []
    for t in channel_types:
        if t in types:
            n_channel_types += 1
            ch_types_used.append(t)

    if axes is None:
        pl.clf()
        axes = [pl.subplot(n_channel_types, 1, c)
                for c in range(n_channel_types)]
    if not isinstance(axes, list):
        axes = [axes]
    if not len(axes) == n_channel_types:
        raise ValueError('Number of axes (%g) must match number of channel '
                         'types (%g)' % (len(axes), n_channel_types))

    times = 1e3 * evoked.times  # time in miliseconds
    for ax, t in zip(axes, ch_types_used):
        ch_unit = units[t]
        this_scaling = scalings[t]
        if unit is False:
            this_scaling = 1.0
            ch_unit = 'NA'  # no unit
        idx = [picks[i] for i in range(len(picks)) if types[i] == t]
        if len(idx) > 0:
            D = this_scaling * evoked.data[idx, :]
            if proj:
                projs = activate_proj(evoked.info['projs'])
                this_ch_names = [evoked.ch_names[k] for k in idx]
                P, ncomp, _ = make_projector(projs, this_ch_names)
                D = np.dot(P, D)

            pl.axes(ax)
            ax.plot(times, D.T)
            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                pl.xlim(xlim)
            if ylim is not None and t in ylim:
                pl.ylim(ylim[t])
            pl.title(titles[t] + ' (%d channels)' % len(D))
            pl.xlabel('time (ms)')
            pl.ylabel('data (%s)' % ch_unit)

            if hline is not None:
                for h in hline:
                    pl.axhline(h, color='k', linestyle='--', linewidth=2)

    pl.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
    tight_layout()

    if show:
        pl.show()


def plot_sparse_source_estimates(src, stcs, colors=None, linewidth=2,
                                 fontsize=18, bgcolor=(.05, 0, .1),
                                 opacity=0.2, brain_color=(0.7, ) * 3,
                                 show=True, high_resolution=False,
                                 fig_name=None, fig_number=None, labels=None,
                                 modes=['cone', 'sphere'],
                                 scale_factors=[1, 0.6],
                                 verbose=None, **kwargs):
    """Plot source estimates obtained with sparse solver

    Active dipoles are represented in a "Glass" brain.
    If the same source is active in multiple source estimates it is
    displayed with a sphere otherwise with a cone in 3D.

    Parameters
    ----------
    src : dict
        The source space
    stcs : instance of SourceEstimate or list of instances of SourceEstimate
        The source estimates (up to 3)
    colors : list
        List of colors
    linewidth : int
        Line width in 2D plot
    fontsize : int
        Font size
    bgcolor : tuple of length 3
        Background color in 3D
    opacity : float in [0, 1]
        Opacity of brain mesh
    brain_color : tuple of length 3
        Brain color
    show : bool
        Show figures if True
    fig_name :
        Mayavi figure name
    fig_number :
        Pylab figure number
    labels : ndarray or list of ndarrays
        Labels to show sources in clusters. Sources with the same
        label and the waveforms within each cluster are presented in
        the same color. labels should be a list of ndarrays when
        stcs is a list ie. one label for each stc.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    kwargs : kwargs
        Keyword arguments to pass to mlab.triangular_mesh
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
                            use_faces, color=brain_color, opacity=opacity,
                            **kwargs)

    import pylab as pl
    # Show time courses
    pl.figure(fig_number)
    pl.clf()

    colors = cycle(colors)

    logger.info("Total number of active sources: %d" % len(unique_vertnos))

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


@verbose
def plot_cov(cov, info, exclude=[], colorbar=True, proj=False, show_svd=True,
             show=True, verbose=None):
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
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
            logger.info('    Created an SSP operator (subspace dimension'
                        ' = %d)' % ncomp)
            C = np.dot(P, np.dot(C, P.T))
        else:
            logger.info('    The projection vectors do not apply to these '
                        'channels.')

    import pylab as pl

    pl.figure(figsize=(2.5 * len(idx_names), 2.7))
    for k, (idx, name, _, _) in enumerate(idx_names):
        pl.subplot(1, len(idx_names), k + 1)
        pl.imshow(C[idx][:, idx], interpolation="nearest")
        pl.title(name)
    pl.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
    tight_layout()

    if show_svd:
        pl.figure()
        for k, (idx, name, unit, scaling) in enumerate(idx_names):
            _, s, _ = linalg.svd(C[idx][:, idx])
            pl.subplot(1, len(idx_names), k + 1)
            pl.ylabel('Noise std (%s)' % unit)
            pl.xlabel('Eigenvalue index')
            pl.semilogy(np.sqrt(s) * scaling)
            pl.title(name)
            tight_layout()

    if show:
        pl.show()


def plot_source_estimates(stc, subject, surface='inflated', hemi='lh',
                          colormap='hot', time_label='time=%0.2f ms',
                          smoothing_steps=10, fmin=5., fmid=10., fmax=15.,
                          transparent=True, time_viewer=False,
                          subjects_dir=None):
    """Plot SourceEstimates with PySurfer

    Note: PySurfer currently needs the SUBJECTS_DIR environment variable,
    which will automatically be set by this function. Plotting multiple
    SourceEstimates with different values for subjects_dir will cause
    PySurfer to use the wrong FreeSurfer surfaces when using methods of
    the returned Brain object. It is therefore recommended to set the
    SUBJECTS_DIR environment variable or always use the same value for
    subjects_dir (within the same Python session).

    Parameters
    ----------
    stc : SourceEstimates
        The source estimates to plot.
    subject : str
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None the environment will be used.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str, 'lh' | 'rh'
        The hemisphere to display.
    colormap : str
        The type of colormap to use.
    time_label : str
        How to print info about the time instant visualized.
    smoothing_steps : int
        The amount of smoothing
    fmin : float
        The minimum value to display.
    fmid : float
        The middle value on the colormap.
    fmax : float
        The maximum value for the colormap.
    transparent : bool
        If True, use a linear transparency between fmin and fmid.
    time_viewer : bool
        Display time viewer GUI.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.

    Returns
    -------
    brain : Brain
        A instance of surfer.viz.Brain from PySurfer.
    """
    from surfer import Brain, TimeViewer

    assert hemi in ['lh', 'rh']

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir)

    hemi_idx = 0 if hemi == 'lh' else 1

    if subject is None:
        if 'SUBJECT' in os.environ:
            subject = os.environ['SUBJECT']
        else:
            raise ValueError('SUBJECT environment variable not set')

    args = inspect.getargspec(Brain.__init__)[0]
    if 'subjects_dir' in args:
        brain = Brain(subject, hemi, surface, subjects_dir=subjects_dir)
    else:
        # Current PySurfer versions need the SUBJECTS_DIR env. var.
        # so we set it here. This is a hack as it can break other things
        os.environ['SUBJECTS_DIR'] = subjects_dir
        brain = Brain(subject, hemi, surface)

    if hemi_idx == 0:
        data = stc.data[:len(stc.vertno[0])]
    else:
        data = stc.data[len(stc.vertno[0]):]

    vertices = stc.vertno[hemi_idx]

    time = 1e3 * stc.times
    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=smoothing_steps, time=time,
                   time_label=time_label)

    # scale colormap and set time (index) to display
    brain.scale_data_colormap(fmin=fmin, fmid=fmid, fmax=fmax,
                              transparent=transparent)

    if time_viewer:
        viewer = TimeViewer(brain)

    return brain


@deprecated('Use plot_source_estimates. Will be removed in v0.7.')
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


def _plot_ica_panel_onpick(event, sources=None, ylims=None):
    """Onpick callback for plot_ica_panel"""

    # make sure that the swipe gesture in OS-X doesn't open many figures
    if event.mouseevent.inaxes is None or event.mouseevent.button != 1:
        return

    artist = event.artist
    try:
        import pylab as pl
        pl.figure()
        src_idx = artist._mne_src_idx
        component = artist._mne_component
        pl.plot(sources[src_idx], 'r')
        pl.ylim(ylims)
        pl.grid(linestyle='-', color='gray', linewidth=.25)
        pl.title(component)
    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers, so we print
        # it here to know what went wrong
        print err
        raise err


@verbose
def plot_ica_panel(sources, start=None, stop=None, n_components=None,
                   source_idx=None, ncol=3, nrow=10, verbose=None,
                   show=True):
    """Create panel plots of ICA sources

    Note. Inspired by an example from Carl Vogel's stats blog 'Will it Python?'

    Clicking on the plot of an individual source opens a new figure showing
    the source.

    Parameters
    ----------
    sources : ndarray
        Sources as drawn from ica.get_sources.
    start : int
        x-axis start index. If None from the beginning.
    stop : int
        x-axis stop index. If None to the end.
    n_components : int
        Number of components fitted.
    source_idx : array-like
        Indices for subsetting the sources.
    ncol : int
        Number of panel-columns.
    nrow : int
        Number of panel-rows.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    show : bool
        If True, plot will be shown, else just the figure is returned.

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
        logger.info('More sources selected than rows and cols specified.'
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
            this_ax = xs.plot(sources[idx], linewidth=0.5, color='red',
                              picker=1e9)
            xs.text(0.05, .95, component,
                    transform=panel_axes[row, col].transAxes,
                    verticalalignment='top')
            # emebed idx and comp. name to use in callback
            this_ax[0].__dict__['_mne_src_idx'] = idx
            this_ax[0].__dict__['_mne_component'] = component
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

    # register callback
    callback = partial(_plot_ica_panel_onpick, sources=sources, ylims=ylims)
    fig.canvas.mpl_connect('pick_event', callback)

    if show:
        pl.show()

    return fig


def plot_image_epochs(epochs, picks, sigma=0.3, vmin=None,
                      vmax=None, colorbar=True, order=None, show=True,
                      units=dict(eeg='uV', grad='fT/cm', mag='fT'),
                      scalings=dict(eeg=1e6, grad=1e13, mag=1e15)):
    """Plot Event Related Potential / Fields image

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    picks : int | array of int
        The indices of the channels to consider
    sigma : float
        The standard deviation of the Gaussian smoothing to apply along
        the epoch axis to apply in the image.
    vmin : float
        The min value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers
    vmax : float
        The max value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers
    colorbar : bool
        Display or not a colorbar
    order : None | array of int | callable
        If not None, order is used to reorder the epochs on the y-axis
        of the image. If it's an array of int it should be of length
        the number of good epochs. If it's a callable the arguments
        passed are the times vector and the data as 2d array
        (data.shape[1] == len(times)
    show : bool
        Show or not the figure at the end
    units : dict
        The units of the channel types used for axes lables.
    scalings : dict
        The scalings of the channel types to be applied for plotting.

    Returns
    -------
    figs : the list of matplotlib figures
        One figure per channel displayed
    """
    import pylab as pl

    if units.keys() != scalings.keys():
        raise ValueError('Scalings and units must have the same keys.')

    picks = np.atleast_1d(picks)
    evoked = epochs.average(picks)
    data = epochs.get_data()[:, picks, :]
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    figs = list()
    for i, (this_data, idx) in enumerate(zip(np.swapaxes(data, 0, 1), picks)):
        print idx
        this_fig = pl.figure()
        figs.append(this_fig)

        ch_type = channel_type(epochs.info, idx)
        this_data *= scalings[ch_type]

        this_order = order
        if callable(order):
            this_order = order(epochs.times, this_data)

        if this_order is not None:
            this_data = this_data[this_order]

        this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

        ax1 = pl.subplot2grid((3, 10), (0, 0), colspan=9, rowspan=2)
        im = pl.imshow(this_data,
                   extent=[1e3 * epochs.times[0], 1e3 * epochs.times[-1],
                           0, len(data)],
                   aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax)
        ax2 = pl.subplot2grid((3, 10), (2, 0), colspan=9, rowspan=1)
        if colorbar:
            ax3 = pl.subplot2grid((3, 10), (0, 9), colspan=1, rowspan=3)
        ax1.set_title(epochs.ch_names[idx])
        ax1.set_ylabel('Epochs')
        ax1.axis('auto')
        ax1.axis('tight')
        ax1.axvline(0, color='m', linewidth=3, linestyle='--')
        ax2.plot(1e3 * evoked.times, scalings[ch_type] * evoked.data[i])
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel(units[ch_type])
        ax2.set_ylim([vmin, vmax])
        ax2.axvline(0, color='m', linewidth=3, linestyle='--')
        if colorbar:
            pl.colorbar(im, cax=ax3)
            tight_layout()

    if show:
        pl.show()

    return figs


def mne_analyze_colormap(limits=[5, 10, 15], format='mayavi'):
    """Return a colormap similar to that used by mne_analyze

    Parameters
    ----------
    limits : list (or array) of length 3
        Bounds for the colormap.
    format : str
        Type of colormap to return. If 'matplotlib', will return a
        matplotlib.colors.LinearSegmentedColormap. If 'mayavi', will
        return an RGBA array of shape (256, 4).

    Returns
    -------
    cmap : instance of matplotlib.pylab.colormap | array
        A teal->blue->gray->red->yellow colormap.

    Notes
    -----
    For this will return a colormap that will display correctly for data
    that are scaled by the plotting function to span [-fmax, fmax].

    Examples
    --------
    The following code will plot a STC using standard MNE limits:

        colormap = mne.viz.mne_analyze_colormap(limits=[5, 10, 15])
        brain = stc.plot('fsaverage', 'inflated', 'rh', colormap)
        brain.scale_data_colormap(fmin=-15, fmid=0, fmax=15, transparent=False)

    """
    l = np.asarray(limits, dtype='float')
    if len(l) != 3:
        raise ValueError('limits must have 3 elements')
    if any(l < 0):
        raise ValueError('limits must all be positive')
    if any(np.diff(l) <= 0):
        raise ValueError('limits must be monotonically increasing')
    if format == 'matplotlib':
        from matplotlib import colors
        l = (np.concatenate((-np.flipud(l), l)) + l[-1]) / (2 * l[-1])
        cdict = {'red': ((l[0], 0.0, 0.0),
                         (l[1], 0.0, 0.0),
                         (l[2], 0.5, 0.5),
                         (l[3], 0.5, 0.5),
                         (l[4], 1.0, 1.0),
                         (l[5], 1.0, 1.0)),
                 'green': ((l[0], 1.0, 1.0),
                           (l[1], 0.0, 0.0),
                           (l[2], 0.5, 0.5),
                           (l[3], 0.5, 0.5),
                           (l[4], 0.0, 0.0),
                           (l[5], 1.0, 1.0)),
                 'blue': ((l[0], 1.0, 1.0),
                          (l[1], 1.0, 1.0),
                          (l[2], 0.5, 0.5),
                          (l[3], 0.5, 0.5),
                          (l[4], 0.0, 0.0),
                          (l[5], 0.0, 0.0))}
        return colors.LinearSegmentedColormap('mne_analyze', cdict)
    elif format == 'mayavi':
        l = np.concatenate((-np.flipud(l), [0], l)) / l[-1]
        r = np.array([0, 0, 0, 0, 1, 1, 1])
        g = np.array([1, 0, 0, 0, 0, 0, 1])
        b = np.array([1, 1, 1, 0, 0, 0, 0])
        a = np.array([1, 1, 0, 0, 0, 1, 1])
        xp = (np.arange(256) - 128) / 128.0
        colormap = np.r_[[np.interp(xp, l, 255 * c) for c in [r, g, b, a]]].T
        return colormap
    else:
        raise ValueError('format must be either matplotlib or mayavi')


def circular_layout(node_names, node_order, start_pos=90, start_between=True):
    """Create layout arranging nodes on a circle.

    Parameters
    ----------
    node_names : list of str
        Node names.
    node_order : list of str
        List with node names defining the order in which the nodes are
        arranged. Must have the elements as node_names but the order can be
        different. The nodes are arranged clockwise starting at "start_pos"
        degrees.
    start_pos : float
        Angle in degrees that defines where the first node is plotted.
    start_between : bool
        If True, the layout starts with the position between the nodes. This is
        the same as adding "180. / len(node_names)" to start_pos.

    Returns
    -------
    node_angles : array, shape=(len(node_names,))
        Node angles in degrees.
    """
    n_nodes = len(node_names)

    if len(node_order) != n_nodes:
        raise ValueError('node_order has to be the same length as node_names')

    # convert it to a list with indices
    node_order = [node_order.index(name) for name in node_names]
    node_order = np.array(node_order)
    if len(np.unique(node_order)) != n_nodes:
        raise ValueError('node_order has repeated entries')

    if start_between:
        start_pos += 180. / n_nodes

    node_angles = start_pos + 360 * node_order / float(n_nodes)

    return node_angles


def plot_connectivity_circle(con, node_names, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_colors=None, facecolor='black',
                             textcolor='white', node_edgecolor='black',
                             linewidth=1.5, colormap='hot', vmin=None,
                             vmax=None, colorbar=True, title=None):
    """Visualize connectivity as a circular graph.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.loria.fr/~rougier/coding/recipes.html

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strenght=abs(con))
        are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, "360. / len(node_names)" is
        used.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.

    Returns
    -------
    fig : instance of pyplot.Figure
        The figure handle.
    """
    import pylab as pl
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        node_width = 2 * np.pi / n_nodes
    else:
        node_width = node_width * np.pi / 180

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [pl.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indices = tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')

    # get the colormap
    if isinstance(colormap, basestring):
        colormap = pl.get_cmap(colormap)

    # Make figure background the same colors as axes
    fig = pl.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    axes = pl.subplot(111, polar=True, axisbg=facecolor)

    # No ticks, we'll put our own
    pl.xticks([])
    pl.yticks([])

    # Set y axes limit
    pl.ylim(0, 10)

    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        con_thresh = 0.

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initalize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start])
                           / float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end])
                         / float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 10

        # End point
        t1, r1 = node_angles[j], 10

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth, alpha=1.)
        axes.add_patch(patch)

    # Draw ring with colored nodes
    radii = np.ones(n_nodes) * 10
    bars = axes.bar(node_angles, radii, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=2, facecolor='.9',
                    align='center')

    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)

    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'

        pl.text(angle_rad, 10.4, name, size=10, rotation=angle_deg,
                rotation_mode='anchor', horizontalalignment=ha,
                verticalalignment='center', color=textcolor)

    if title is not None:
        pl.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.75)
        pl.figtext(0.03, 0.95, title, color=textcolor, fontsize=14)
    else:
        pl.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)

    if colorbar:
        sm = pl.cm.ScalarMappable(cmap=colormap,
                                  norm=pl.normalize(vmin=vmin, vmax=vmax))
        sm.set_array(np.linspace(vmin, vmax))
        ax = fig.add_axes([.92, 0.03, .015, .25])
        cb = fig.colorbar(sm, cax=ax)
        cb_yticks = pl.getp(cb.ax.axes, 'yticklabels')
        pl.setp(cb_yticks, color=textcolor)

    return fig
