"""Functions to plot M/EEG data e.g. topographies
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD
import os
import warnings
from itertools import cycle
from functools import partial
from copy import deepcopy
import math
from distutils.version import LooseVersion

import difflib
import tempfile
import webbrowser

import copy
import inspect
import numpy as np
from scipy import linalg
from scipy import ndimage
from matplotlib import delaunay
from warnings import warn
from collections import deque

# XXX : don't import pyplot here or you will break the doc

from .fixes import tril_indices, Counter
from .baseline import rescale
from .utils import (get_subjects_dir, get_config, set_config, _check_subject,
                    logger, verbose)
from .fiff import show_fiff, FIFF
from .fiff.pick import channel_type, pick_types
from .fiff.proj import make_projector, setup_proj
from .fixes import normalize_colors
from .utils import create_chunks
from .time_frequency import compute_raw_psd

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#473C8B', '#458B74',
          '#CD7F32', '#FF4040', '#ADFF2F', '#8E2323', '#FF1493']


DEFAULTS = dict(color=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='r',
                           emg='k', ref_meg='steelblue', misc='k', stim='k',
                           resp='k', chpi='k', exci='k', ias='k', syst='k'),
                units=dict(eeg='uV', grad='fT/cm', mag='fT', misc='AU'),
                scalings=dict(eeg=1e6, grad=1e13, mag=1e15, misc=1.0),
                scalings_plot_raw=dict(mag=1e-12, grad=4e-11, eeg=20e-6,
                                       eog=150e-6, ecg=5e-4, emg=1e-3,
                                       ref_meg=1e-12, misc=1e-3,
                                       stim=1, resp=1, chpi=1e-4, exci=1,
                                       ias=1, syst=1),
                ylim=dict(mag=(-600., 600.), grad=(-200., 200.),
                          eeg=(-200., 200.), misc=(-5., 5.)),
                titles=dict(eeg='EEG', grad='Gradiometers',
                            mag='Magnetometers', misc='misc'))


def _mutable_defaults(*mappings):
    """ To avoid dicts as default keyword arguments

    Use this function instead to resolve default dict values.
    Example usage:
    scalings, units = _mutable_defaults(('scalings', scalings,
                                         'units', units))
    """
    out = []
    for k, v in mappings:
        this_mapping = DEFAULTS[k]
        if v is not None:
            this_mapping = deepcopy(DEFAULTS[k])
            this_mapping.update(v)
        out += [this_mapping]
    return out


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
    cleaned = []
    for name in names:
        if ' ' in name:
            name = name.replace(' ', '')
        if '-' in name:
            name = name.split('-')[0]
        cleaned.append(name)

    return cleaned


def _check_delayed_ssp(container):
    """ Aux function to be used for interactive SSP selection
    """
    if container.proj is True:
        raise RuntimeError('Projs are already applied. Please initialize'
                           ' the data with proj set to False.')
    elif len(container.info['projs']) < 1:
        raise RuntimeError('No projs found in evoked.')


def tight_layout(pad=1.2, h_pad=None, w_pad=None):
    """ Adjust subplot parameters to give specified padding.

    Note. For plotting please use this function instead of plt.tight_layout

    Parameters
    ----------
    pad : float
        padding between the figure edge and the edges of subplots, as a
        fraction of the font-size.
    h_pad, w_pad : float
        padding (height/width) between edges of adjacent subplots.
        Defaults to `pad_inches`.
    """
    import matplotlib.pyplot as plt
    if plt.get_backend().lower() != 'agg':
        try:
            plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        except:
            msg = ('Matplotlib function \'tight_layout\'%s.'
                   ' Skipping subpplot adjusment.')
            if not hasattr(plt, 'tight_layout'):
                case = ' is not available'
            else:
                case = (' is not supported by your backend: `%s`'
                        % plt.get_backend())
            warn(msg % case)


def _plot_topo(info=None, times=None, show_func=None, layout=None,
               decim=None, vmin=None, vmax=None, ylim=None, colorbar=None,
               border='none', cmap=None, layout_scale=None, title=None,
               x_label=None, y_label=None, vline=None):
    """Helper function to plot on sensor layout"""
    import matplotlib.pyplot as plt
    orig_facecolor = plt.rcParams['axes.facecolor']
    orig_edgecolor = plt.rcParams['axes.edgecolor']
    try:
        if cmap is None:
            cmap = plt.cm.jet
        ch_names = _clean_names(info['ch_names'])
        plt.rcParams['axes.facecolor'] = 'k'
        fig = plt.figure(facecolor='k')
        pos = layout.pos.copy()
        tmin, tmax = times[0], times[-1]
        if colorbar:
            pos[:, :2] *= layout_scale
            plt.rcParams['axes.edgecolor'] = 'k'
            norm = normalize_colors(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(np.linspace(vmin, vmax))
            ax = plt.axes([0.015, 0.025, 1.05, .8], axisbg='k')
            cb = fig.colorbar(sm, ax=ax)
            cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
            plt.setp(cb_yticks, color='w')
        plt.rcParams['axes.edgecolor'] = border
        for idx, name in enumerate(_clean_names(layout.names)):
            if name in ch_names:
                ax = plt.axes(pos[idx], axisbg='k')
                ch_idx = ch_names.index(name)
                # hack to inlcude channel idx and name, to use in callback
                ax.__dict__['_mne_ch_name'] = name
                ax.__dict__['_mne_ch_idx'] = ch_idx

                if layout.kind == 'Vectorview-all' and ylim is not None:
                    this_type = {'mag': 0, 'grad': 1}[channel_type(info,
                                                                   ch_idx)]
                    ylim_ = [v[this_type] if _check_vlim(v) else
                             v for v in ylim]
                else:
                    ylim_ = ylim

                show_func(ax, ch_idx, tmin=tmin, tmax=tmax, vmin=vmin,
                          vmax=vmax, ylim=ylim_)

                if ylim_ and not any(v is None for v in ylim_):
                    plt.ylim(*ylim_)
                plt.xticks([], ())
                plt.yticks([], ())

        # register callback
        callback = partial(_plot_topo_onpick, show_func=show_func, tmin=tmin,
                           tmax=tmax, vmin=vmin, vmax=vmax, ylim=ylim,
                           colorbar=colorbar, title=title, x_label=x_label,
                           y_label=y_label,
                           vline=vline)

        fig.canvas.mpl_connect('pick_event', callback)
        if title is not None:
            plt.figtext(0.03, 0.9, title, color='w', fontsize=19)

    finally:
        # Revert global pyplot config
        plt.rcParams['axes.facecolor'] = orig_facecolor
        plt.rcParams['axes.edgecolor'] = orig_edgecolor

    return fig


def _plot_topo_onpick(event, show_func=None, tmin=None, tmax=None,
                      vmin=None, vmax=None, ylim=None, colorbar=False,
                      title=None, x_label=None, y_label=None, vline=None):
    """Onpick callback that shows a single channel in a new figure"""

    # make sure that the swipe gesture in OS-X doesn't open many figures
    if event.mouseevent.inaxes is None or event.mouseevent.button != 1:
        return

    artist = event.artist
    try:
        import matplotlib.pyplot as plt
        ch_idx = artist.axes._mne_ch_idx
        fig, ax = plt.subplots(1)
        ax.set_axis_bgcolor('k')
        show_func(plt, ch_idx, tmin, tmax, vmin, vmax, ylim=ylim,
                  vline=vline)
        if colorbar:
            plt.colorbar()
        if title is not None:
            plt.title(title + ' ' + artist.axes._mne_ch_name)
        else:
            plt.title(artist.axes._mne_ch_name)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers, so we print
        # it here to know what went wrong
        print err
        raise err


def _imshow_tfr(ax, ch_idx, tmin, tmax, vmin, vmax, ylim=None, tfr=None,
                freq=None, vline=None):
    """ Aux function to show time-freq map on topo """
    extent = (tmin, tmax, freq[0], freq[-1])
    ax.imshow(tfr[ch_idx], extent=extent, aspect="auto", origin="lower",
              vmin=vmin, vmax=vmax, picker=True)


def _plot_timeseries(ax, ch_idx, tmin, tmax, vmin, vmax, ylim, data, color,
                     times, vline=None):
    """ Aux function to show time series on topo """
    picker_flag = False
    for data_, color_ in zip(data, color):
        if not picker_flag:
            # use large tol for picker so we can click anywhere in the axes
            ax.plot(times, data_[ch_idx], color_, picker=1e9)
            picker_flag = True
        else:
            ax.plot(times, data_[ch_idx], color_)
    if vline:
        import matplotlib.pyplot as plt
        [plt.axvline(x, color='w', linewidth=0.5) for x in vline]


def _check_vlim(vlim):
    """AUX function"""
    return not np.isscalar(vlim) and not vlim is None


def plot_topo(evoked, layout=None, layout_scale=0.945, color=None,
              border='none', ylim=None, scalings=None, title=None, proj=False,
              vline=[0.0]):
    """Plot 2D topography of evoked responses.

    Clicking on the plot of an individual sensor opens a new figure showing
    the evoked response for the selected sensor.

    Parameters
    ----------
    evoked : list of Evoked | Evoked
        The evoked response to plot.
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    layout_scale: float
        Scaling factor for adjusting the relative size of the layout
        on the canvas
    color : list of color objects | color object | None
        Everything matplotlib accepts to specify colors. If not list-like,
        the color specified will be repeated. If None, colors are
        automatically drawn.
    border : str
        matplotlib borders style to be used for each sensor plot.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    ylim : dict | None
        ylim for plots. The value determines the upper and lower subplot
        limits. e.g. ylim = dict(eeg=[-200e-6, 200e6]). Valid keys are eeg,
        mag, grad, misc. If None, the ylim parameter for each channel is
        determined by the maximum absolute peak.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    title : str
        Title of the figure.
    vline : list of floats | None
        The values at which to show a vertical line.

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

    info = evoked[0].info
    ch_names = evoked[0].ch_names
    if not all([e.ch_names == ch_names for e in evoked]):
        raise ValueError('All evoked.picks must be the same')
    ch_names = _clean_names(ch_names)

    if layout is None:
        from .layouts.layout import find_layout
        layout = find_layout(info['chs'])

    # XXX. at the moment we are committed to 1- / 2-sensor-types layouts
    layout = copy.deepcopy(layout)
    layout.names = _clean_names(layout.names)
    chs_in_layout = set(layout.names) & set(ch_names)
    types_used = set(channel_type(info, ch_names.index(ch))
                     for ch in chs_in_layout)
    # one check for all vendors
    meg_types = ['mag'], ['grad'], ['mag', 'grad'],
    is_meg = any(types_used == set(k) for k in meg_types)
    if is_meg:
        types_used = list(types_used)[::-1]  # -> restore kwarg order
        picks = [pick_types(info, meg=kk, ref_meg=False, exclude=[])
                 for kk in types_used]
    else:
        types_used_kwargs = dict((t, True) for t in types_used)
        picks = [pick_types(info, meg=False, **types_used_kwargs)]
    assert isinstance(picks, list) and len(types_used) == len(picks)

    scalings = _mutable_defaults(('scalings', scalings))[0]
    evoked = [e.copy() for e in evoked]
    for e in evoked:
        for pick, t in zip(picks, types_used):
            e.data[pick] = e.data[pick] * scalings[t]

    if proj is True and all([e.proj is not True for e in evoked]):
        evoked = [e.apply_proj() for e in evoked]
    elif proj == 'interactive':  # let it fail early.
        for e in evoked:
            _check_delayed_ssp(e)

    plot_fun = partial(_plot_timeseries, data=[e.data for e in evoked],
                       color=color, times=times, vline=vline)

    if ylim is None:
        set_ylim = lambda x: np.abs(x).max()
        ylim_ = [set_ylim([e.data[t] for e in evoked]) for t in picks]
        ymax = np.array(ylim_)
        ylim_ = (-ymax, ymax)
    elif isinstance(ylim, dict):
        ylim_ = _mutable_defaults(('ylim', ylim))[0]
        ylim_ = [ylim_[kk] for kk in types_used]
        ylim_ = zip(*[np.array(yl) for yl in ylim_])
    else:
        raise ValueError('ylim must be None ore a dict')

    fig = _plot_topo(info=info, times=times, show_func=plot_fun, layout=layout,
                     decim=1, colorbar=False, ylim=ylim_, cmap=None,
                     layout_scale=layout_scale, border=border, title=title,
                     x_label='Time (s)', vline=vline)

    if proj == 'interactive':
        for e in evoked:
            _check_delayed_ssp(e)
        params = dict(evokeds=evoked, times=times,
                      plot_update_proj_callback=_plot_update_evoked_topo,
                      projs=evoked[0].info['projs'], fig=fig)
        _draw_proj_checkbox(None, params)

    return fig


def _plot_update_evoked_topo(params, bools):
    """Helper function to update topo sensor plots"""
    evokeds, times, fig = [params[k] for k in 'evokeds', 'times', 'fig']

    projs = [proj for ii, proj in enumerate(params['projs'])
             if ii in np.where(bools)[0]]

    params['proj_bools'] = bools
    evokeds = [e.copy() for e in evokeds]
    for e in evokeds:
        e.info['projs'] = []
        e.add_proj(projs)
        e.apply_proj()

    # make sure to only modify the time courses, not the ticks
    axes = fig.get_axes()
    n_lines = len(axes[0].lines)
    n_diff = len(evokeds) - n_lines
    ax_slice = slice(abs(n_diff)) if n_diff < 0 else slice(n_lines)
    for ax in axes:
        lines = ax.lines[ax_slice]
        for line, evoked in zip(lines, evokeds):
            line.set_data(times, evoked.data[ax._mne_ch_idx])

    fig.canvas.draw()


def plot_topo_tfr(epochs, tfr, freq, layout=None, colorbar=True, vmin=None,
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
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    colorbar : bool
        If true, colorbar will be added to the plot
    vmin : float
        Minimum value mapped to lowermost color
    vmax : float
        Minimum value mapped to upppermost color
    cmap : instance of matplotlib.pyplot.colormap
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

    if layout is None:
        from .layouts.layout import find_layout
        layout = find_layout(epochs.info['chs'])
    layout = copy.deepcopy(layout)
    layout.names = _clean_names(layout.names)

    tfr_imshow = partial(_imshow_tfr, tfr=tfr.copy(), freq=freq)

    fig = _plot_topo(info=epochs.info, times=epochs.times,
                     show_func=tfr_imshow, layout=layout, border='w',
                     colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                     layout_scale=layout_scale, title=title,
                     x_label='Time (s)', y_label='Frequency (Hz)')

    return fig


def plot_topo_power(epochs, power, freq, layout=None, baseline=None,
                    mode='mean', decim=1, colorbar=True, vmin=None, vmax=None,
                    cmap=None, layout_scale=0.945, dB=True, title=None):
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
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
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
    cmap : instance of matplotlib.pyplot.colormap
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
    times = epochs.times[::decim] * 1e3
    if mode is not None:
        if baseline is None:
            baseline = epochs.baseline
        power = rescale(power.copy(), times, baseline, mode)
    if dB:
        power = 20 * np.log10(power)
    if vmin is None:
        vmin = power.min()
    if vmax is None:
        vmax = power.max()
    if layout is None:
        from .layouts.layout import find_layout
        layout = find_layout(epochs.info['chs'])
    layout = copy.deepcopy(layout)
    layout.names = _clean_names(layout.names)

    power_imshow = partial(_imshow_tfr, tfr=power.copy(), freq=freq)

    fig = _plot_topo(info=epochs.info, times=times,
                     show_func=power_imshow, layout=layout, decim=decim,
                     colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                     layout_scale=layout_scale, title=title, border='w',
                     x_label='Time (s)', y_label='Frequency (Hz)')

    return fig


def plot_topo_phase_lock(epochs, phase, freq, layout=None, baseline=None,
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
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
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
    cmap : instance of matplotlib.pyplot.colormap
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
    times = epochs.times[::decim] * 1e3
    if mode is not None:
        if baseline is None:
            baseline = epochs.baseline
        phase = rescale(phase.copy(), times, baseline, mode)
    if vmin is None:
        vmin = phase.min()
    if vmax is None:
        vmax = phase.max()
    if layout is None:
        from .layouts.layout import find_layout
        layout = find_layout(epochs.info['chs'])
    layout = copy.deepcopy(layout)
    layout.names = _clean_names(layout.names)

    phase_imshow = partial(_imshow_tfr, tfr=phase.copy(), freq=freq)

    fig = _plot_topo(info=epochs.info, times=times,
                     show_func=phase_imshow, layout=layout, decim=decim,
                     colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                     layout_scale=layout_scale, title=title, border='w',
                     x_label='Time (s)', y_label='Frequency (Hz)')

    return fig


def _erfimage_imshow(ax, ch_idx, tmin, tmax, vmin, vmax, ylim=None,
                     data=None, epochs=None, sigma=None,
                     order=None, scalings=None, vline=None):
    """Aux function to plot erfimage on sensor topography"""

    this_data = data[:, ch_idx, :].copy()
    ch_type = channel_type(epochs.info, ch_idx)
    if not ch_type in scalings:
        raise KeyError('%s channel type not in scalings' % ch_type)
    this_data *= scalings[ch_type]

    if callable(order):
        order = order(epochs.times, this_data)

    if order is not None:
        this_data = this_data[order]

    this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

    ax.imshow(this_data, extent=[tmin, tmax, 0, len(data)], aspect='auto',
              origin='lower', vmin=vmin, vmax=vmax, picker=True)


def plot_topo_image_epochs(epochs, layout=None, sigma=0.3, vmin=None,
                           vmax=None, colorbar=True, order=None, cmap=None,
                           layout_scale=.95, title=None, scalings=None):
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
    cmap : instance of matplotlib.pyplot.colormap
        Colors to be mapped to the values.
    layout_scale: float
        scaling factor for adjusting the relative size of the layout
        on the canvas.
    title : str
        Title of the figure.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If
        None, defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.

    Returns
    -------
    fig : instacne fo matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    scalings = _mutable_defaults(('scalings', scalings))[0]
    data = epochs.get_data()
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    if layout is None:
        from .layouts.layout import find_layout
        layout = find_layout(epochs.info['chs'])
    layout = copy.deepcopy(layout)
    layout.names = _clean_names(layout.names)

    erf_imshow = partial(_erfimage_imshow, scalings=scalings, order=order,
                         data=data, epochs=epochs, sigma=sigma)

    fig = _plot_topo(info=epochs.info, times=epochs.times,
                     show_func=erf_imshow, layout=layout, decim=1,
                     colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                     layout_scale=layout_scale, title=title,
                     border='w', x_label='Time (s)', y_label='Epoch')

    return fig


def plot_evoked_topomap(evoked, times=None, ch_type='mag', layout=None,
                        vmax=None, cmap='RdBu_r', sensors='k,', colorbar=True,
                        scale=None, unit=None, res=256, size=1, format='%3.1f',
                        proj=False, show=True):
    """Plot topographic maps of specific time points of evoked data

    Parameters
    ----------
    evoked : Evoked
        The Evoked object.
    times : float | array of floats | None.
        The time point(s) to plot. If None, 10 topographies will be shown
        will a regular time spacing between the first and last time instant.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg'
        The channel type to plot. For 'grad', the gradiometers are collected in
        pairs and the RMS for each pair is plotted.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    vmax : scalar
        The value specfying the range of the color scale (-vmax to +vmax). If
        None, the largest absolute value in the data is used.
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    scale : float | None
        Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
        for grad and 1e15 for mag.
    units : str | None
        The units of the channel types used for colorbar lables. If
        scale == None the unit is automatically determined.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : float
        Side length per topomap in inches.
    format : str
        String format for colorbar values.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be show.
    show : bool
        Call pyplot.show() at the end.
    """
    import matplotlib.pyplot as plt

    if scale is None:
        if ch_type.startswith('planar'):
            key = 'grad'
        else:
            key = ch_type
        scale = DEFAULTS['scalings'][key]
        unit = DEFAULTS['units'][key]

    if times is None:
        times = np.linspace(evoked.times[0], evoked.times[-1], 10)
    elif np.isscalar(times):
        times = [times]
    if len(times) > 20:
        raise RuntimeError('Too many plots requested. Please pass fewer '
                           'than 20 time instants.')
    tmin, tmax = evoked.times[[0, -1]]
    for t in times:
        if not tmin <= t <= tmax:
            raise ValueError('Times should be between %0.3f and %0.3f. (Got '
                             '%0.3f).' % (tmin, tmax, t))

    picks, pos, merge_grads = _prepare_topo_plot(evoked, ch_type, layout)

    n = len(times)
    nax = n + bool(colorbar)
    width = size * nax
    height = size * 1. + max(0, 0.1 * (3 - size))
    fig = plt.figure(figsize=(width, height))
    w_frame = plt.rcParams['figure.subplot.wspace'] / (2 * nax)
    top_frame = max(.05, .2 / size)
    fig.subplots_adjust(left=w_frame, right=1 - w_frame, bottom=0,
                        top=1 - top_frame)
    time_idx = [np.where(evoked.times >= t)[0][0] for t in times]

    if proj is True and evoked.proj is not True:
        data = evoked.copy().apply_proj().data
    else:
        data = evoked.data

    data = data[np.ix_(picks, time_idx)] * scale
    if merge_grads:
        from .layouts.layout import _merge_grad_data
        data = _merge_grad_data(data)
    vmax = vmax or np.max(np.abs(data))
    images = []
    for i, t in enumerate(times):
        plt.subplot(1, nax, i + 1)
        images.append(plot_topomap(data[:, i], pos, vmax=vmax, cmap=cmap,
                      sensors=sensors, res=res))
        plt.title('%i ms' % (t * 1000))

    if colorbar:
        cax = plt.subplot(1, n + 1, n + 1)
        plt.colorbar(cax=cax, ticks=[-vmax, 0, vmax], format=format)
        # resize the colorbar (by default the color fills the whole axes)
        cpos = cax.get_position()
        cpos.x0 = 1 - (.7 + .1 / size) / nax
        cpos.x1 = cpos.x0 + .1 / nax
        cpos.y0 = .1
        cpos.y1 = .7
        cax.set_position(cpos)
        if unit is not None:
            cax.set_title(unit)

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
                      picks=picks, images=images, time_idx=time_idx,
                      scale=scale, merge_grads=merge_grads, res=res, pos=pos,
                      plot_update_proj_callback=_plot_update_evoked_topomap)
        _draw_proj_checkbox(None, params)

    if show:
        plt.show()

    return fig


def _plot_update_evoked_topomap(params, bools):
    """ Helper to update topomaps """
    projs = [proj for ii, proj in enumerate(params['projs'])
             if ii in np.where(bools)[0]]

    params['proj_bools'] = bools
    new_evoked = params['evoked'].copy()
    new_evoked.info['projs'] = []
    new_evoked.add_proj(projs)
    new_evoked.apply_proj()

    data = new_evoked.data[np.ix_(params['picks'],
                                  params['time_idx'])] * params['scale']
    if params['merge_grads']:
        from .layouts.layout import _merge_grad_data
        data = _merge_grad_data(data)

    pos = np.asarray(params['pos'])
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    xmin, xmax = pos_x.min(), pos_x.max()
    ymin, ymax = pos_y.min(), pos_y.max()
    triang = delaunay.Triangulation(pos_x, pos_y)
    x = np.linspace(xmin, xmax, params['res'])
    y = np.linspace(ymin, ymax, params['res'])
    xi, yi = np.meshgrid(x, y)

    for ii, im in enumerate(params['images']):
        interp = triang.linear_interpolator(data[:, ii])
        im_ = interp[yi.min():yi.max():complex(0, yi.shape[0]),
                     xi.min():xi.max():complex(0, xi.shape[1])]
        im_ = np.ma.masked_array(im_, im_ == np.nan)
        im.set_data(im_)
    params['fig'].canvas.draw()


def plot_projs_topomap(projs, layout=None, cmap='RdBu_r', sensors='k,',
                       colorbar=False, res=256, size=1, show=True):
    """Plot topographic maps of SSP projections

    Parameters
    ----------
    projs : list of Projection
        The projections
    layout : None | Layout | list of Layout
        Layout instance specifying sensor positions (does not need to be
        specified for Neuromag data). Or a list of Layout if projections
        are from different sensor types.
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : scalar
        Side length of the topomaps in inches (only applies when plotting
        multiple topomaps at a time).
    show : bool
        Show figures if True
    """
    import matplotlib.pyplot as plt

    if layout is None:
        from .layouts import read_layout
        layout = read_layout('Vectorview-all')

    if not isinstance(layout, list):
        layout = [layout]

    n_projs = len(projs)
    nrows = math.floor(math.sqrt(n_projs))
    ncols = math.ceil(n_projs / nrows)

    plt.clf()
    for k, proj in enumerate(projs):

        ch_names = _clean_names(proj['data']['col_names'])
        data = proj['data']['data'].ravel()

        idx = []
        for l in layout:
            l = copy.deepcopy(l)
            l.names = _clean_names(l.names)
            is_vv = l.kind.startswith('Vectorview')
            if is_vv:
                from .layouts.layout import _pair_grad_sensors_from_ch_names
                grad_pairs = _pair_grad_sensors_from_ch_names(ch_names)
                if grad_pairs:
                    ch_names = [ch_names[i] for i in grad_pairs]

            idx = [l.names.index(c) for c in ch_names if c in l.names]
            if len(idx) == 0:
                continue

            pos = l.pos[idx]
            if is_vv and grad_pairs:
                from .layouts.layout import _merge_grad_data
                shape = (len(idx) / 2, 2, -1)
                pos = pos.reshape(shape).mean(axis=1)
                data = _merge_grad_data(data[grad_pairs]).ravel()

            break

        ax = plt.subplot(nrows, ncols, k + 1)
        ax.set_title(proj['desc'])
        if len(idx):
            plot_topomap(data, pos, vmax=None, cmap=cmap,
                         sensors=sensors, res=res)
            if colorbar:
                plt.colorbar()
        else:
            raise RuntimeError('Cannot find a proper layout for projection %s'
                               % proj['desc'])

    if show:
        plt.show()


def plot_topomap(data, pos, vmax=None, cmap='RdBu_r', sensors='k,', res=100,
                 axis=None):
    """Plot a topographic map as image

    Parameters
    ----------
    data : array, length = n_points
        The data values to plot.
    pos : array, shape = (n_points, 2)
        For each data point, the x and y coordinates.
    vmax : scalar
        The value specfying the range of the color scale (-vmax to +vmax). If
        None, the largest absolute value in the data is used.
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses).
    res : int
        The resolution of the topomap image (n pixels along each side).
    axis : instance of Axes | None
        The axis to plot to. If None, the current axis will be used.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    pos = np.asarray(pos)
    if data.ndim > 1:
        err = ("Data needs to be array of shape (n_sensors,); got shape "
               "%s." % str(data.shape))
        raise ValueError(err)
    elif len(data) != len(pos):
        err = ("Data and pos need to be of same length. Got data of shape %s, "
               "pos of shape %s." % (str(), str()))

    axes = plt.gca()
    axes.set_frame_on(False)

    vmax = vmax or np.abs(data).max()

    plt.xticks(())
    plt.yticks(())

    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    ax = axis if axis else plt
    if sensors:
        if sensors is True:
            sensors = 'k,'
        ax.plot(pos_x, pos_y, sensors)

    xmin, xmax = pos_x.min(), pos_x.max()
    ymin, ymax = pos_y.min(), pos_y.max()
    triang = delaunay.Triangulation(pos_x, pos_y)
    interp = triang.linear_interpolator(data)
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    xi, yi = np.meshgrid(x, y)

    im = interp[yi.min():yi.max():complex(0, yi.shape[0]),
                xi.min():xi.max():complex(0, xi.shape[1])]
    im = np.ma.masked_array(im, im == np.nan)
    im = ax.imshow(im, cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower',
                   aspect='equal', extent=(xmin, xmax, ymin, ymax))
    return im


def plot_evoked(evoked, picks=None, exclude='bads', unit=True, show=True,
                ylim=None, proj=False, xlim='tight', hline=None, units=None,
                scalings=None, titles=None, axes=None):
    """Plot evoked data

    Note: If bad channels are not excluded they are shown in red.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    picks : None | array-like of int
        The indices of channels to plot. If None show all.
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded.
    unit : bool
        Scale plot with channel (SI) unit.
    show : bool
        Call pyplot.show() as the end or not.
    ylim : dict | None
        ylim for plots. e.g. ylim = dict(eeg=[-200e-6, 200e6])
        Valid keys are eeg, mag, grad, misc. If None, the ylim parameter
        for each channel equals the pyplot default.
    xlim : 'tight' | tuple | None
        xlim for plots.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    hline : list of floats | None
        The values at which to show an horizontal line.
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    titles : dict | None
        The titles associated with the channels. If None, defaults to
        `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
    axes : instance of Axes | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
    """
    import matplotlib.pyplot as plt
    if axes is not None and proj == 'interactive':
        raise RuntimeError('Currently only single axis figures are supported'
                           ' for interactive SSP selection.')

    scalings, titles, units = _mutable_defaults(('scalings', scalings),
                                                ('titles', titles),
                                                ('units', units))

    channel_types = set(key for d in [scalings, titles, units] for key in d)
    if picks is None:
        picks = range(evoked.info['nchan'])

    bad_ch_idx = [evoked.ch_names.index(ch) for ch in evoked.info['bads']
                  if ch in evoked.ch_names]
    if len(exclude) > 0:
        if isinstance(exclude, basestring) and exclude == 'bads':
            exclude = bad_ch_idx
        elif (isinstance(exclude, list)
              and all([isinstance(ch, basestring) for ch in exclude])):
            exclude = [evoked.ch_names.index(ch) for ch in exclude]
        else:
            raise ValueError('exclude has to be a list of channel names or '
                             '"bads"')

        picks = list(set(picks).difference(exclude))

    types = [channel_type(evoked.info, idx) for idx in picks]
    n_channel_types = 0
    ch_types_used = []
    for t in channel_types:
        if t in types:
            n_channel_types += 1
            ch_types_used.append(t)

    axes_init = axes  # remember if axes where given as input
    
    fig = None
    if axes is None:
        fig, axes = plt.subplots(n_channel_types, 1)

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)

    if axes_init is not None:
        fig = axes[0].get_figure()
    
    if not len(axes) == n_channel_types:
        raise ValueError('Number of axes (%g) must match number of channel '
                         'types (%g)' % (len(axes), n_channel_types))


    # instead of projecting during each iteration let's use the mixin here.
    if proj is True and evoked.proj is not True:
        evoked = evoked.copy()
        evoked.apply_proj()

    times = 1e3 * evoked.times  # time in miliseconds
    for ax, t in zip(axes, ch_types_used):
        ch_unit = units[t]
        this_scaling = scalings[t]
        if unit is False:
            this_scaling = 1.0
            ch_unit = 'NA'  # no unit
        idx = [picks[i] for i in range(len(picks)) if types[i] == t]
        if len(idx) > 0:
            if any([i in bad_ch_idx for i in idx]):
                colors = ['k'] * len(idx)
                for i in bad_ch_idx:
                    if i in idx:
                        colors[idx.index(i)] = 'r'

                ax._get_lines.color_cycle = iter(colors)
            else:
                ax._get_lines.color_cycle = cycle(['k'])

            D = this_scaling * evoked.data[idx, :]
            # plt.axes(ax)
            ax.plot(times, D.T)
            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                ax.set_xlim(xlim)
            if ylim is not None and t in ylim:
                ax.set_ylim(ylim[t])
            ax.set_title(titles[t] + ' (%d channel%s)' % (
                         len(D), 's' if len(D) > 1 else ''))
            ax.set_xlabel('time (ms)')
            ax.set_ylabel('data (%s)' % ch_unit)

            if hline is not None:
                for h in hline:
                    ax.axhline(h, color='r', linestyle='--', linewidth=2)

    if axes_init is None:
        plt.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
        tight_layout()

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
                      axes=axes, types=types, units=units, scalings=scalings,
                      unit=unit, ch_types_used=ch_types_used, picks=picks,
                      plot_update_proj_callback=_plot_update_evoked)
        _draw_proj_checkbox(None, params)

    if show and plt.get_backend() != 'agg':
        fig.show()

    return fig


def _plot_update_evoked(params, bools):
    """ update the plot evoked lines
    """
    picks, evoked = [params[k] for k in 'picks', 'evoked']
    times = evoked.times * 1e3
    projs = [proj for ii, proj in enumerate(params['projs'])
             if ii in np.where(bools)[0]]
    params['proj_bools'] = bools
    new_evoked = evoked.copy()
    new_evoked.info['projs'] = []
    new_evoked.add_proj(projs)
    new_evoked.apply_proj()
    for ax, t in zip(params['axes'], params['ch_types_used']):
        this_scaling = params['scalings'][t]
        idx = [picks[i] for i in range(len(picks)) if params['types'][i] == t]
        D = this_scaling * new_evoked.data[idx, :]
        [line.set_data(times, di) for line, di in zip(ax.lines, D)]
    params['fig'].canvas.draw()


def _draw_proj_checkbox(event, params, draw_current_state=True):
    """Toggle options (projectors) dialog"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    projs = params['projs']
    # turn on options dialog
    fig_proj = figure_nobar()
    fig_proj.canvas.set_window_title('SSP projection vectors')
    ax_temp = plt.axes((0, 0, 1, 1))
    ax_temp.get_yaxis().set_visible(False)
    ax_temp.get_xaxis().set_visible(False)
    fig_proj.add_axes(ax_temp)
    labels = [p['desc'] for p in projs]
    actives = [p['active'] for p in projs] if draw_current_state else \
              [True] * len(params['projs'])
    proj_checks = mpl.widgets.CheckButtons(ax_temp, labels=labels,
                                           actives=actives)
    # change already-applied projectors to red
    for ii, p in enumerate(projs):
        if p['active'] is True:
            for x in proj_checks.lines[ii]:
                x.set_color('r')
    # make minimal size
    width = max([len(p['desc']) for p in projs]) / 6.0 + 0.5
    height = len(projs) / 6.0 + 0.5
    # have to try/catch when there's no toolbar
    try:
        fig_proj.set_size_inches((width, height), forward=True)
    except Exception:
        pass
    # pass key presses from option dialog over
    proj_checks.on_clicked(partial(_toggle_proj, params=params))
    params['proj_checks'] = proj_checks
    # this should work for non-test cases
    try:
        fig_proj.canvas.show()
    except Exception:
        pass


def plot_sparse_source_estimates(src, stcs, colors=None, linewidth=2,
                                 fontsize=18, bgcolor=(.05, 0, .1),
                                 opacity=0.2, brain_color=(0.7,) * 3,
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
    if mlab.options.backend != 'test':
        f.scene.disable_render = True
    surface = mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
                                   use_faces, color=brain_color,
                                   opacity=opacity, **kwargs)

    import matplotlib.pyplot as plt
    # Show time courses
    plt.figure(fig_number)
    plt.clf()

    colors = cycle(colors)

    logger.info("Total number of active sources: %d" % len(unique_vertnos))

    if labels is not None:
        colors = [colors.next() for _ in
                  range(np.unique(np.concatenate(labels).ravel()).size)]

    for idx, v in enumerate(unique_vertnos):
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

        if (isinstance(scale_factor, (np.ndarray, list, tuple))
            and len(unique_vertnos) == len(scale_factor)):
            scale_factor = scale_factor[idx]

        x, y, z = points[v]
        nx, ny, nz = normals[v]
        mlab.quiver3d(x, y, z, nx, ny, nz, color=color_converter.to_rgb(c),
                      mode=mode, scale_factor=scale_factor)

        for k in ind:
            vertno = vertnos[k]
            mask = (vertno == v)
            assert np.sum(mask) == 1
            linestyle = linestyles[k]
            plt.plot(1e3 * stc.times, 1e9 * stcs[k].data[mask].ravel(), c=c,
                     linewidth=linewidth, linestyle=linestyle)

    plt.xlabel('Time (ms)', fontsize=18)
    plt.ylabel('Source amplitude (nAm)', fontsize=18)

    if fig_name is not None:
        plt.title(fig_name)

    if show:
        plt.show()

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
        The covariance matrix.
    info: dict
        Measurement info.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any channel.
        If 'bads', exclude info['bads'].
    colorbar : bool
        Show colorbar or not.
    proj : bool
        Apply projections or not.
    show : bool
        Call pyplot.show() as the end or not.
    show_svd : bool
        Plot also singular values of the noise covariance for each sensor type.
        We show square roots ie. standard deviations.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    if exclude == 'bads':
        exclude = info['bads']
    ch_names = [n for n in cov.ch_names if not n in exclude]
    ch_idx = [cov.ch_names.index(n) for n in ch_names]
    info_ch_names = info['ch_names']
    sel_eeg = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude=exclude)
    sel_mag = pick_types(info, meg='mag', eeg=False, ref_meg=False,
                         exclude=exclude)
    sel_grad = pick_types(info, meg='grad', eeg=False, ref_meg=False,
                          exclude=exclude)
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

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.5 * len(idx_names), 2.7))
    for k, (idx, name, _, _) in enumerate(idx_names):
        plt.subplot(1, len(idx_names), k + 1)
        plt.imshow(C[idx][:, idx], interpolation="nearest")
        plt.title(name)
    plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
    tight_layout()

    if show_svd:
        plt.figure()
        for k, (idx, name, unit, scaling) in enumerate(idx_names):
            _, s, _ = linalg.svd(C[idx][:, idx])
            plt.subplot(1, len(idx_names), k + 1)
            plt.ylabel('Noise std (%s)' % unit)
            plt.xlabel('Eigenvalue index')
            plt.semilogy(np.sqrt(s) * scaling)
            plt.title(name)
            tight_layout()

    if show:
        plt.show()


def plot_source_estimates(stc, subject=None, surface='inflated', hemi='lh',
                          colormap='hot', time_label='time=%0.2f ms',
                          smoothing_steps=10, fmin=5., fmid=10., fmax=15.,
                          transparent=True, alpha=1.0, time_viewer=False,
                          config_opts={}, subjects_dir=None, figure=None,
                          views='lat', colorbar=True):
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
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None stc.subject will be used. If that
        is None, the environment will be used.
    surface : str
        The type of surface (inflated, white etc.).
    hemi : str, 'lh' | 'rh' | 'split' | 'both'
        The hemisphere to display. Using 'both' or 'split' requires
        PySurfer version 0.4 or above.
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
    alpha : float
        Alpha value to apply globally to the overlay.
    time_viewer : bool
        Display time viewer GUI.
    config_opts : dict
        Keyword arguments for Brain initialization.
        See pysurfer.viz.Brain.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    figure : instance of mayavi.core.scene.Scene | list | int | None
        If None, a new figure will be created. If multiple views or a
        split view is requested, this must be a list of the appropriate
        length. If int is provided it will be used to identify the Mayavi
        figure by it's id or create a new figure with the given id.
    views : str | list
        View to use. See surfer.Brain().
    colorbar : bool
        If True, display colorbar on scene.

    Returns
    -------
    brain : Brain
        A instance of surfer.viz.Brain from PySurfer.
    """
    import surfer
    from surfer import Brain, TimeViewer

    if hemi in ['split', 'both'] and LooseVersion(surfer.__version__) < '0.4':
        raise NotImplementedError('hemi type "%s" not supported with your '
                                  'version of pysurfer. Please upgrade to '
                                  'version 0.4 or higher.' % hemi)

    try:
        import mayavi
        from mayavi import mlab
    except ImportError:
        from enthought import mayavi
        from enthought.mayavi import mlab

    # import here to avoid circular import problem
    from .source_estimate import SourceEstimate

    if not isinstance(stc, SourceEstimate):
        raise ValueError('stc has to be a surface source estimate')

    if hemi not in ['lh', 'rh', 'split', 'both']:
        raise ValueError('hemi has to be either "lh", "rh", "split", '
                         'or "both"')

    n_split = 2 if hemi == 'split' else 1
    n_views = 1 if isinstance(views, basestring) else len(views)
    if figure is not None:
        # use figure with specified id or create new figure
        if isinstance(figure, int):
            figure = mlab.figure(figure, size=(600, 600))
        # make sure it is of the correct type
        if not isinstance(figure, list):
            figure = [figure]
        if not all([isinstance(f, mayavi.core.scene.Scene) for f in figure]):
            raise TypeError('figure must be a mayavi scene or list of scenes')
        # make sure we have the right number of figures
        n_fig = len(figure)
        if not n_fig == n_split * n_views:
            raise RuntimeError('`figure` must be a list with the same '
                               'number of elements as PySurfer plots that '
                               'will be created (%s)' % n_split * n_views)

    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir)

    subject = _check_subject(stc.subject, subject, False)
    if subject is None:
        if 'SUBJECT' in os.environ:
            subject = os.environ['SUBJECT']
        else:
            raise ValueError('SUBJECT environment variable not set')

    if hemi in ['both', 'split']:
        hemis = ['lh', 'rh']
    else:
        hemis = [hemi]

    title = subject if len(hemis) > 1 else '%s - %s' % (subject, hemis[0])
    args = inspect.getargspec(Brain.__init__)[0]
    kwargs = dict(title=title, figure=figure, config_opts=config_opts,
                  subjects_dir=subjects_dir)
    if 'views' in args:
        kwargs['views'] = views
    else:
        logger.info('PySurfer does not support "views" argument, please '
                    'consider updating to a newer version (0.4 or later)')
    brain = Brain(subject, hemi, surface, **kwargs)
    for hemi in hemis:
        hemi_idx = 0 if hemi == 'lh' else 1
        if hemi_idx == 0:
            data = stc.data[:len(stc.vertno[0])]
        else:
            data = stc.data[len(stc.vertno[0]):]
        vertices = stc.vertno[hemi_idx]
        time = 1e3 * stc.times
        brain.add_data(data, colormap=colormap, vertices=vertices,
                       smoothing_steps=smoothing_steps, time=time,
                       time_label=time_label, alpha=alpha, hemi=hemi,
                       colorbar=colorbar)

        # scale colormap and set time (index) to display
        brain.scale_data_colormap(fmin=fmin, fmid=fmid, fmax=fmax,
                                  transparent=transparent)

    if time_viewer:
        TimeViewer(brain)

    return brain


def _plot_ica_panel_onpick(event, sources=None, ylims=None):
    """Onpick callback for plot_ica_panel"""

    # make sure that the swipe gesture in OS-X doesn't open many figures
    if event.mouseevent.inaxes is None or event.mouseevent.button != 1:
        return

    artist = event.artist
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        src_idx = artist._mne_src_idx
        component = artist._mne_component
        plt.plot(sources[src_idx], 'r')
        plt.ylim(ylims)
        plt.grid(linestyle='-', color='gray', linewidth=.25)
        plt.title(component)
    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers, so we print
        # it here to know what went wrong
        print err
        raise err


@verbose
def plot_ica_panel(sources, start=None, stop=None, n_components=None,
                   source_idx=None, ncol=3, nrow=10, verbose=None,
                   title=None, show=True):
    """Create panel plots of ICA sources

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
    title : str
        The figure title. If None a default is provided.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    show : bool
        If True, plot will be shown, else just the figure is returned.

    Returns
    -------
    fig : instance of pyplot.Figure
    """
    import matplotlib.pyplot as plt

    if source_idx is None:
        source_idx = np.arange(len(sources))
    else:
        source_idx = np.array(source_idx)

    for param in ['nrow', 'n_components']:
        if eval(param) is not None:
            warnings.warn('The `%s` parameter is deprecated and will be'
                          'removed in MNE-Python 0.8' % param,
                          DeprecationWarning)

    n_components = len(sources)
    sources = sources[source_idx, start:stop]
    ylims = sources.min(), sources.max()
    xlims = np.arange(sources.shape[-1])[[0, -1]]
    fig, axes = _prepare_trellis(n_components, ncol)
    if title is None:
        fig.suptitle('MEG signal decomposition'
                     ' -- %i components.' % n_components, size=16)
    elif title:
        fig.suptitle(title, size=16)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for idx, (ax, source) in enumerate(zip(axes, sources)):
        ax.grid(linestyle='-', color='gray', linewidth=.25)
        component = '[%i]' % idx

        # plot+ emebed idx and comp. name to use in callback
        line = ax.plot(source, linewidth=0.5, color='red', picker=1e9)[0]
        vars(line)['_mne_src_idx'] = idx
        vars(line)['_mne_component'] = component
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.text(0.05, .95, component, transform=ax.transAxes,
                verticalalignment='top')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    # register callback
    callback = partial(_plot_ica_panel_onpick, sources=sources, ylims=ylims)
    fig.canvas.mpl_connect('pick_event', callback)

    if show:
        plt.show()

    return fig


def plot_ica_topomap(ica, source_idx, ch_type='mag', res=500, layout=None,
                     vmax=None, cmap='RdBu_r', sensors='k,', colorbar=True,
                     show=True):
    """ Plot topographic map from ICA component.

    Parameters
    ----------
    ica : instance of mne.prerocessing.ICA
        The ica object to plot from.
    source_idx : int | array-like
        The indices of the sources to be plotted.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg'
        The channel type to plot. For 'grad', the gradiometers are collected in
        pairs and the RMS for each pair is plotted.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    vmax : scalar
        The value specfying the range of the color scale (-vmax to +vmax). If
        None, the largest absolute value in the data is used.
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    res : int
        The resolution of the topomap image (n pixels along each side).
    show : bool
        Call pyplot.show() at the end.
    """
    import matplotlib.pyplot as plt

    if np.isscalar(source_idx):
        source_idx = [source_idx]

    data = np.dot(ica.mixing_matrix_[:, source_idx].T,
                  ica.pca_components_[:ica.n_components_])

    if ica.info is None:
        raise RuntimeError('The ICA\'s measurement info is missing. Please '
                           'fit the ICA or add the corresponding info object.')

    picks, pos, merge_grads = _prepare_topo_plot(ica, ch_type, layout)
    data = np.atleast_2d(data)
    data = data[:, picks]

    # prepare data for iteration
    fig, axes = _prepare_trellis(len(data), max_col=5)

    if vmax is None:
        vrange = np.array([f(data) for f in np.min, np.max])
        vmax = max(abs(vrange))

    if merge_grads:
        from .layouts.layout import _merge_grad_data
    for ii, data_, ax in zip(source_idx, data, axes):
        data_ = _merge_grad_data(data_) if merge_grads else data_
        plot_topomap(data_.flatten(), pos, vmax=vmax, res=res, axis=ax)
        ax.set_title('IC #%03d' % ii, fontsize=12)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_frame_on(False)

    tight_layout()
    if colorbar:
        vmax_ = normalize_colors(vmin=-vmax, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=vmax_)
        sm.set_array(np.linspace(-vmax, vmax))
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(sm, cax=cax)
        cax.set_title('AU')

    if show is True:
        plt.show()

    return fig


def _prepare_topo_plot(obj, ch_type, layout):
    """"Aux Function"""
    info = copy.deepcopy(obj.info)
    if layout is None and ch_type is not 'eeg':
        from .layouts.layout import find_layout
        layout = find_layout(info['chs'])
    elif layout == 'auto':
        layout = None

    info['ch_names'] = _clean_names(info['ch_names'])
    for ii, this_ch in enumerate(info['chs']):
        this_ch['ch_name'] = info['ch_names'][ii]

    if layout is not None:
        layout = copy.deepcopy(layout)
        layout.names = _clean_names(layout.names)

    # special case for merging grad channels
    if (ch_type == 'grad' and FIFF.FIFFV_COIL_VV_PLANAR_T1 in
                    np.unique([ch['coil_type'] for ch in info['chs']])):
        from .layouts.layout import _pair_grad_sensors
        picks, pos = _pair_grad_sensors(info, layout)
        merge_grads = True
    else:
        merge_grads = False
        if ch_type == 'eeg':
            picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                               exclude='bads')
        else:
            picks = pick_types(info, meg=ch_type, ref_meg=False,
                               exclude='bads')

        if len(picks) == 0:
            raise ValueError("No channels of type %r" % ch_type)

        if layout is None:
            chs = [info['chs'][i] for i in picks]
            from .layouts.layout import _find_topomap_coords
            pos = _find_topomap_coords(chs, layout)
        else:
            pos = [layout.pos[layout.names.index(info['ch_names'][k])] for k in
                   picks]

    return picks, pos, merge_grads


def plot_image_epochs(epochs, picks=None, sigma=0.3, vmin=None,
                      vmax=None, colorbar=True, order=None, show=True,
                      units=None, scalings=None):
    """Plot Event Related Potential / Fields image

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    picks : int | array of int | None
        The indices of the channels to consider. If None, all good
        data channels are plotted.
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
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `units=dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting.
        If None, defaults to `scalings=dict(eeg=1e6, grad=1e13, mag=1e15)`

    Returns
    -------
    figs : the list of matplotlib figures
        One figure per channel displayed
    """
    units, scalings = _mutable_defaults(('units', units),
                                        ('scalings', scalings))

    import matplotlib.pyplot as plt
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')

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
        this_fig = plt.figure()
        figs.append(this_fig)

        ch_type = channel_type(epochs.info, idx)
        if not ch_type in scalings:
            # We know it's not in either scalings or units since keys match
            raise KeyError('%s type not in scalings and units' % ch_type)
        this_data *= scalings[ch_type]

        this_order = order
        if callable(order):
            this_order = order(epochs.times, this_data)

        if this_order is not None:
            this_data = this_data[this_order]

        this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

        ax1 = plt.subplot2grid((3, 10), (0, 0), colspan=9, rowspan=2)
        im = plt.imshow(this_data,
                        extent=[1e3 * epochs.times[0], 1e3 * epochs.times[-1],
                                0, len(data)],
                        aspect='auto', origin='lower',
                        vmin=vmin, vmax=vmax)
        ax2 = plt.subplot2grid((3, 10), (2, 0), colspan=9, rowspan=1)
        if colorbar:
            ax3 = plt.subplot2grid((3, 10), (0, 9), colspan=1, rowspan=3)
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
            plt.colorbar(im, cax=ax3)
            tight_layout()

    if show:
        plt.show()

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
    cmap : instance of matplotlib.pyplot.colormap | array
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
        If not None, only the n_lines strongest connections (strength=abs(con))
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
    import matplotlib.pyplot as plt
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
        node_colors = [plt.cm.spectral(i / float(n_nodes))
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
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    axes = plt.subplot(111, polar=True, axisbg=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit
    plt.ylim(0, 10)

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

        plt.text(angle_rad, 10.4, name, size=10, rotation=angle_deg,
                 rotation_mode='anchor', horizontalalignment=ha,
                 verticalalignment='center', color=textcolor)

    if title is not None:
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.75)
        plt.figtext(0.03, 0.95, title, color=textcolor, fontsize=14)
    else:
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)

    if colorbar:
        norm = normalize_colors(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax))
        ax = fig.add_axes([.92, 0.03, .015, .25])
        cb = fig.colorbar(sm, cax=ax)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        plt.setp(cb_yticks, color=textcolor)

    return fig


def plot_drop_log(drop_log, threshold=0, n_max_plot=20, subject='Unknown',
                  color=(0.9, 0.9, 0.9), width=0.8):
    """Show the channel stats based on a drop_log from Epochs

    Parameters
    ----------
    drop_log : list of lists
        Epoch drop log from Epochs.drop_log.
    threshold : float
        The percentage threshold to use to decide whether or not to
        plot. Default is zero (always plot).
    n_max_plot : int
        Maximum number of channels to show stats for.
    subject : str
        The subject name to use in the title of the plot.
    color : tuple | str
        Color to use for the bars.
    width : float
        Width of the bars.

    Returns
    -------
    perc : float
        Total percentage of epochs dropped.
    """
    if not isinstance(drop_log, list) or not isinstance(drop_log[0], list):
        raise ValueError('drop_log must be a list of lists')
    import matplotlib.pyplot as plt
    scores = Counter([ch for d in drop_log for ch in d])
    ch_names = np.array(scores.keys())
    perc = 100 * np.mean([len(d) > 0 for d in drop_log])
    if perc < threshold or len(ch_names) == 0:
        return perc
    counts = 100 * np.array(scores.values(), dtype=float) / len(drop_log)
    n_plot = min(n_max_plot, len(ch_names))
    order = np.flipud(np.argsort(counts))
    plt.figure()
    plt.title('%s: %0.1f%%' % (subject, perc))
    x = np.arange(n_plot)
    plt.bar(x, counts[order[:n_plot]], color=color, width=width)
    plt.xticks(x + width / 2.0, ch_names[order[:n_plot]], rotation=45,
               horizontalalignment='right')
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.ylabel('% of epochs rejected')
    plt.xlim((-width / 2.0, (n_plot - 1) + width * 3 / 2))
    plt.grid(True, axis='y')
    plt.show()
    return perc


def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=None,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order='type',
             show_options=False, title=None, show=True, block=False):
    """Plot raw data

    Parameters
    ----------
    raw : instance of Raw
        The raw data to plot.
    events : array | None
        Events to show with vertical bars.
    duration : float
        Time window (sec) to plot in a given time.
    start : float
        Initial time to show (can be changed dynamically once plotted).
    n_channels : int
        Number of channels to plot at once.
    bgcolor : color object
        Color of the background.
    color : dict | color object | None
        Color for the data traces. If None, defaults to:
        `dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='r', emg='k',
             ref_meg='steelblue', misc='k', stim='k', resp='k', chpi='k')`
    bad_color : color object
        Color to make bad channels.
    event_color : color object
        Color to use for events.
    scalings : dict | None
        Scale factors for the traces. If None, defaults to:
        `dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4, emg=1e-3,
             ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
    remove_dc : bool
        If True remove DC component when plotting data.
    order : 'type' | 'original' | array
        Order in which to plot data. 'type' groups by channel type,
        'original' plots in the order of ch_names, array gives the
        indices to use in plotting.
    show_options : bool
        If True, a dialog for options related to projecion is shown.
    title : str | None
        The title of the window. If None, and either the filename of the
        raw object or '<unknown>' will be displayed as title.
    show : bool
        Show figure if True
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for setting bad channels on the fly by clicking on a line.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Raw traces.

    Notes
    -----
    The arrow keys (up/down/left/right) can typically be used to navigate
    between channels and time ranges, but this depends on the backend
    matplotlib is configured to use (e.g., mpl.use('TkAgg') should work).
    To mark or un-mark a channel as bad, click on the rather flat segments
    of a channel's time series. The changes will be reflected immediately
    in the raw object's ``raw.info['bads']`` entry.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    color, scalings = _mutable_defaults(('color', color),
                                        ('scalings_plot_raw', scalings))

    # make a copy of info, remove projection (for now)
    info = copy.deepcopy(raw.info)
    projs = info['projs']
    info['projs'] = []
    n_times = raw.n_times

    # allow for raw objects without filename, e.g., ICA
    if title is None:
        title = raw.info.get('filenames', None)  # should return a list
        if not title:  # empty list or absent key
            title = '<unknown>'
        else:
            if len(title) > 1:
                title = '<unknown>'
            else:
                title = title[0]
    elif not isinstance(title, basestring):
        raise TypeError('title must be None or a string')
    if len(title) > 60:
        title = '...' + title[-60:]
    if len(raw.info['filenames']) > 1:
        title += ' ... (+ %d more) ' % (len(raw.info['filenames']) - 1)
    if events is not None:
        events = events[:, 0].astype(float) - raw.first_samp
        events /= info['sfreq']

    # reorganize the data in plotting order
    inds = list()
    types = list()
    for t in ['grad', 'mag']:
        inds += [pick_types(info, meg=t, ref_meg=False, exclude=[])]
        types += [t] * len(inds[-1])
    pick_args = dict(meg=False, exclude=[])
    for t in ['eeg', 'eog', 'ecg', 'emg', 'ref_meg', 'stim', 'resp',
              'misc', 'chpi', 'syst', 'ias', 'exci']:
        pick_args[t] = True
        inds += [pick_types(raw.info, **pick_args)]
        types += [t] * len(inds[-1])
        pick_args[t] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(info['ch_names']):
        raise RuntimeError('Some channels not classified, please report '
                           'this problem')

    # put them back to original or modified order for natral plotting
    reord = np.argsort(inds)
    types = [types[ri] for ri in reord]
    if isinstance(order, str):
        if order == 'original':
            inds = inds[reord]
        elif order != 'type':
            raise ValueError('Unknown order type %s' % order)
    elif isinstance(order, np.ndarray):
        if not np.array_equal(np.sort(order),
                              np.arange(len(info['ch_names']))):
            raise ValueError('order, if array, must have integers from '
                             '0 to n_channels - 1')
        # put back to original order first, then use new order
        inds = inds[reord][order]

    # set up projection and data parameters
    params = dict(raw=raw, ch_start=0, t_start=start, duration=duration,
                  info=info, projs=projs, remove_dc=remove_dc,
                  n_channels=n_channels, scalings=scalings, types=types,
                  n_times=n_times, events=events)

    # set up plotting
    fig = figure_nobar(facecolor=bgcolor)
    fig.canvas.set_window_title('mne_browse_raw')
    size = get_config('MNE_BROWSE_RAW_SIZE')
    if size is not None:
        size = size.split(',')
        size = tuple([float(s) for s in size])
        # have to try/catch when there's no toolbar
        try:
            fig.set_size_inches(size, forward=True)
        except Exception:
            pass
    ax = plt.subplot2grid((10, 10), (0, 0), colspan=9, rowspan=9)
    ax.set_title(title, fontsize=12)
    ax_hscroll = plt.subplot2grid((10, 10), (9, 0), colspan=9)
    ax_hscroll.get_yaxis().set_visible(False)
    ax_hscroll.set_xlabel('Time (s)')
    ax_vscroll = plt.subplot2grid((10, 10), (0, 9), rowspan=9)
    ax_vscroll.set_axis_off()
    ax_button = plt.subplot2grid((10, 10), (9, 9))
    # store these so they can be fixed on resize
    params['fig'] = fig
    params['ax'] = ax
    params['ax_hscroll'] = ax_hscroll
    params['ax_vscroll'] = ax_vscroll
    params['ax_button'] = ax_button

    # populate vertical and horizontal scrollbars
    for ci in xrange(len(info['ch_names'])):
        this_color = (bad_color if info['ch_names'][inds[ci]] in info['bads']
                      else color)
        if isinstance(this_color, dict):
            this_color = this_color[types[inds[ci]]]
        ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                   facecolor=this_color,
                                                   edgecolor=this_color))
    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       facecolor='w', edgecolor='w')
    ax_vscroll.add_patch(vsel_patch)
    params['vsel_patch'] = vsel_patch
    hsel_patch = mpl.patches.Rectangle((start, 0), duration, 1, color='k',
                                       edgecolor=None, alpha=0.5)
    ax_hscroll.add_patch(hsel_patch)
    params['hsel_patch'] = hsel_patch
    ax_hscroll.set_xlim(0, n_times / float(info['sfreq']))
    n_ch = len(info['ch_names'])
    ax_vscroll.set_ylim(n_ch, 0)
    ax_vscroll.set_title('Ch.')

    # make shells for plotting traces
    offsets = np.arange(n_channels) * 2 + 1
    ax.set_yticks(offsets)
    ax.set_ylim([n_channels * 2 + 1, 0])
    # plot event_line first so it's in the back
    event_line = ax.plot([np.nan], color=event_color)[0]
    lines = [ax.plot([np.nan])[0] for _ in xrange(n_ch)]
    ax.set_yticklabels(['X' * max([len(ch) for ch in info['ch_names']])])

    params['plot_fun'] = partial(_plot_traces, params=params, inds=inds,
                                 color=color, bad_color=bad_color, lines=lines,
                                 event_line=event_line, offsets=offsets)

    # set up callbacks
    opt_button = mpl.widgets.Button(ax_button, 'Opt')
    callback_option = partial(_toggle_options, params=params)
    opt_button.on_clicked(callback_option)
    callback_key = partial(_plot_raw_onkey, params=params)
    fig.canvas.mpl_connect('key_press_event', callback_key)
    callback_pick = partial(_mouse_click, params=params)
    fig.canvas.mpl_connect('button_press_event', callback_pick)
    callback_resize = partial(_helper_resize, params=params)
    fig.canvas.mpl_connect('resize_event', callback_resize)

    # As here code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_raw_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    params['callback_key'] = callback_key
    # have to store this, or it could get garbage-collected
    params['opt_button'] = opt_button

    # do initial plots
    callback_proj('none')
    _layout_raw(params)

    # deal with projectors
    params['fig_opts'] = None
    if show_options is True:
        _toggle_options(None, params)

    if show:
        plt.show(block=block)
    
    return fig


def _toggle_options(event, params):
    """Toggle options (projectors) dialog"""
    import matplotlib.pyplot as plt
    if len(params['projs']) > 0:
        if params['fig_opts'] is None:
            _draw_proj_checkbox(event, params, draw_current_state=False)
        else:
            # turn off options dialog
            plt.close(params['fig_opts'])
            del params['proj_checks']
            params['fig_opts'] = None


def _toggle_proj(event, params):
    """Operation to perform when proj boxes clicked"""
    # read options if possible
    if 'proj_checks' in params:
        bools = [x[0].get_visible() for x in params['proj_checks'].lines]
        for bi, (b, p) in enumerate(zip(bools, params['projs'])):
            # see if they tried to deactivate an active one
            if not b and p['active']:
                bools[bi] = True
    else:
        bools = [True] * len(params['projs'])

    compute_proj = False
    if not 'proj_bools' in params:
        compute_proj = True
    elif not np.array_equal(bools, params['proj_bools']):
        compute_proj = True

    # if projectors changed, update plots
    if compute_proj is True:
        params['plot_update_proj_callback'](params, bools)


def _plot_update_raw_proj(params, bools):
    """Helper only needs to be called when proj is changed"""
    inds = np.where(bools)[0]
    params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                               for ii in inds]
    params['proj_bools'] = bools
    params['projector'], _ = setup_proj(params['info'], add_eeg_ref=False,
                                        verbose=False)
    _update_raw_data(params)
    params['plot_fun']()


def _update_raw_data(params):
    """Helper only needs to be called when time or proj is changed"""
    start = params['t_start']
    stop = params['raw'].time_as_index(start + params['duration'])[0]
    start = params['raw'].time_as_index(start)[0]
    data, times = params['raw'][:, start:stop]
    if params['projector'] is not None:
        data = np.dot(params['projector'], data)
    # remove DC
    if params['remove_dc'] is True:
        data -= np.mean(data, axis=1)[:, np.newaxis]
    # scale
    for di in xrange(data.shape[0]):
        data[di] /= params['scalings'][params['types'][di]]
        # stim channels should be hard limited
        if params['types'][di] == 'stim':
            data[di] = np.minimum(data[di], 1.0)
    params['data'] = data
    params['times'] = times


def _layout_raw(params):
    """Set raw figure layout"""
    s = params['fig'].get_size_inches()
    scroll_width = 0.33
    hscroll_dist = 0.33
    vscroll_dist = 0.1
    l_border = 1.2
    r_border = 0.1
    t_border = 0.33
    b_border = 0.5

    # only bother trying to reset layout if it's reasonable to do so
    if s[0] < 2 * scroll_width or s[1] < 2 * scroll_width + hscroll_dist:
        return

    # convert to relative units
    scroll_width_x = scroll_width / s[0]
    scroll_width_y = scroll_width / s[1]
    vscroll_dist /= s[0]
    hscroll_dist /= s[1]
    l_border /= s[0]
    r_border /= s[0]
    t_border /= s[1]
    b_border /= s[1]
    # main axis (traces)
    ax_width = 1.0 - scroll_width_x - l_border - r_border - vscroll_dist
    ax_y = hscroll_dist + scroll_width_y + b_border
    ax_height = 1.0 - ax_y - t_border
    params['ax'].set_position([l_border, ax_y, ax_width, ax_height])
    # vscroll (channels)
    pos = [ax_width + l_border + vscroll_dist, ax_y,
           scroll_width_x, ax_height]
    params['ax_vscroll'].set_position(pos)
    # hscroll (time)
    pos = [l_border, b_border, ax_width, scroll_width_y]
    params['ax_hscroll'].set_position(pos)
    # options button
    pos = [l_border + ax_width + vscroll_dist, b_border,
           scroll_width_x, scroll_width_y]
    params['ax_button'].set_position(pos)
    params['fig'].canvas.draw()


def _helper_resize(event, params):
    """Helper for resizing"""
    size = ','.join([str(s) for s in params['fig'].get_size_inches()])
    set_config('MNE_BROWSE_RAW_SIZE', size)
    _layout_raw(params)


def _pick_bad_channels(event, params):
    """Helper for selecting / dropping bad channels onpick"""
    bads = params['raw'].info['bads']
    # trade-off, avoid selecting more than one channel when drifts are present
    # however for clean data don't click on peaks but on flat segments
    f = lambda x, y: y(np.mean(x), x.std() * 2)
    for l in event.inaxes.lines:
        ydata = l.get_ydata()
        if not isinstance(ydata, list) and not np.isnan(ydata).any():
            ymin, ymax = f(ydata, np.subtract), f(ydata, np.add)
            if ymin <= event.ydata <= ymax:
                this_chan = vars(l)['ch_name']
                if this_chan in params['raw'].ch_names:
                    if this_chan not in bads:
                        bads.append(this_chan)
                        l.set_color(params['bad_color'])
                    else:
                        bads.pop(bads.index(this_chan))
                        l.set_color(vars(l)['def-color'])
                event.canvas.draw()
                break
    # update deep-copied info to persistently draw bads
    params['info']['bads'] = bads


def _mouse_click(event, params):
    """Vertical select callback"""
    if event.inaxes is None or event.button != 1:
        return
    plot_fun = params['plot_fun']
    # vertical scrollbar changed
    if event.inaxes == params['ax_vscroll']:
        ch_start = max(int(event.ydata) - params['n_channels'] // 2, 0)
        if params['ch_start'] != ch_start:
            params['ch_start'] = ch_start
            plot_fun()
    # horizontal scrollbar changed
    elif event.inaxes == params['ax_hscroll']:
        _plot_raw_time(event.xdata - params['duration'] / 2, params)

    elif event.inaxes == params['ax']:
        _pick_bad_channels(event, params)


def _plot_raw_time(value, params):
    """Deal with changed time value"""
    info = params['info']
    max_times = params['n_times'] / float(info['sfreq']) - params['duration']
    if value > max_times:
        value = params['n_times'] / info['sfreq'] - params['duration']
    if value < 0:
        value = 0
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)
        _update_raw_data(params)
        params['plot_fun']()


def _plot_raw_onkey(event, params):
    """Interpret key presses"""
    import matplotlib.pyplot as plt
    # check for initial plot
    plot_fun = params['plot_fun']
    if event is None:
        plot_fun()
        return

    # quit event
    if event.key == 'escape':
        plt.close(params['fig'])
        return

    # change plotting params
    ch_changed = False
    if event.key == 'down':
        params['ch_start'] += params['n_channels']
        ch_changed = True
    elif event.key == 'up':
        params['ch_start'] -= params['n_channels']
        ch_changed = True
    elif event.key == 'right':
        _plot_raw_time(params['t_start'] + params['duration'], params)
        return
    elif event.key == 'left':
        _plot_raw_time(params['t_start'] - params['duration'], params)
        return
    elif event.key in ['o', 'p']:
        _toggle_options(None, params)
        return

    # deal with plotting changes
    if ch_changed is True:
        if params['ch_start'] >= len(params['info']['ch_names']):
            params['ch_start'] = 0
        elif params['ch_start'] < 0:
            # wrap to end
            rem = len(params['info']['ch_names']) % params['n_channels']
            params['ch_start'] = len(params['info']['ch_names'])
            params['ch_start'] -= rem if rem != 0 else params['n_channels']

    if ch_changed:
        plot_fun()


def _plot_traces(params, inds, color, bad_color, lines, event_line, offsets):
    """Helper for plotting raw"""

    info = params['info']
    n_channels = params['n_channels']
    params['bad_color'] = bad_color
    # do the plotting
    tick_list = []
    for ii in xrange(n_channels):
        ch_ind = ii + params['ch_start']
        # let's be generous here and allow users to pass
        # n_channels per view >= the number of traces available
        if ii >= len(lines):
            break
        elif ch_ind < len(info['ch_names']):
            # scale to fit
            ch_name = info['ch_names'][inds[ch_ind]]
            tick_list += [ch_name]
            offset = offsets[ii]

            # do NOT operate in-place lest this get screwed up
            this_data = params['data'][inds[ch_ind]]
            this_color = bad_color if ch_name in info['bads'] else color
            if isinstance(this_color, dict):
                this_color = this_color[params['types'][inds[ch_ind]]]

            # subtraction here gets corect orientation for flipped ylim
            lines[ii].set_ydata(offset - this_data)
            lines[ii].set_xdata(params['times'])
            lines[ii].set_color(this_color)
            vars(lines[ii])['ch_name'] = ch_name
            vars(lines[ii])['def-color'] = color[params['types'][inds[ch_ind]]]
        else:
            # "remove" lines
            lines[ii].set_xdata([])
            lines[ii].set_ydata([])
    # deal with event lines
    if params['events'] is not None:
        t = params['events']
        t = t[np.where(np.logical_and(t >= params['times'][0],
                       t <= params['times'][-1]))[0]]
        if len(t) > 0:
            xs = list()
            ys = list()
            for tt in t:
                xs += [tt, tt, np.nan]
                ys += [0, 2 * n_channels + 1, np.nan]
            event_line.set_xdata(xs)
            event_line.set_ydata(ys)
        else:
            event_line.set_xdata([])
            event_line.set_ydata([])
    # finalize plot
    params['ax'].set_xlim(params['times'][0],
                          params['times'][0] + params['duration'], False)
    params['ax'].set_yticklabels(tick_list)
    params['vsel_patch'].set_y(params['ch_start'])
    params['fig'].canvas.draw()


def figure_nobar(*args, **kwargs):
    """Make matplotlib figure with no toolbar"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    old_val = mpl.rcParams['toolbar']
    try:
        mpl.rcParams['toolbar'] = 'none'
        fig = plt.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        for key in fig.canvas.callbacks.callbacks['key_press_event'].keys():
            fig.canvas.callbacks.disconnect(key)
    except Exception as ex:
        raise ex
    finally:
        mpl.rcParams['toolbar'] = old_val
    return fig


@verbose
def plot_raw_psds(raw, tmin=0.0, tmax=60.0, fmin=0, fmax=np.inf,
                  proj=False, n_fft=2048, picks=None, ax=None, color='black',
                  area_mode='std', area_alpha=0.33, n_jobs=1, verbose=None):
    """Plot the power spectral density across channels

    Parameters
    ----------
    raw : instance of fiff.Raw
        The raw instance to use.
    tmin : float
        Start time for calculations.
    tmax : float
        End time for calculations.
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    proj : bool
        Apply projection.
    n_fft : int
        Number of points to use in Welch FFT calculations.
    picks : list | None
        List of channels to use. Cannot be None if `ax` is supplied. If both
        `picks` and `ax` are None, separate subplots will be created for
        each standard channel type (`mag`, `grad`, and `eeg`).
    ax : instance of matplotlib Axes | None
        Axes to plot into. If None, axes will be created.
    color : str | tuple
        A matplotlib-compatible color to use.
    area_mode : str | None
        Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
        will be plotted. If 'range', the min and max (across channels) will be
        plotted. Bad channels will be excluded from these calculations.
        If None, no area will be plotted.
    area_alpha : float
        Alpha for the area.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    import matplotlib.pyplot as plt
    if area_mode not in [None, 'std', 'range']:
        raise ValueError('"area_mode" must be "std", "range", or None')
    if picks is None:
        if ax is not None:
            raise ValueError('If "ax" is not supplied (None), then "picks" '
                             'must also be supplied')
        megs = ['mag', 'grad', False]
        eegs = [False, False, True]
        names = ['Magnetometers', 'Gradiometers', 'EEG']
        picks_list = list()
        titles_list = list()
        for meg, eeg, name in zip(megs, eegs, names):
            picks = pick_types(raw.info, meg=meg, eeg=eeg, ref_meg=False)
            if len(picks) > 0:
                picks_list.append(picks)
                titles_list.append(name)
        if len(picks_list) == 0:
            raise RuntimeError('No MEG or EEG channels found')
    else:
        picks_list = [picks]
        titles_list = ['Selected channels']
        ax_list = [ax]

    make_label = False
    if ax is None:
        plt.figure()
        ax_list = list()
        for ii in range(len(picks_list)):
            # Make x-axes change together
            if ii > 0:
                ax_list.append(plt.subplot(len(picks_list), 1, ii + 1,
                                           sharex=ax_list[0]))
            else:
                ax_list.append(plt.subplot(len(picks_list), 1, ii + 1))
        make_label = True

    for ii, (picks, title, ax) in enumerate(zip(picks_list, titles_list,
                                                ax_list)):
        psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                                      fmin=fmin, fmax=fmax, NFFT=n_fft,
                                      n_jobs=n_jobs, plot=False, proj=proj)

        # Convert PSDs to dB
        psds = 10 * np.log10(psds)
        psd_mean = np.mean(psds, axis=0)
        if area_mode == 'std':
            psd_std = np.std(psds, axis=0)
            hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
        elif area_mode == 'range':
            hyp_limits = (np.min(psds, axis=0), np.max(psds, axis=0))
        else:  # area_mode is None
            hyp_limits = None

        ax.plot(freqs, psd_mean, color=color)
        if hyp_limits is not None:
            ax.fill_between(freqs, hyp_limits[0], y2=hyp_limits[1],
                            color=color, alpha=area_alpha)
        if make_label:
            if ii == len(picks_list) - 1:
                ax.set_xlabel('Freq (Hz)')
            if ii == len(picks_list) / 2:
                ax.set_ylabel('Power Spectral Density (dB/Hz)')
            ax.set_title(title)
            ax.set_xlim(freqs[0], freqs[-1])
    if make_label:
        tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
    plt.show()


@verbose
def compare_fiff(fname_1, fname_2, fname_out=None, show=True, indent='    ',
                 read_limit=np.inf, max_str=30, verbose=None):
    """Compare the contents of two fiff files using diff and show_fiff

    Parameters
    ----------
    fname_1 : str
        First file to compare.
    fname_2 : str
        Second file to compare.
    fname_out : str | None
        Filename to store the resulting diff. If None, a temporary
        file will be created.
    show : bool
        If True, show the resulting diff in a new tab in a web browser.
    indent : str
        How to indent the lines.
    read_limit : int
        Max number of bytes of data to read from a tag. Can be np.inf
        to always read all data (helps test read completion).
    max_str : int
        Max number of characters of string representation to print for
        each tag's data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fname_out : str
        The filename used for storing the diff. Could be useful for
        when a temporary file is used.
    """
    file_1 = show_fiff(fname_1, output=list, indent=indent,
                       read_limit=read_limit, max_str=max_str)
    file_2 = show_fiff(fname_2, output=list, indent=indent,
                       read_limit=read_limit, max_str=max_str)
    diff = difflib.HtmlDiff().make_file(file_1, file_2, fname_1, fname_2)
    if fname_out is not None:
        f = open(fname_out, 'w')
    else:
        f = tempfile.NamedTemporaryFile('w', delete=False)
        fname_out = f.name
    with f as fid:
        fid.write(diff)
    if show is True:
        webbrowser.open_new_tab(fname_out)
    return fname_out


def _prepare_trellis(n_cells, max_col):
    """Aux function
    """
    import matplotlib.pyplot as plt
    if n_cells == 1:
        nrow = ncol = 1
    elif n_cells <= max_col:
        nrow, ncol = 1, n_cells
    else:
        nrow, ncol = int(math.ceil(n_cells / float(max_col))), max_col

    fig, axes = plt.subplots(nrow, ncol)
    axes = [axes] if ncol == nrow == 1 else axes.flatten()
    for ax in axes[n_cells:]:  # hide unused axes
        ax.set_visible(False)
    return fig, axes


def _plot_epochs_get_data(epochs, epoch_idx, n_channels, times, picks,
                          scalings, types):
    """Aux function
    """
    data = np.zeros((len(epoch_idx), n_channels, len(times)))
    for ii, epoch in enumerate(epochs.get_data()[epoch_idx][:, picks]):
        for jj, (this_type, this_channel) in enumerate(zip(types, epoch)):
            data[ii, jj] = this_channel / scalings[this_type]
    return data


def _draw_epochs_axes(epoch_idx, good_ch_idx, bad_ch_idx, data, times, axes,
                      title_str, axes_handler):
    """Aux functioin"""
    this = axes_handler[0]
    for ii, data_, ax in zip(epoch_idx, data, axes):
        [l.set_data(times, d) for l, d in zip(ax.lines, data_[good_ch_idx])]
        if bad_ch_idx is not None:
            bad_lines = [ax.lines[k] for k in bad_ch_idx]
            [l.set_data(times, d) for l, d in zip(bad_lines,
                                                  data_[bad_ch_idx])]
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks([])
        ax.set_xticks([])
        if vars(ax)[this]['reject'] is True:
            #  memorizing reject
            [l.set_color((0.8, 0.8, 0.8)) for l in ax.lines]
            ax.get_figure().canvas.draw()
        else:
            #  forgetting previous reject
            for k in axes_handler:
                if k == this:
                    continue
                if vars(ax).get(k, {}).get('reject', None) is True:
                    [l.set_color('k') for l in ax.lines[:len(good_ch_idx)]]
                    if bad_ch_idx is not None:
                        [l.set_color('r') for l in ax.lines[-len(bad_ch_idx):]]
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Aux function"""
    import matplotlib.pyplot as plt
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            p['epochs'].drop_epochs(p['reject_idx'])
        plt.close(p['fig'])
        plt.close(event.inaxes.get_figure())

    if here is not None:
        p['idx_handler'].rotate(here)
        p['axes_handler'].rotate(here)
        this_idx = p['idx_handler'][0]
        data = _plot_epochs_get_data(p['epochs'], this_idx, p['n_channels'],
                                     p['times'], p['picks'], p['scalings'],
                                     p['types'])
        _draw_epochs_axes(this_idx, p['good_ch_idx'], p['bad_ch_idx'], data,
                          p['times'], p['axes'], p['title_str'],
                          p['axes_handler'])
            # XXX don't ask me why
        p['axes'][0].get_figure().canvas.draw()


def _epochs_axes_onclick(event, params):
    """Aux function"""
    reject_color = (0.8, 0.8, 0.8)
    ax = event.inaxes
    p = params
    here = vars(ax)[p['axes_handler'][0]]
    if here.get('reject', None) is False:
        idx = here['idx']
        if idx not in p['reject_idx']:
            p['reject_idx'].append(idx)
            [l.set_color(reject_color) for l in ax.lines]
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            good_lines = [ax.lines[k] for k in p['good_ch_idx']]
            [l.set_color('k') for l in good_lines]
            if p['bad_ch_idx'] is not None:
                bad_lines = ax.lines[-len(p['bad_ch_idx']):]
                [l.set_color('r') for l in bad_lines]
            here['reject'] = False
    ax.get_figure().canvas.draw()


def plot_epochs(epochs, epoch_idx=None, picks=None, scalings=None,
                title_str='#%003i', show=True, block=False):
    """ Visualize single trials using Trellis plot.

    Parameters
    ----------

    epochs : instance of Epochs
        The epochs object
    epoch_idx : array-like | int | None
        The epochs to visualize. If None, the first 20 epochs are shown.
        Defaults to None.
    picks : array-like | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    scalings : dict | None
        Scale factors for the traces. If None, defaults to:
        `dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4, emg=1e-3,
             ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
    title_str : None | str
        The string formatting to use for axes titles. If None, no titles
        will be shown. Defaults expand to ``#001, #002, ...``
    show : bool
        Whether to show the figure or not.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on a
        sub plot.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    scalings = _mutable_defaults(('scalings_plot_raw', None))[0]
    if np.isscalar(epoch_idx):
        epoch_idx = [epoch_idx]
    if epoch_idx is None:
        n_events = len(epochs.events)
        epoch_idx = range(n_events)
    else:
        n_events = len(epoch_idx)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, 20))

    if picks is None:
        if any('ICA' in k for k in epochs.ch_names):
            picks = pick_types(epochs.info, misc=True, ref_meg=False,
                               exclude=[])
        else:
            picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                               exclude=[])
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')
    times = epochs.times * 1e3
    n_channels = len(picks)
    types = [channel_type(epochs.info, idx) for idx in
             picks]

    # preallocation needed for min / max scaling
    data = _plot_epochs_get_data(epochs, idx_handler[0], n_channels,
                                 times, picks, scalings, types)
    n_events = len(epochs.events)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, 20))
    # handle bads
    bad_ch_idx = None
    ch_names = epochs.ch_names
    bads = epochs.info['bads']
    if any([ch_names[k] in bads for k in picks]):
        ch_picked = [k for k in ch_names if ch_names.index(k) in picks]
        bad_ch_idx = [ch_picked.index(k) for k in bads if k in ch_names]
        good_ch_idx = [p for p in picks if p not in bad_ch_idx]
    else:
        good_ch_idx = np.arange(n_channels)

    fig, axes = _prepare_trellis(len(data), max_col=5)
    axes_handler = deque(range(len(idx_handler)))
    for ii, data_, ax in zip(idx_handler[0], data, axes):
        ax.plot(times, data_[good_ch_idx].T, color='k')
        if bad_ch_idx is not None:
            ax.plot(times, data_[bad_ch_idx].T, color='r')
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks([])
        ax.set_xticks([])
        vars(ax)[axes_handler[0]] = {'idx': ii, 'reject': False}

    # initialize memory
    for this_view, this_inds in zip(axes_handler, idx_handler):
        for ii, ax in zip(this_inds, axes):
            vars(ax)[this_view] = {'idx': ii, 'reject': False}

    tight_layout()
    navigation = figure_nobar(figsize=(3, 1.5))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    params = {
        'fig': fig,
        'idx_handler': idx_handler,
        'epochs': epochs,
        'n_channels': n_channels,
        'picks': picks,
        'times': times,
        'scalings': scalings,
        'types': types,
        'good_ch_idx': good_ch_idx,
        'bad_ch_idx': bad_ch_idx,
        'axes': axes,
        'back': mpl.widgets.Button(ax1, 'back'),
        'next': mpl.widgets.Button(ax2, 'next'),
        'reject-quit': mpl.widgets.Button(ax3, 'reject-quit'),
        'title_str': title_str,
        'reject_idx': [],
        'axes_handler': axes_handler
    }
    fig.canvas.mpl_connect('button_press_event',
                           partial(_epochs_axes_onclick, params=params))
    navigation.canvas.mpl_connect('button_press_event',
                                  partial(_epochs_navigation_onclick,
                                          params=params))
    if show is True:
        plt.show(block=block)
    return fig


def plot_source_spectrogram(stcs, freq_bins, source_index=None, colorbar=False,
                            show=True):
    """Plot source power in time-freqency grid

    Parameters
    ----------
    stcs : list of SourceEstimate
        Source power for consecutive time windows, one SourceEstimate object
        should be provided for each frequency bin.
    freq_bins : list of tuples of float
        Start and end points of frequency bins of interest.
    source_index : int | None
        Index of source for which the spectrogram will be plotted. If None,
        the source with the largest activation will be selected.
    colorbar : bool
        If true, a colorbar will be added to the plot.
    show : bool
        Show figure if True.
    """
    import matplotlib.pyplot as plt

    # Gathering results for each time window
    source_power = np.array([stc.data for stc in stcs])

    # Finding the source with maximum source power
    if source_index is None:
        source_index = np.unravel_index(source_power.argmax(),
                                        source_power.shape)[1]

    # Preparing time-frequency cell boundaries for plotting
    stc = stcs[0]
    time_bounds = np.append(stc.times, stc.times[-1] + stc.tstep)
    freq_bounds = sorted(set(np.ravel(freq_bins)))
    freq_ticks = deepcopy(freq_bounds)

    # If there is a gap in the frequency bins record its locations so that it
    # can be covered with a gray horizontal bar
    gap_bounds = []
    for i in range(len(freq_bins) - 1):
        lower_bound = freq_bins[i][1]
        upper_bound = freq_bins[i+1][0]
        if lower_bound != upper_bound:
            freq_bounds.remove(lower_bound)
            gap_bounds.append((lower_bound, upper_bound))

    # Preparing time-frequency grid for plotting
    time_grid, freq_grid = np.meshgrid(time_bounds, freq_bounds)

    # Plotting the results
    plt.figure(figsize=(9, 6))
    plt.pcolor(time_grid, freq_grid, source_power[:, source_index, :],
               cmap=plt.cm.jet)
    ax = plt.gca()

    plt.title('Time-frequency source power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    time_tick_labels = [str(np.round(t, 2)) for t in time_bounds]
    n_skip = 1 + len(time_bounds) // 10
    for i in range(len(time_bounds)):
        if i % n_skip != 0:
            time_tick_labels[i] = ''

    ax.set_xticks(time_bounds)
    ax.set_xticklabels(time_tick_labels)
    plt.xlim(time_bounds[0], time_bounds[-1])
    plt.yscale('log')
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels([np.round(freq, 2) for freq in freq_ticks])
    plt.ylim(freq_bounds[0], freq_bounds[-1])

    plt.grid(True, ls='-')
    if colorbar:
        plt.colorbar()
    tight_layout()

    # Covering frequency gaps with horizontal bars
    for lower_bound, upper_bound in gap_bounds:
        plt.barh(lower_bound, time_bounds[-1] - time_bounds[0], upper_bound -
                 lower_bound, time_bounds[0], color='lightgray')

    if show:
        plt.show()
