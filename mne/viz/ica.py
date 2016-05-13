"""Functions to plot ICA specific data (besides topographies)
"""
from __future__ import print_function

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: Simplified BSD

from functools import partial

import numpy as np

from .utils import (tight_layout, _prepare_trellis, _select_bads,
                    _layout_figure, _plot_raw_onscroll, _mouse_click,
                    _helper_raw_resize, _plot_raw_onkey, plt_show)
from .raw import _prepare_mne_browse_raw, _plot_raw_traces
from .epochs import _prepare_mne_browse_epochs
from .evoked import _butterfly_on_button_press, _butterfly_onpick
from .topomap import _prepare_topo_plot, plot_topomap, _hide_frame
from ..utils import warn
from ..defaults import _handle_default
from ..io.meas_info import create_info
from ..io.pick import pick_types
from ..externals.six import string_types


def _ica_plot_sources_onpick_(event, sources=None, ylims=None):
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
        plt.plot(sources[src_idx], 'r' if artist._mne_is_bad else 'k')
        plt.ylim(ylims)
        plt.grid(linestyle='-', color='gray', linewidth=.25)
        plt.title('ICA #%i' % component)
    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers, so we print
        # it here to know what went wrong
        print(err)
        raise err


def plot_ica_sources(ica, inst, picks=None, exclude=None, start=None,
                     stop=None, show=True, title=None, block=False):
    """Plot estimated latent sources given the unmixing matrix.

    Typical usecases:

    1. plot evolution of latent sources over time based on (Raw input)
    2. plot latent source around event related time windows (Epochs input)
    3. plot time-locking in ICA space (Evoked input)


    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    inst : instance of mne.io.Raw, mne.Epochs, mne.Evoked
        The object to plot the sources from.
    picks : int | array_like of int | None.
        The components to be displayed. If None, plot will show the
        sources in the order as fitted.
    exclude : array_like of int
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    start : int
        X-axis start index. If None, from the beginning.
    stop : int
        X-axis stop index. If None, next 20 are shown, in case of evoked to the
        end.
    show : bool
        Show figure if True.
    title : str | None
        The figure title. If None a default is provided.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for interactive selection of components in raw and epoch
        plotter. For evoked, this parameter has no effect. Defaults to False.

    Returns
    -------
    fig : instance of pyplot.Figure
        The figure.

    Notes
    -----
    For raw and epoch instances, it is possible to select components for
    exclusion by clicking on the line. The selected components are added to
    ``ica.exclude`` on close.

    .. versionadded:: 0.10.0
    """

    from ..io.base import _BaseRaw
    from ..evoked import Evoked
    from ..epochs import _BaseEpochs

    if exclude is None:
        exclude = ica.exclude
    elif len(ica.exclude) > 0:
        exclude = np.union1d(ica.exclude, exclude)
    if isinstance(inst, _BaseRaw):
        fig = _plot_sources_raw(ica, inst, picks, exclude, start=start,
                                stop=stop, show=show, title=title,
                                block=block)
    elif isinstance(inst, _BaseEpochs):
        fig = _plot_sources_epochs(ica, inst, picks, exclude, start=start,
                                   stop=stop, show=show, title=title,
                                   block=block)
    elif isinstance(inst, Evoked):
        sources = ica.get_sources(inst)
        if start is not None or stop is not None:
            inst = inst.copy().crop(start, stop)
        fig = _plot_ica_sources_evoked(
            evoked=sources, picks=picks, exclude=exclude, title=title,
            labels=getattr(ica, 'labels_', None), show=show)
    else:
        raise ValueError('Data input must be of Raw or Epochs type')

    return fig


def _plot_ica_grid(sources, start, stop,
                   source_idx, ncol, exclude,
                   title, show):
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
    title : str
        The figure title. If None a default is provided.
    show : bool
        If True, all open plots will be shown.
    """
    import matplotlib.pyplot as plt

    if source_idx is None:
        source_idx = np.arange(len(sources))
    elif isinstance(source_idx, list):
        source_idx = np.array(source_idx)
    if exclude is None:
        exclude = []

    n_components = len(sources)
    ylims = sources.min(), sources.max()
    xlims = np.arange(sources.shape[-1])[[0, -1]]
    fig, axes = _prepare_trellis(n_components, ncol)
    if title is None:
        fig.suptitle('Reconstructed latent sources', size=16)
    elif title:
        fig.suptitle(title, size=16)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    my_iter = enumerate(zip(source_idx, axes, sources))
    for i_source, (i_selection, ax, source) in my_iter:
        component = '[%i]' % i_selection
        # plot+ emebed idx and comp. name to use in callback
        color = 'r' if i_selection in exclude else 'k'
        line = ax.plot(source, linewidth=0.5, color=color, picker=1e9)[0]
        vars(line)['_mne_src_idx'] = i_source
        vars(line)['_mne_component'] = i_selection
        vars(line)['_mne_is_bad'] = i_selection in exclude
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.text(0.05, .95, component, transform=ax.transAxes,
                verticalalignment='top')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    # register callback
    callback = partial(_ica_plot_sources_onpick_, sources=sources, ylims=ylims)
    fig.canvas.mpl_connect('pick_event', callback)
    plt_show(show)
    return fig


def _plot_ica_sources_evoked(evoked, picks, exclude, title, show, labels=None):
    """Plot average over epochs in ICA space

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The Evoked to be used.
    picks : int | array_like of int | None.
        The components to be displayed. If None, plot will show the
        sources in the order as fitted.
    exclude : array_like of int
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    title : str
        The figure title.
    show : bool
        Show figure if True.
    labels : None | dict
        The ICA labels attribute.
    """
    import matplotlib.pyplot as plt
    if title is None:
        title = 'Reconstructed latent sources, time-locked'

    fig, axes = plt.subplots(1)
    ax = axes
    axes = [axes]
    idxs = [0]
    times = evoked.times * 1e3

    # plot unclassified sources and label excluded ones
    lines = list()
    texts = list()
    if picks is None:
        picks = np.arange(evoked.data.shape[0])
    picks = np.sort(picks)
    idxs = [picks]
    color = 'r'

    if labels is not None:
        labels_used = [k for k in labels if '/' not in k]

    exclude_labels = list()
    for ii in picks:
        if ii in exclude:
            line_label = 'ICA %03d' % (ii + 1)
            if labels is not None:
                annot = list()
                for this_label in labels_used:
                    indices = labels[this_label]
                    if ii in indices:
                        annot.append(this_label)

                line_label += (' - ' + ', '.join(annot))
            exclude_labels.append(line_label)
        else:
            exclude_labels.append(None)

    if labels is not None:
        # compute colors only based on label categories
        unique_labels = set([k.split(' - ')[1] for k in exclude_labels if k])
        label_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label_colors = dict(zip(unique_labels, label_colors))
    else:
        label_colors = dict((k, 'red') for k in exclude_labels)

    for exc_label, ii in zip(exclude_labels, picks):
        if exc_label is not None:
            # create look up for color ...
            if ' - ' in exc_label:
                key = exc_label.split(' - ')[1]
            else:
                key = exc_label
            color = label_colors[key]
            # ... but display component number too
            lines.extend(ax.plot(times, evoked.data[ii].T, picker=3.,
                         zorder=2, color=color, label=exc_label))
        else:
            lines.extend(ax.plot(times, evoked.data[ii].T, picker=3.,
                                 color='k', zorder=1))

    ax.set_title(title)
    ax.set_xlim(times[[0, -1]])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('(NA)')
    if len(exclude) > 0:
        plt.legend(loc='best')
    tight_layout(fig=fig)

    # for old matplotlib, we actually need this to have a bounding
    # box (!), so we have to put some valid text here, change
    # alpha and  path effects later
    texts.append(ax.text(0, 0, 'blank', zorder=3,
                         verticalalignment='baseline',
                         horizontalalignment='left',
                         fontweight='bold', alpha=0))
    # this is done to give the structure of a list of lists of a group of lines
    # in each subplot
    lines = [lines]
    ch_names = evoked.ch_names

    from matplotlib import patheffects
    path_effects = [patheffects.withStroke(linewidth=2, foreground="w",
                                           alpha=0.75)]
    params = dict(axes=axes, texts=texts, lines=lines, idxs=idxs,
                  ch_names=ch_names, need_draw=False,
                  path_effects=path_effects)
    fig.canvas.mpl_connect('pick_event',
                           partial(_butterfly_onpick, params=params))
    fig.canvas.mpl_connect('button_press_event',
                           partial(_butterfly_on_button_press,
                                   params=params))
    plt_show(show)
    return fig


def plot_ica_scores(ica, scores,
                    exclude=None, labels=None,
                    axhline=None,
                    title='ICA component scores',
                    figsize=(12, 6), show=True):
    """Plot scores related to detected components.

    Use this function to asses how well your score describes outlier
    sources and how well you were detecting them.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    scores : array_like of float, shape (n ica components) | list of arrays
        Scores based on arbitrary metric to characterize ICA components.
    exclude : array_like of int
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    labels : str | list | 'ecg' | 'eog' | None
        The labels to consider for the axes tests. Defaults to None.
        If list, should match the outer shape of `scores`.
        If 'ecg' or 'eog', the ``labels_`` attributes will be looked up.
        Note that '/' is used internally for sublabels specifying ECG and
        EOG channels.
    axhline : float
        Draw horizontal line to e.g. visualize rejection threshold.
    title : str
        The figure title.
    figsize : tuple of int
        The figure size. Defaults to (12, 6).
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    my_range = np.arange(ica.n_components_)
    if exclude is None:
        exclude = ica.exclude
    exclude = np.unique(exclude)
    if not isinstance(scores[0], (list, np.ndarray)):
        scores = [scores]
    n_rows = len(scores)
    figsize = (12, 6) if figsize is None else figsize
    fig, axes = plt.subplots(n_rows, figsize=figsize, sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    plt.suptitle(title)

    if labels == 'ecg':
        labels = [l for l in ica.labels_ if l.startswith('ecg/')]
    elif labels == 'eog':
        labels = [l for l in ica.labels_ if l.startswith('eog/')]
        labels.sort(key=lambda l: l.split('/')[1])  # sort by index
    elif isinstance(labels, string_types):
        if len(axes) > 1:
            raise ValueError('Need as many labels as axes (%i)' % len(axes))
        labels = [labels]
    elif isinstance(labels, (tuple, list)):
        if len(labels) != len(axes):
            raise ValueError('Need as many labels as axes (%i)' % len(axes))
    elif labels is None:
        labels = (None, None)
    for label, this_scores, ax in zip(labels, scores, axes):
        if len(my_range) != len(this_scores):
            raise ValueError('The length of `scores` must equal the '
                             'number of ICA components.')
        ax.bar(my_range, this_scores, color='w')
        for excl in exclude:
            ax.bar(my_range[excl], this_scores[excl], color='r')
        if axhline is not None:
            if np.isscalar(axhline):
                axhline = [axhline]
            for axl in axhline:
                ax.axhline(axl, color='r', linestyle='--')
        ax.set_ylabel('score')

        if label is not None:
            if 'eog/' in label:
                split = label.split('/')
                label = ', '.join([split[0], split[2]])
            elif '/' in label:
                label = ', '.join(label.split('/'))
            ax.set_title('(%s)' % label)
        ax.set_xlabel('ICA components')
        ax.set_xlim(0, len(this_scores))

    tight_layout(fig=fig)
    if len(axes) > 1:
        plt.subplots_adjust(top=0.9)
    plt_show(show)
    return fig


def plot_ica_overlay(ica, inst, exclude=None, picks=None, start=None,
                     stop=None, title=None, show=True):
    """Overlay of raw and cleaned signals given the unmixing matrix.

    This method helps visualizing signal quality and artifact rejection.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    inst : instance of mne.io.Raw or mne.Evoked
        The signals to be compared given the ICA solution. If Raw input,
        The raw data are displayed before and after cleaning. In a second
        panel the cross channel average will be displayed. Since dipolar
        sources will be canceled out this display is sensitive to
        artifacts. If evoked input, butterfly plots for clean and raw
        signals will be superimposed.
    exclude : array_like of int
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels
        are used that were included on fitting).
    start : int
        X-axis start index. If None from the beginning.
    stop : int
        X-axis stop index. If None to the end.
    title : str
        The figure title.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of pyplot.Figure
        The figure.
    """
    # avoid circular imports
    from ..io.base import _BaseRaw
    from ..evoked import Evoked
    from ..preprocessing.ica import _check_start_stop

    if not isinstance(inst, (_BaseRaw, Evoked)):
        raise ValueError('Data input must be of Raw or Evoked type')
    if title is None:
        title = 'Signals before (red) and after (black) cleaning'
    if picks is None:
        picks = [inst.ch_names.index(k) for k in ica.ch_names]
    if exclude is None:
        exclude = ica.exclude
    if isinstance(inst, _BaseRaw):
        if start is None:
            start = 0.0
        if stop is None:
            stop = 3.0
        ch_types_used = [k for k in ['mag', 'grad', 'eeg'] if k in ica]
        start_compare, stop_compare = _check_start_stop(inst, start, stop)
        data, times = inst[picks, start_compare:stop_compare]

        raw_cln = ica.apply(inst.copy(), exclude=exclude,
                            start=start, stop=stop)
        data_cln, _ = raw_cln[picks, start_compare:stop_compare]
        fig = _plot_ica_overlay_raw(data=data, data_cln=data_cln,
                                    times=times * 1e3, title=title,
                                    ch_types_used=ch_types_used, show=show)
    elif isinstance(inst, Evoked):
        if start is not None and stop is not None:
            inst = inst.copy().crop(start, stop)
        if picks is not None:
            inst.pick_channels([inst.ch_names[p] for p in picks])
        evoked_cln = ica.apply(inst.copy(), exclude=exclude)
        fig = _plot_ica_overlay_evoked(evoked=inst, evoked_cln=evoked_cln,
                                       title=title, show=show)

    return fig


def _plot_ica_overlay_raw(data, data_cln, times, title, ch_types_used, show):
    """Plot evoked after and before ICA cleaning

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    epochs : instance of mne.Epochs
        The Epochs to be regarded.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of pyplot.Figure
    """
    import matplotlib.pyplot as plt
    # Restore sensor space data and keep all PCA components
    # let's now compare the date before and after cleaning.
    # first the raw data
    assert data.shape == data_cln.shape
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.suptitle(title)
    ax1.plot(times, data.T, color='r')
    ax1.plot(times, data_cln.T, color='k')
    ax1.set_xlabel('time (s)')
    ax1.set_xlim(times[0], times[-1])
    ax1.set_xlim(times[0], times[-1])
    ax1.set_title('Raw data')

    _ch_types = {'mag': 'Magnetometers',
                 'grad': 'Gradiometers',
                 'eeg': 'EEG'}
    ch_types = ', '.join([_ch_types[k] for k in ch_types_used])
    ax2.set_title('Average across channels ({0})'.format(ch_types))
    ax2.plot(times, data.mean(0), color='r')
    ax2.plot(times, data_cln.mean(0), color='k')
    ax2.set_xlim(100, 106)
    ax2.set_xlabel('time (ms)')
    ax2.set_xlim(times[0], times[-1])
    tight_layout(fig=fig)

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()
    plt_show(show)
    return fig


def _plot_ica_overlay_evoked(evoked, evoked_cln, title, show):
    """Plot evoked after and before ICA cleaning

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    epochs : instance of mne.Epochs
        The Epochs to be regarded.
    show : bool
        If True, all open plots will be shown.

    Returns
    -------
    fig : instance of pyplot.Figure
    """
    import matplotlib.pyplot as plt
    ch_types_used = [c for c in ['mag', 'grad', 'eeg'] if c in evoked]
    n_rows = len(ch_types_used)
    ch_types_used_cln = [c for c in ['mag', 'grad', 'eeg'] if
                         c in evoked_cln]

    if len(ch_types_used) != len(ch_types_used_cln):
        raise ValueError('Raw and clean evokeds must match. '
                         'Found different channels.')

    fig, axes = plt.subplots(n_rows, 1)
    fig.suptitle('Average signal before (red) and after (black) ICA')
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    evoked.plot(axes=axes, show=show)
    for ax in fig.axes:
        for l in ax.get_lines():
            l.set_color('r')
    fig.canvas.draw()
    evoked_cln.plot(axes=axes, show=show)
    tight_layout(fig=fig)

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()
    plt_show(show)
    return fig


def _plot_sources_raw(ica, raw, picks, exclude, start, stop, show, title,
                      block):
    """Function for plotting the ICA components as raw array."""
    color = _handle_default('color', (0., 0., 0.))
    orig_data = ica._transform_raw(raw, 0, len(raw.times)) * 0.2
    if picks is None:
        picks = range(len(orig_data))
    types = ['misc' for _ in picks]
    picks = list(sorted(picks))
    eog_chs = pick_types(raw.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(raw.info, meg=False, ecg=True, ref_meg=False)
    data = [orig_data[pick] for pick in picks]
    c_names = ['ICA %03d' % x for x in range(len(orig_data))]
    for eog_idx in eog_chs:
        c_names.append(raw.ch_names[eog_idx])
        types.append('eog')
    for ecg_idx in ecg_chs:
        c_names.append(raw.ch_names[ecg_idx])
        types.append('ecg')
    extra_picks = np.append(eog_chs, ecg_chs).astype(int)
    if len(extra_picks) > 0:
        eog_ecg_data, _ = raw[extra_picks, :]
        for idx in range(len(eog_ecg_data)):
            if idx < len(eog_chs):
                eog_ecg_data[idx] /= 150e-6  # scaling for eog
            else:
                eog_ecg_data[idx] /= 5e-4  # scaling for ecg
        data = np.append(data, eog_ecg_data, axis=0)

    for idx in range(len(extra_picks)):
        picks = np.append(picks, ica.n_components_ + idx)
    if title is None:
        title = 'ICA components'
    info = create_info([c_names[x] for x in picks], raw.info['sfreq'])

    info['bads'] = [c_names[x] for x in exclude]
    if start is None:
        start = 0
    if stop is None:
        stop = start + 20
        stop = min(stop, raw.times[-1])
    duration = stop - start
    if duration <= 0:
        raise RuntimeError('Stop must be larger than start.')
    t_end = int(duration * raw.info['sfreq'])
    times = raw.times[0:t_end]
    bad_color = (1., 0., 0.)
    inds = list(range(len(picks)))
    data = np.array(data)
    n_channels = min([20, len(picks)])
    params = dict(raw=raw, orig_data=data, data=data[:, 0:t_end],
                  ch_start=0, t_start=start, info=info, duration=duration,
                  ica=ica, n_channels=n_channels, times=times, types=types,
                  n_times=raw.n_times, bad_color=bad_color, picks=picks)
    _prepare_mne_browse_raw(params, title, 'w', color, bad_color, inds,
                            n_channels)
    params['scale_factor'] = 1.0
    params['plot_fun'] = partial(_plot_raw_traces, params=params, inds=inds,
                                 color=color, bad_color=bad_color)
    params['update_fun'] = partial(_update_data, params)
    params['pick_bads_fun'] = partial(_pick_bads, params=params)
    params['label_click_fun'] = partial(_label_clicked, params=params)
    _layout_figure(params)
    # callbacks
    callback_key = partial(_plot_raw_onkey, params=params)
    params['fig'].canvas.mpl_connect('key_press_event', callback_key)
    callback_scroll = partial(_plot_raw_onscroll, params=params)
    params['fig'].canvas.mpl_connect('scroll_event', callback_scroll)
    callback_pick = partial(_mouse_click, params=params)
    params['fig'].canvas.mpl_connect('button_press_event', callback_pick)
    callback_resize = partial(_helper_raw_resize, params=params)
    params['fig'].canvas.mpl_connect('resize_event', callback_resize)
    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    params['fig_proj'] = None
    params['event_times'] = None
    params['update_fun']()
    params['plot_fun']()
    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)
    return params['fig']


def _update_data(params):
    """Function for preparing the data on horizontal shift of the viewport."""
    sfreq = params['info']['sfreq']
    start = int(params['t_start'] * sfreq)
    end = int((params['t_start'] + params['duration']) * sfreq)
    params['data'] = params['orig_data'][:, start:end]
    params['times'] = params['raw'].times[start:end]


def _pick_bads(event, params):
    """Function for selecting components on click."""
    bads = params['info']['bads']
    params['info']['bads'] = _select_bads(event, params, bads)
    params['update_fun']()
    params['plot_fun']()


def _close_event(events, params):
    """Function for excluding the selected components on close."""
    info = params['info']
    c_names = ['ICA %03d' % x for x in range(params['ica'].n_components_)]
    exclude = [c_names.index(x) for x in info['bads'] if x.startswith('ICA')]
    params['ica'].exclude = exclude


def _plot_sources_epochs(ica, epochs, picks, exclude, start, stop, show,
                         title, block):
    """Function for plotting the components as epochs."""
    data = ica._transform_epochs(epochs, concatenate=True)
    eog_chs = pick_types(epochs.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(epochs.info, meg=False, ecg=True, ref_meg=False)
    c_names = ['ICA %03d' % x for x in range(ica.n_components_)]
    ch_types = np.repeat('misc', ica.n_components_)
    for eog_idx in eog_chs:
        c_names.append(epochs.ch_names[eog_idx])
        ch_types = np.append(ch_types, 'eog')
    for ecg_idx in ecg_chs:
        c_names.append(epochs.ch_names[ecg_idx])
        ch_types = np.append(ch_types, 'ecg')
    extra_picks = np.append(eog_chs, ecg_chs).astype(int)
    if len(extra_picks) > 0:
        eog_ecg_data = np.concatenate(epochs.get_data()[:, extra_picks],
                                      axis=1)
        data = np.append(data, eog_ecg_data, axis=0)
    scalings = _handle_default('scalings_plot_raw')
    scalings['misc'] = 5.0
    info = create_info(ch_names=c_names, sfreq=epochs.info['sfreq'],
                       ch_types=ch_types)
    info['projs'] = list()
    info['bads'] = [c_names[x] for x in exclude]
    if title is None:
        title = 'ICA components'
    if picks is None:
        picks = list(range(ica.n_components_))
    if start is None:
        start = 0
    if stop is None:
        stop = start + 20
        stop = min(stop, len(epochs.events))
    for idx in range(len(extra_picks)):
        picks = np.append(picks, ica.n_components_ + idx)
    n_epochs = stop - start
    if n_epochs <= 0:
        raise RuntimeError('Stop must be larger than start.')
    params = {'ica': ica,
              'epochs': epochs,
              'info': info,
              'orig_data': data,
              'bads': list(),
              'bad_color': (1., 0., 0.),
              't_start': start * len(epochs.times)}
    params['label_click_fun'] = partial(_label_clicked, params=params)
    _prepare_mne_browse_epochs(params, projs=list(), n_channels=20,
                               n_epochs=n_epochs, scalings=scalings,
                               title=title, picks=picks,
                               order=['misc', 'eog', 'ecg'])
    params['plot_update_proj_callback'] = _update_epoch_data
    _update_epoch_data(params)
    params['hsel_patch'].set_x(params['t_start'])
    callback_close = partial(_close_epochs_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)
    return params['fig']


def _update_epoch_data(params):
    """Function for preparing the data on horizontal shift."""
    start = params['t_start']
    n_epochs = params['n_epochs']
    end = start + n_epochs * len(params['epochs'].times)
    data = params['orig_data'][:, start:end]
    types = params['types']
    for pick, ind in enumerate(params['inds']):
        params['data'][pick] = data[ind] / params['scalings'][types[pick]]
    params['plot_fun']()


def _close_epochs_event(events, params):
    """Function for excluding the selected components on close."""
    info = params['info']
    exclude = [info['ch_names'].index(x) for x in info['bads']
               if x.startswith('ICA')]
    params['ica'].exclude = exclude


def _label_clicked(pos, params):
    """Function for plotting independent components on click to label."""
    import matplotlib.pyplot as plt
    offsets = np.array(params['offsets']) + params['offsets'][0]
    line_idx = np.searchsorted(offsets, pos[1]) + params['ch_start']
    if line_idx >= len(params['picks']):
        return
    ic_idx = [params['picks'][line_idx]]
    types = list()
    info = params['ica'].info
    if len(pick_types(info, meg=False, eeg=True, ref_meg=False)) > 0:
        types.append('eeg')
    if len(pick_types(info, meg='mag', ref_meg=False)) > 0:
        types.append('mag')
    if len(pick_types(info, meg='grad', ref_meg=False)) > 0:
        types.append('grad')

    ica = params['ica']
    data = np.dot(ica.mixing_matrix_[:, ic_idx].T,
                  ica.pca_components_[:ica.n_components_])
    data = np.atleast_2d(data)
    fig, axes = _prepare_trellis(len(types), max_col=3)
    for ch_idx, ch_type in enumerate(types):
        try:
            data_picks, pos, merge_grads, _, _ = _prepare_topo_plot(ica,
                                                                    ch_type,
                                                                    None)
        except Exception as exc:
            warn(exc)
            plt.close(fig)
            return
        this_data = data[:, data_picks]
        ax = axes[ch_idx]
        if merge_grads:
            from ..channels.layout import _merge_grad_data
        for ii, data_ in zip(ic_idx, this_data):
            ax.set_title('IC #%03d ' % ii + ch_type, fontsize=12)
            data_ = _merge_grad_data(data_) if merge_grads else data_
            plot_topomap(data_.flatten(), pos, axes=ax, show=False)
            _hide_frame(ax)
    tight_layout(fig=fig)
    fig.subplots_adjust(top=0.95)
    fig.canvas.draw()
    plt_show(True)
