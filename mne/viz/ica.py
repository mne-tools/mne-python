"""Functions to plot ICA specific data (besides topographies)
"""
from __future__ import print_function

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

from functools import partial

import numpy as np

from .utils import tight_layout, _prepare_trellis


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
                     stop=None, show=True, title=None):
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
    picks : ndarray | None.
        The components to be displayed. If None, plot will show the
        sources in the order as fitted.
    exclude : array_like of int
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    start : int
        X-axis start index. If None from the beginning.
    stop : int
        X-axis stop index. If None to the end.
    show : bool
        Show figure if True.
    title : str | None
        The figure title. If None a default is provided.

    Returns
    -------
    fig : instance of pyplot.Figure
        The figure.
    """

    from ..io.base import _BaseRaw
    from ..evoked import Evoked
    from ..epochs import _BaseEpochs

    if exclude is None:
        exclude = ica.exclude

    if isinstance(inst, (_BaseRaw, _BaseEpochs)):
        if isinstance(inst, _BaseRaw):
            sources = ica._transform_raw(inst, start, stop)
        else:
            if start is not None or stop is not None:
                inst = inst.crop(start, stop, copy=True)
            sources = ica._transform_epochs(inst, concatenate=True)
        if picks is not None:
            if np.isscalar(picks):
                picks = [picks]
            sources = np.atleast_2d(sources[picks])

        fig = _plot_ica_grid(sources, start=start, stop=stop,
                             ncol=len(sources) // 10 or 1,
                             exclude=exclude,
                             source_idx=picks,
                             title=title, show=show)
    elif isinstance(inst, Evoked):
        sources = ica.get_sources(inst)
        if start is not None or stop is not None:
            inst = inst.crop(start, stop, copy=True)
        fig = _plot_ica_sources_evoked(evoked=sources,
                                       exclude=exclude,
                                       title=title, show=show)
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

    if show:
        plt.show()

    return fig


def _plot_ica_sources_evoked(evoked, exclude, title, show):
    """Plot average over epochs in ICA space

    Parameters
    ----------
    ica : instance of mne.prerocessing.ICA
        The ICA object.
    epochs : instance of mne.Epochs
        The Epochs to be regarded.
    title : str
        The figure title.
    show : bool
        Show figure if True.
    """
    import matplotlib.pyplot as plt
    if title is None:
        title = 'Reconstructed latent sources, time-locked'

    fig = plt.figure()
    times = evoked.times * 1e3

    # plot unclassified sources
    plt.plot(times, evoked.data.T, 'k')
    for ii in exclude:
        # use indexing to expose event related sources
        plt.plot(times, evoked.data[ii].T, color='r', label='ICA %02d' % ii)

    plt.title(title)
    plt.xlim(times[[0, -1]])
    plt.xlabel('Time (ms)')
    plt.ylabel('(NA)')
    plt.legend(loc='best')
    tight_layout(fig=fig)

    if show:
        plt.show()

    return fig


def plot_ica_scores(ica, scores, exclude=None, axhline=None,
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
    for this_scores, ax in zip(scores, axes):
        if len(my_range) != len(this_scores):
            raise ValueError('The length ofr `scores` must equal the '
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
        ax.set_xlabel('ICA components')
        ax.set_xlim(0, len(this_scores))

    tight_layout(fig=fig)
    if len(axes) > 1:
        plt.subplots_adjust(top=0.9)

    if show:
        plt.show()
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
        raise ValueError('Data input must be of Raw or Epochs type')
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

        raw_cln = ica.apply(inst, exclude=exclude, start=start, stop=stop,
                            copy=True)
        data_cln, _ = raw_cln[picks, start_compare:stop_compare]
        fig = _plot_ica_overlay_raw(data=data, data_cln=data_cln,
                                    times=times * 1e3, title=title,
                                    ch_types_used=ch_types_used, show=show)
    elif isinstance(inst, Evoked):
        if start is not None and stop is not None:
            inst = inst.crop(start, stop, copy=True)
        if picks is not None:
            inst.pick_channels([inst.ch_names[p] for p in picks])
        evoked_cln = ica.apply(inst, exclude=exclude, copy=True)
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

    if show:
        plt.show()

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
        [l.set_color('r') for l in ax.get_lines()]
    fig.canvas.draw()
    evoked_cln.plot(axes=axes, show=show)
    tight_layout(fig=fig)

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()

    if show:
        plt.show()

    return fig
