"""Functions to plot ICA specific data (besides topographies)."""

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

from functools import partial

import numpy as np

from .utils import (tight_layout, _prepare_trellis, _select_bads,
                    _plot_raw_onscroll, _mouse_click,
                    _plot_raw_onkey, plt_show, _convert_psds)
from .topomap import (_prepare_topo_plot, plot_topomap, _hide_frame,
                      _plot_ica_topomap)
from .raw import _prepare_mne_browse_raw, _plot_raw_traces
from .epochs import _prepare_mne_browse_epochs, plot_epochs_image
from .evoked import _butterfly_on_button_press, _butterfly_onpick
from ..utils import warn, _validate_type, fill_doc
from ..defaults import _handle_default
from ..io.meas_info import create_info
from ..io.pick import (pick_types, _picks_to_idx, _get_channel_types,
                       _DATA_CH_TYPES_ORDER_DEFAULT)
from ..time_frequency.psd import psd_multitaper
from ..utils import _reject_data_segments


@fill_doc
def plot_ica_sources(ica, inst, picks=None, exclude='deprecated', start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False, show_scrollbars=True):
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
    %(picks_base)s all sources in the order as fitted.
    exclude : 'deprecated'
        The ``exclude`` parameter is deprecated and will be removed in version
        0.20; specify excluded components using the ``ICA.exclude`` attribute
        instead.
    start : int
        X-axis start index. If None, from the beginning.
    stop : int
        X-axis stop index. If None, next 20 are shown, in case of evoked to the
        end.
    title : str | None
        The window title. If None a default is provided.
    show : bool
        Show figure if True.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for interactive selection of components in raw and epoch
        plotter. For evoked, this parameter has no effect. Defaults to False.
    show_first_samp : bool
        If True, show time axis relative to the ``raw.first_samp``.
    %(show_scrollbars)s

    Returns
    -------
    fig : instance of Figure
        The figure.

    Notes
    -----
    For raw and epoch instances, it is possible to select components for
    exclusion by clicking on the line. The selected components are added to
    ``ica.exclude`` on close.

    .. versionadded:: 0.10.0
    """
    from ..io.base import BaseRaw
    from ..evoked import Evoked
    from ..epochs import BaseEpochs

    if exclude != 'deprecated':
        warn('The "exclude" parameter is deprecated and will be removed in '
             'version 0.20; specify excluded components using the ICA.exclude '
             'attribute instead. Provided value of {} will be ignored; falling'
             ' back to ICA.exclude'.format(exclude), DeprecationWarning)
    exclude = ica.exclude
    picks = _picks_to_idx(ica.n_components_, picks, 'all')

    if isinstance(inst, BaseRaw):
        fig = _plot_sources_raw(ica, inst, picks, exclude, start=start,
                                stop=stop, show=show, title=title,
                                block=block, show_first_samp=show_first_samp,
                                show_scrollbars=show_scrollbars)
    elif isinstance(inst, BaseEpochs):
        fig = _plot_sources_epochs(ica, inst, picks, exclude, start=start,
                                   stop=stop, show=show, title=title,
                                   block=block,
                                   show_scrollbars=show_scrollbars)
    elif isinstance(inst, Evoked):
        if start is not None or stop is not None:
            inst = inst.copy().crop(start, stop)
        sources = ica.get_sources(inst)
        fig = _plot_ica_sources_evoked(
            evoked=sources, picks=picks, exclude=exclude, title=title,
            labels=getattr(ica, 'labels_', None), show=show, ica=ica)
    else:
        raise ValueError('Data input must be of Raw or Epochs type')

    return fig


def _create_properties_layout(figsize=None):
    """Create main figure and axes layout used by plot_ica_properties."""
    import matplotlib.pyplot as plt
    if figsize is None:
        figsize = [7., 6.]
    fig = plt.figure(figsize=figsize, facecolor=[0.95] * 3)

    axes_params = (('topomap', [0.08, 0.5, 0.3, 0.45]),
                   ('image', [0.5, 0.6, 0.45, 0.35]),
                   ('erp', [0.5, 0.5, 0.45, 0.1]),
                   ('spectrum', [0.08, 0.1, 0.32, 0.3]),
                   ('variance', [0.5, 0.1, 0.45, 0.25]))
    axes = [fig.add_axes(loc, label=name) for name, loc in axes_params]

    return fig, axes


def _plot_ica_properties(pick, ica, inst, psds_mean, freqs, n_trials,
                         epoch_var, plot_lowpass_edge, epochs_src,
                         set_title_and_labels, plot_std, psd_ylabel,
                         spectrum_std, topomap_args, image_args, fig, axes,
                         kind, dropped_indices):
    """Plot ICA properties (helper)."""
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from scipy.stats import gaussian_kde

    topo_ax, image_ax, erp_ax, spec_ax, var_ax = axes

    # plotting
    # --------
    # component topomap
    _plot_ica_topomap(ica, pick, show=False, axes=topo_ax, **topomap_args)

    # image and erp
    # we create a new epoch with dropped rows
    epoch_data = epochs_src.get_data()
    epoch_data = np.insert(arr=epoch_data,
                           obj=(dropped_indices -
                                np.arange(len(dropped_indices))).astype(int),
                           values=0.0,
                           axis=0)
    from ..epochs import EpochsArray
    epochs_src = EpochsArray(epoch_data, epochs_src.info, verbose=0)

    plot_epochs_image(epochs_src, picks=pick, axes=[image_ax, erp_ax],
                      combine=None, colorbar=False, show=False,
                      **image_args)

    # spectrum
    spec_ax.plot(freqs, psds_mean, color='k')
    if plot_std:
        spec_ax.fill_between(freqs, psds_mean - spectrum_std[0],
                             psds_mean + spectrum_std[1],
                             color='k', alpha=.2)
    if plot_lowpass_edge:
        spec_ax.axvline(inst.info['lowpass'], lw=2, linestyle='--',
                        color='k', alpha=0.2)

    # epoch variance
    var_ax_divider = make_axes_locatable(var_ax)
    hist_ax = var_ax_divider.append_axes("right", size="33%", pad="2.5%")
    var_ax.scatter(range(len(epoch_var)), epoch_var, alpha=0.5,
                   facecolor=[0, 0, 0], lw=0)
    # rejected epochs in red
    var_ax.scatter(dropped_indices, epoch_var[dropped_indices],
                   alpha=1., facecolor=[1, 0, 0], lw=0)
    # compute percentage of dropped epochs
    var_percent = float(len(dropped_indices)) / float(len(epoch_var)) * 100.

    var_ax.set_yticks([])

    # histogram & histogram
    _, counts, _ = hist_ax.hist(epoch_var, orientation="horizontal",
                                color="k", alpha=.5)

    # kde
    kde = gaussian_kde(epoch_var)
    ymin, ymax = hist_ax.get_ylim()
    x = np.linspace(ymin, ymax, 50)
    kde_ = kde(x)
    kde_ /= kde_.max()
    kde_ *= hist_ax.get_xlim()[-1] * .9
    hist_ax.plot(kde_, x, color="k")
    hist_ax.set_ylim(ymin, ymax)

    # aesthetics
    # ----------
    topo_ax.set_title(ica._ica_names[pick])

    set_title_and_labels(image_ax, kind + ' image and ERP/ERF', [], kind)

    # erp
    set_title_and_labels(erp_ax, [], 'Time (s)', 'AU\n')
    erp_ax.spines["right"].set_color('k')
    erp_ax.set_xlim(epochs_src.times[[0, -1]])
    # remove half of yticks if more than 5
    yt = erp_ax.get_yticks()
    if len(yt) > 5:
        erp_ax.yaxis.set_ticks(yt[::2])

    # remove xticks - erp plot shows xticks for both image and erp plot
    image_ax.xaxis.set_ticks([])
    yt = image_ax.get_yticks()
    image_ax.yaxis.set_ticks(yt[1:])
    image_ax.set_ylim([-0.5, n_trials + 0.5])

    # spectrum
    set_title_and_labels(spec_ax, 'Spectrum', 'Frequency (Hz)', psd_ylabel)
    spec_ax.yaxis.labelpad = 0
    spec_ax.set_xlim(freqs[[0, -1]])
    ylim = spec_ax.get_ylim()
    air = np.diff(ylim)[0] * 0.1
    spec_ax.set_ylim(ylim[0] - air, ylim[1] + air)
    image_ax.axhline(0, color='k', linewidth=.5)

    # epoch variance
    var_ax_title = 'Dropped segments : %.2f %%' % var_percent
    set_title_and_labels(var_ax, var_ax_title,
                         kind + ' (index)',
                         'Variance (AU)')

    hist_ax.set_ylabel("")
    hist_ax.set_yticks([])
    set_title_and_labels(hist_ax, None, None, None)

    return fig


def _get_psd_label_and_std(this_psd, dB, ica, num_std):
    """Handle setting up PSD for one component, for plot_ica_properties."""
    psd_ylabel = _convert_psds(this_psd, dB, estimate='auto', scaling=1.,
                               unit='AU', ch_names=ica.ch_names)
    psds_mean = this_psd.mean(axis=0)
    diffs = this_psd - psds_mean
    # the distribution of power for each frequency bin is highly
    # skewed so we calculate std for values below and above average
    # separately - this is used for fill_between shade
    spectrum_std = [
        [np.sqrt((d[d < 0] ** 2).mean(axis=0)) for d in diffs.T],
        [np.sqrt((d[d > 0] ** 2).mean(axis=0)) for d in diffs.T]]
    spectrum_std = np.array(spectrum_std) * num_std

    return psd_ylabel, psds_mean, spectrum_std


@fill_doc
def plot_ica_properties(ica, inst, picks=None, axes=None, dB=True,
                        plot_std=True, topomap_args=None, image_args=None,
                        psd_args=None, figsize=None, show=True, reject='auto'):
    """Display component properties.

    Properties include the topography, epochs image, ERP/ERF, power
    spectrum, and epoch variance.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    inst: instance of Epochs or Raw
        The data to use in plotting properties.
    %(picks_base)s the first five sources.
        If more than one components were chosen in the picks,
        each one will be plotted in a separate figure.
    axes: list of matplotlib axes | None
        List of five matplotlib axes to use in plotting: [topomap_axis,
        image_axis, erp_axis, spectrum_axis, variance_axis]. If None a new
        figure with relevant axes is created. Defaults to None.
    dB: bool
        Whether to plot spectrum in dB. Defaults to True.
    plot_std: bool | float
        Whether to plot standard deviation/confidence intervals in ERP/ERF and
        spectrum plots.
        Defaults to True, which plots one standard deviation above/below for
        the spectrum. If set to float allows to control how many standard
        deviations are plotted for the spectrum. For example 2.5 will plot 2.5
        standard deviation above/below.
        For the ERP/ERF, by default, plot the 95 percent parametric confidence
        interval is calculated. To change this, use ``ci`` in ``ts_args`` in
        ``image_args`` (see below).
    topomap_args : dict | None
        Dictionary of arguments to ``plot_topomap``. If None, doesn't pass any
        additional arguments. Defaults to None.
    image_args : dict | None
        Dictionary of arguments to ``plot_epochs_image``. If None, doesn't pass
        any additional arguments. Defaults to None.
    psd_args : dict | None
        Dictionary of arguments to ``psd_multitaper``. If None, doesn't pass
        any additional arguments. Defaults to None.
    figsize : array-like, shape (2,) | None
        Allows to control size of the figure. If None, the figure size
        defaults to [7., 6.].
    show : bool
        Show figure if True.
    reject : 'auto' | dict | None
        Allows to specify rejection parameters used to drop epochs
        (or segments if continuous signal is passed as inst).
        If None, no rejection is applied. The default is 'auto',
        which applies the rejection parameters used when fitting
        the ICA object.

    Returns
    -------
    fig : list
        List of matplotlib figures.

    Notes
    -----
    .. versionadded:: 0.13
    """
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..preprocessing import ICA
    from ..io import RawArray

    # input checks and defaults
    # -------------------------
    _validate_type(inst, (BaseRaw, BaseEpochs), "inst", "Raw or Epochs")
    _validate_type(ica, ICA, "ica", "ICA")
    if isinstance(plot_std, bool):
        num_std = 1. if plot_std else 0.
    elif isinstance(plot_std, (float, int)):
        num_std = plot_std
        plot_std = True
    else:
        raise ValueError('plot_std has to be a bool, int or float, '
                         'got %s instead' % type(plot_std))

    # if no picks given - plot the first 5 components
    limit = min(5, ica.n_components_) if picks is None else len(ica.ch_names)
    picks = _picks_to_idx(ica.info, picks, 'all')[:limit]
    if axes is None:
        fig, axes = _create_properties_layout(figsize=figsize)
    else:
        if len(picks) > 1:
            raise ValueError('Only a single pick can be drawn '
                             'to a set of axes.')
        from .utils import _validate_if_list_of_axes
        _validate_if_list_of_axes(axes, obligatory_len=5)
        fig = axes[0].get_figure()

    psd_args = dict() if psd_args is None else psd_args
    topomap_args = dict() if topomap_args is None else topomap_args
    image_args = dict() if image_args is None else image_args
    image_args["ts_args"] = dict(truncate_xaxis=False, show_sensors=False)
    if plot_std:
        from ..stats.parametric import _parametric_ci
        image_args["ts_args"]["ci"] = _parametric_ci
    elif "ts_args" not in image_args or "ci" not in image_args["ts_args"]:
        image_args["ts_args"]["ci"] = False

    for item_name, item in (("psd_args", psd_args),
                            ("topomap_args", topomap_args),
                            ("image_args", image_args)):
        _validate_type(item, dict, item_name, "dictionary")
    if dB is not None:
        _validate_type(dB, bool, "dB", "bool")

    # calculations
    # ------------

    if isinstance(inst, BaseRaw):
        # when auto, delegate reject to the ica
        if reject == 'auto':
            reject = getattr(ica, 'reject_', None)
        else:
            pass

        if reject is None:
            inst_rejected = inst
            drop_inds = None
        else:
            data = inst.get_data()
            data, drop_inds = _reject_data_segments(data, ica.reject_,
                                                    flat=None, decim=None,
                                                    info=inst.info,
                                                    tstep=2.0)
            inst_rejected = RawArray(data, inst.info)

        # break up continuous signal into segments
        from ..epochs import _segment_raw
        inst_rejected = _segment_raw(inst_rejected,
                                     segment_length=2.,
                                     verbose=False,
                                     preload=True)
        inst = _segment_raw(inst, segment_length=2., verbose=False,
                            preload=True)
        kind = "Segment"
    else:
        drop_inds = None
        inst_rejected = inst
        kind = "Epochs"

    epochs_src = ica.get_sources(inst_rejected)
    data = epochs_src.get_data()

    ica_data = np.swapaxes(data[:, picks, :], 0, 1)

    # getting dropped epochs indexes
    if drop_inds is not None:
        dropped_indices = [(d[0] // len(inst.times)) + 1
                           for d in drop_inds]
    else:
        dropped_indices = []

    # getting ica sources from inst
    dropped_src = ica.get_sources(inst).get_data()
    dropped_src = np.swapaxes(dropped_src[:, picks, :], 0, 1)

    # spectrum
    Nyquist = inst.info['sfreq'] / 2.
    lp = inst.info['lowpass']
    if 'fmax' not in psd_args:
        psd_args['fmax'] = min(lp * 1.25, Nyquist)
    plot_lowpass_edge = lp < Nyquist and (psd_args['fmax'] > lp)
    psds, freqs = psd_multitaper(epochs_src, picks=picks, **psd_args)

    def set_title_and_labels(ax, title, xlab, ylab):
        if title:
            ax.set_title(title)
        if xlab:
            ax.set_xlabel(xlab)
        if ylab:
            ax.set_ylabel(ylab)
        ax.axis('auto')
        ax.tick_params('both', labelsize=8)
        ax.axis('tight')

    # plot
    # ----
    all_fig = list()
    for idx, pick in enumerate(picks):

        # calculate component-specific spectrum stuff
        psd_ylabel, psds_mean, spectrum_std = _get_psd_label_and_std(
            psds[:, idx, :].copy(), dB, ica, num_std)

        # if more than one component, spawn additional figures and axes
        if idx > 0:
            fig, axes = _create_properties_layout(figsize=figsize)

        # we reconstruct an epoch_variance with 0 where indexes where dropped
        epoch_var = np.var(ica_data[idx], axis=1)
        drop_var = np.var(dropped_src[idx], axis=1)
        drop_indices_corrected = \
            (dropped_indices -
             np.arange(len(dropped_indices))).astype(int)
        epoch_var = np.insert(arr=epoch_var,
                              obj=drop_indices_corrected,
                              values=drop_var[dropped_indices],
                              axis=0)

        # the actual plot
        fig = _plot_ica_properties(
            pick, ica, inst, psds_mean, freqs, ica_data.shape[1],
            epoch_var, plot_lowpass_edge,
            epochs_src, set_title_and_labels, plot_std, psd_ylabel,
            spectrum_std, topomap_args, image_args, fig, axes, kind,
            dropped_indices)
        all_fig.append(fig)

    plt_show(show)
    return all_fig


def _plot_ica_sources_evoked(evoked, picks, exclude, title, show, ica,
                             labels=None):
    """Plot average over epochs in ICA space.

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The Evoked to be used.
    %(picks_base)s all sources in the order as fitted.
    exclude : array-like of int
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
    from matplotlib import patheffects

    if title is None:
        title = 'Reconstructed latent sources, time-locked'

    fig, axes = plt.subplots(1)
    ax = axes
    axes = [axes]
    times = evoked.times * 1e3

    # plot unclassified sources and label excluded ones
    lines = list()
    texts = list()
    picks = np.sort(picks)
    idxs = [picks]

    if labels is not None:
        labels_used = [k for k in labels if '/' not in k]

    exclude_labels = list()
    for ii in picks:
        if ii in exclude:
            line_label = ica._ica_names[ii]
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
        unique_labels = {k.split(' - ')[1] for k in exclude_labels if k}
        label_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label_colors = dict(zip(unique_labels, label_colors))
    else:
        label_colors = {k: 'red' for k in exclude_labels}

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

    ax.set(title=title, xlim=times[[0, -1]], xlabel='Time (ms)', ylabel='(NA)')
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


def plot_ica_scores(ica, scores, exclude=None, labels=None, axhline=None,
                    title='ICA component scores', figsize=None, show=True):
    """Plot scores related to detected components.

    Use this function to asses how well your score describes outlier
    sources and how well you were detecting them.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    scores : array-like of float, shape (n_ica_components,) | list of array
        Scores based on arbitrary metric to characterize ICA components.
    exclude : array-like of int
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
    figsize : tuple of int | None
        The figure size. If None it gets set automatically.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of Figure
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
    if figsize is None:
        figsize = (6.4, 2.7 * n_rows)
    fig, axes = plt.subplots(n_rows, figsize=figsize, sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    axes[0].set_title(title)

    if labels == 'ecg':
        labels = [l for l in ica.labels_ if l.startswith('ecg/')]
    elif labels == 'eog':
        labels = [l for l in ica.labels_ if l.startswith('eog/')]
        labels.sort(key=lambda l: l.split('/')[1])  # sort by index
    elif isinstance(labels, str):
        if len(axes) > 1:
            raise ValueError('Need as many labels as axes (%i)' % len(axes))
        labels = [labels]
    elif isinstance(labels, (tuple, list)):
        if len(labels) != len(axes):
            raise ValueError('Need as many labels as axes (%i)' % len(axes))
    elif labels is None:
        labels = (None,) * n_rows

    for label, this_scores, ax in zip(labels, scores, axes):
        if len(my_range) != len(this_scores):
            raise ValueError('The length of `scores` must equal the '
                             'number of ICA components.')
        ax.bar(my_range, this_scores, color='gray', edgecolor='k')
        for excl in exclude:
            ax.bar(my_range[excl], this_scores[excl], color='r', edgecolor='k')
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
        ax.set_xlim(-0.6, len(this_scores) - 0.4)

    tight_layout(fig=fig)
    plt_show(show)
    return fig


@fill_doc
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
    exclude : array-like of int | None (default)
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    %(picks_base)s all channels that were included during fitting.
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
    fig : instance of Figure
        The figure.
    """
    # avoid circular imports
    from ..io.base import BaseRaw
    from ..evoked import Evoked
    from ..preprocessing.ica import _check_start_stop

    _validate_type(inst, (BaseRaw, Evoked), "inst", "Raw or Evoked")
    if title is None:
        title = 'Signals before (red) and after (black) cleaning'
    picks = ica.ch_names if picks is None else picks
    picks = _picks_to_idx(inst.info, picks, exclude=())
    ch_types_used = _get_channel_types(inst.info, picks=picks, unique=True)
    if exclude is None:
        exclude = ica.exclude
    if not isinstance(exclude, (np.ndarray, list)):
        raise TypeError('exclude must be of type list. Got %s'
                        % type(exclude))
    if isinstance(inst, BaseRaw):
        if start is None:
            start = 0.0
        if stop is None:
            stop = 3.0
        start_compare, stop_compare = _check_start_stop(inst, start, stop)
        data, times = inst[picks, start_compare:stop_compare]

        raw_cln = ica.apply(inst.copy(), exclude=exclude,
                            start=start, stop=stop)
        data_cln, _ = raw_cln[picks, start_compare:stop_compare]
        fig = _plot_ica_overlay_raw(data=data, data_cln=data_cln,
                                    times=times, title=title,
                                    ch_types_used=ch_types_used, show=show)
    elif isinstance(inst, Evoked):
        inst = inst.copy().crop(start, stop)
        if picks is not None:
            inst.info['comps'] = []  # can be safely disabled
            inst.pick_channels([inst.ch_names[p] for p in picks])
        evoked_cln = ica.apply(inst.copy(), exclude=exclude)
        fig = _plot_ica_overlay_evoked(evoked=inst, evoked_cln=evoked_cln,
                                       title=title, show=show)

    return fig


def _plot_ica_overlay_raw(data, data_cln, times, title, ch_types_used, show):
    """Plot evoked after and before ICA cleaning.

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
    fig : instance of Figure
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
    ax1.set(xlabel='Time (s)', xlim=times[[0, -1]], title='Raw data')

    _ch_types = {'mag': 'Magnetometers',
                 'grad': 'Gradiometers',
                 'eeg': 'EEG'}
    ch_types = ', '.join([_ch_types[k] for k in ch_types_used])
    ax2.set_title('Average across channels ({})'.format(ch_types))
    ax2.plot(times, data.mean(0), color='r')
    ax2.plot(times, data_cln.mean(0), color='k')
    ax2.set(xlabel='Time (s)', xlim=times[[0, -1]])
    tight_layout(fig=fig)

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()
    plt_show(show)
    return fig


def _plot_ica_overlay_evoked(evoked, evoked_cln, title, show):
    """Plot evoked after and before ICA cleaning.

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
    fig : instance of Figure
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

    evoked.plot(axes=axes, show=show, time_unit='s')
    for ax in fig.axes:
        for l in ax.get_lines():
            l.set_color('r')
    fig.canvas.draw()
    evoked_cln.plot(axes=axes, show=show, time_unit='s')
    tight_layout(fig=fig)

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()
    plt_show(show)
    return fig


def _plot_sources_raw(ica, raw, picks, exclude, start, stop, show, title,
                      block, show_first_samp, show_scrollbars):
    """Plot the ICA components as raw array."""
    color = _handle_default('color', (0., 0., 0.))
    orig_data = ica._transform_raw(raw, 0, len(raw.times)) * 0.2
    types = ['misc' for _ in picks]
    eog_chs = pick_types(raw.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(raw.info, meg=False, ecg=True, ref_meg=False)
    data = [orig_data[pick] for pick in picks]
    c_names = list(ica._ica_names)  # new list
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
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    params = dict(raw=raw, orig_data=data, data=data[:, 0:t_end], inds=inds,
                  ch_start=0, t_start=start, info=info, duration=duration,
                  ica=ica, n_channels=n_channels, times=times, types=types,
                  n_times=raw.n_times, bad_color=bad_color, picks=picks,
                  first_time=first_time, data_picks=[], decim=1,
                  noise_cov=None, whitened_ch_names=(), clipping=None,
                  use_scalebars=False, show_scrollbars=show_scrollbars)
    _prepare_mne_browse_raw(params, title, 'w', color, bad_color, inds,
                            n_channels)
    params['scale_factor'] = 1.0
    params['plot_fun'] = partial(_plot_raw_traces, params=params, color=color,
                                 bad_color=bad_color)
    params['update_fun'] = partial(_update_data, params)
    params['pick_bads_fun'] = partial(_pick_bads, params=params)
    params['label_click_fun'] = partial(_label_clicked, params=params)
    # callbacks
    callback_key = partial(_plot_raw_onkey, params=params)
    params['fig'].canvas.mpl_connect('key_press_event', callback_key)
    callback_scroll = partial(_plot_raw_onscroll, params=params)
    params['fig'].canvas.mpl_connect('scroll_event', callback_scroll)
    callback_pick = partial(_mouse_click, params=params)
    params['fig'].canvas.mpl_connect('button_press_event', callback_pick)
    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    params['fig_proj'] = None
    params['event_times'] = None
    params['butterfly'] = False
    params['update_fun']()
    params['plot_fun']()
    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)
    return params['fig']


def _update_data(params):
    """Prepare the data on horizontal shift of the viewport."""
    sfreq = params['info']['sfreq']
    start = int((params['t_start'] - params['first_time']) * sfreq)
    end = int((params['t_start'] + params['duration']) * sfreq)
    params['data'] = params['orig_data'][:, start:end]
    params['times'] = params['raw'].times[start:end]


def _pick_bads(event, params):
    """Select components on click."""
    bads = params['info']['bads']
    params['info']['bads'] = _select_bads(event, params, bads)
    params['update_fun']()
    params['plot_fun']()


def _close_event(events, params):
    """Exclude the selected components on close."""
    info = params['info']
    exclude = [params['ica']._ica_names.index(x)
               for x in info['bads'] if x.startswith('ICA')]
    params['ica'].exclude = exclude


def _plot_sources_epochs(ica, epochs, picks, exclude, start, stop, show,
                         title, block, show_scrollbars):
    """Plot the components as epochs."""
    data = ica._transform_epochs(epochs, concatenate=True)
    eog_chs = pick_types(epochs.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(epochs.info, meg=False, ecg=True, ref_meg=False)
    c_names = list(ica._ica_names)
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
    params = dict(ica=ica, epochs=epochs, info=info, orig_data=data,
                  bads=list(), bad_color=(1., 0., 0.),
                  t_start=start * len(epochs.times),
                  data_picks=list(), decim=1, whitened_ch_names=(),
                  noise_cov=None, show_scrollbars=show_scrollbars)
    params['label_click_fun'] = partial(_label_clicked, params=params)
    # changing the order to 'misc' before 'eog' and 'ecg'
    order = list(_DATA_CH_TYPES_ORDER_DEFAULT)
    order.pop(order.index('misc'))
    order.insert(order.index('eog'), 'misc')
    _prepare_mne_browse_epochs(params, projs=list(), n_channels=20,
                               n_epochs=n_epochs, scalings=scalings,
                               title=title, picks=picks,
                               order=order, info=info)
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
    """Prepare the data on horizontal shift."""
    start = params['t_start']
    n_epochs = params['n_epochs']
    end = start + n_epochs * len(params['epochs'].times)
    data = params['orig_data'][:, start:end]
    types = params['types']
    for pick, ind in enumerate(params['inds']):
        params['data'][pick] = data[ind] / params['scalings'][types[pick]]
    params['plot_fun']()


def _close_epochs_event(events, params):
    """Exclude the selected components on close."""
    info = params['info']
    exclude = [info['ch_names'].index(x) for x in info['bads']
               if x.startswith('IC')]
    params['ica'].exclude = exclude


def _label_clicked(pos, params):
    """Plot independent components on click to label."""
    import matplotlib.pyplot as plt
    offsets = np.array(params['offsets']) + params['offsets'][0]
    line_idx = np.searchsorted(offsets, pos[1]) + params['ch_start']
    if line_idx >= len(params['picks']):
        return
    ic_idx = [params['picks'][line_idx]]
    if params['types'][line_idx] != 'misc':
        warn('Can only plot ICA components.')
        return
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
            ax.set_title('%s %s' % (ica._ica_names[ii], ch_type), fontsize=12)
            data_ = _merge_grad_data(data_) if merge_grads else data_
            plot_topomap(data_.flatten(), pos, axes=ax, show=False)
            _hide_frame(ax)
    tight_layout(fig=fig)
    fig.subplots_adjust(top=0.88, bottom=0.)
    fig.canvas.draw()
    plt_show(True)
