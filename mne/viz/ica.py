"""Functions to plot ICA specific data (besides topographies)."""

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

from functools import partial
import warnings

import numpy as np

from .utils import (tight_layout, _make_event_color_dict, _get_cmap,
                    plt_show, _convert_psds, _compute_scalings,
                    _handle_precompute)
from .topomap import _plot_ica_topomap
from .epochs import plot_epochs_image
from .evoked import _butterfly_on_button_press, _butterfly_onpick
from ..channels.channels import _get_ch_type
from ..utils import _validate_type, fill_doc
from ..defaults import _handle_default
from ..io.meas_info import create_info
from ..io.pick import pick_types, _picks_to_idx
from ..utils import _reject_data_segments, verbose


@fill_doc
def plot_ica_sources(ica, inst, picks=None, start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False, show_scrollbars=True,
                     time_format='float', precompute=None,
                     use_opengl=None, *, theme=None, overview_mode=None):
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
    %(picks_ica)s
    start, stop : float | int | None
       If ``inst`` is a `~mne.io.Raw` or an `~mne.Evoked` object, the first and
       last time point (in seconds) of the data to plot. If ``inst`` is a
       `~mne.io.Raw` object, ``start=None`` and ``stop=None`` will be
       translated into ``start=0.`` and ``stop=3.``, respectively. For
       `~mne.Evoked`, ``None`` refers to the beginning and end of the evoked
       signal. If ``inst`` is an `~mne.Epochs` object, specifies the index of
       the first and last epoch to show.
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
    %(time_format)s
    %(precompute)s
    %(use_opengl)s
    %(theme_pg)s

        .. versionadded:: 1.0
    %(overview_mode)s

        .. versionadded:: 1.1

    Returns
    -------
    %(browser)s

    Notes
    -----
    For raw and epoch instances, it is possible to select components for
    exclusion by clicking on the line. The selected components are added to
    ``ica.exclude`` on close.

    %(notes_2d_backend)s

    .. versionadded:: 0.10.0
    """
    from ..io.base import BaseRaw
    from ..evoked import Evoked
    from ..epochs import BaseEpochs

    exclude = ica.exclude
    picks = _picks_to_idx(ica.n_components_, picks, picks_on="components")

    if isinstance(inst, (BaseRaw, BaseEpochs)):
        fig = _plot_sources(ica, inst, picks, exclude, start=start, stop=stop,
                            show=show, title=title, block=block,
                            show_first_samp=show_first_samp,
                            show_scrollbars=show_scrollbars,
                            time_format=time_format, precompute=precompute,
                            use_opengl=use_opengl, theme=theme,
                            overview_mode=overview_mode)
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


def _create_properties_layout(figsize=None, fig=None):
    """Create main figure and axes layout used by plot_ica_properties."""
    import matplotlib.pyplot as plt
    if fig is not None and figsize is not None:
        raise ValueError('Cannot specify both fig and figsize.')
    if figsize is None:
        figsize = [7., 6.]
    if fig is None:
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
                         spectrum_std, log_scale, topomap_args, image_args,
                         fig, axes, kind, dropped_indices):
    """Plot ICA properties (helper)."""
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from scipy.stats import gaussian_kde

    topo_ax, image_ax, erp_ax, spec_ax, var_ax = axes

    # plotting
    # --------
    # component topomap
    _plot_ica_topomap(ica, pick, show=False, axes=topo_ax, **topomap_args)
    topo_ax._ch_type = _get_ch_type(ica, ch_type=None,
                                    allow_ref_meg=ica.allow_ref_meg)

    # image and erp
    # we create a new epoch with dropped rows
    epoch_data = epochs_src.get_data()
    epoch_data = np.insert(arr=epoch_data,
                           obj=(dropped_indices -
                                np.arange(len(dropped_indices))).astype(int),
                           values=0.0,
                           axis=0)
    from ..epochs import EpochsArray
    epochs_src = EpochsArray(epoch_data, epochs_src.info, tmin=epochs_src.tmin,
                             verbose=0)

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

    # histogram & histogram
    _, counts, _ = hist_ax.hist(epoch_var, orientation="horizontal",
                                color="k", alpha=.5)

    # kde
    ymin, ymax = hist_ax.get_ylim()
    try:
        kde = gaussian_kde(epoch_var)
    except np.linalg.LinAlgError:
        pass  # singular: happens when there is nothing plotted
    else:
        x = np.linspace(ymin, ymax, 50)
        kde_ = kde(x)
        kde_ /= kde_.max() or 1.
        kde_ *= hist_ax.get_xlim()[-1] * .9
        hist_ax.plot(kde_, x, color="k")
        hist_ax.set_ylim(ymin, ymax)

    # aesthetics
    # ----------
    set_title_and_labels(image_ax, kind + ' image and ERP/ERF', [], kind)

    # erp
    set_title_and_labels(erp_ax, [], 'Time (s)', 'AU')
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

    def _set_scale(ax, scale):
        """Set the scale of a matplotlib axis."""
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.relim()
        ax.autoscale()

    # spectrum
    set_title_and_labels(spec_ax, 'Spectrum', 'Frequency (Hz)', psd_ylabel)
    spec_ax.yaxis.labelpad = 0
    spec_ax.set_xlim(freqs[[0, -1]])
    ylim = spec_ax.get_ylim()
    air = np.diff(ylim)[0] * 0.1
    spec_ax.set_ylim(ylim[0] - air, ylim[1] + air)
    image_ax.axhline(0, color='k', linewidth=.5)
    if log_scale:
        _set_scale(spec_ax, 'log')

    # epoch variance
    var_ax_title = 'Dropped segments: %.2f %%' % var_percent
    set_title_and_labels(var_ax, var_ax_title, kind, 'Variance (AU)')

    hist_ax.set_ylabel("")
    hist_ax.set_yticks([])
    set_title_and_labels(hist_ax, None, None, None)

    def _plot_ica_properties_on_press(event, ica, pick, topomap_args):
        """Handle keypress events for ica properties plot."""
        import matplotlib.pyplot as plt
        fig = event.canvas.figure
        if event.key == 'escape':
            plt.close(fig)
        if event.key in ('t', 'l'):
            ax_labels = [ax.get_label() for ax in fig.axes]
            if event.key == 't':
                ax = fig.axes[ax_labels.index('topomap')]
                ax.clear()
                ch_types = list(set(ica.get_channel_types()))
                ch_type = \
                    ch_types[(ch_types.index(ax._ch_type) + 1) % len(ch_types)]
                _plot_ica_topomap(ica, pick, ch_type=ch_type, show=False,
                                  axes=ax, **topomap_args)
                ax._ch_type = ch_type
            elif event.key == 'l':
                ax = fig.axes[ax_labels.index('spectrum')]
                _set_scale(ax, 'linear' if ax.get_xscale() == 'log' else 'log')
            del ax
            fig.canvas.draw()

    # add keypress event handler
    fig.canvas.mpl_connect(
        'key_press_event', lambda event: _plot_ica_properties_on_press(
            event, ica, pick, topomap_args))

    return fig


def _get_psd_label_and_std(this_psd, dB, ica, num_std):
    """Handle setting up PSD for one component, for plot_ica_properties."""
    psd_ylabel = _convert_psds(this_psd, dB, estimate='auto', scaling=1.,
                               unit='AU', first_dim='epoch')
    psds_mean = this_psd.mean(axis=0)
    diffs = this_psd - psds_mean
    # the distribution of power for each frequency bin is highly
    # skewed so we calculate std for values below and above average
    # separately - this is used for fill_between shade
    with warnings.catch_warnings():  # mean of empty slice
        warnings.simplefilter('ignore')
        spectrum_std = [
            [np.sqrt((d[d < 0] ** 2).mean(axis=0)) for d in diffs.T],
            [np.sqrt((d[d > 0] ** 2).mean(axis=0)) for d in diffs.T]]
    spectrum_std = np.array(spectrum_std) * num_std

    return psd_ylabel, psds_mean, spectrum_std


@verbose
def plot_ica_properties(ica, inst, picks=None, axes=None, dB=True,
                        plot_std=True, log_scale=False, topomap_args=None,
                        image_args=None, psd_args=None, figsize=None,
                        show=True, reject='auto', reject_by_annotation=True,
                        *, verbose=None):
    """Display component properties.

    Properties include the topography, epochs image, ERP/ERF, power
    spectrum, and epoch variance.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    inst : instance of Epochs or Raw
        The data to use in plotting properties.

        .. note::
           You can interactively cycle through topographic maps for different
           channel types by pressing :kbd:`T`.
    picks : int | list of int | slice | None
        Indices of the independent components (ICs) to visualize.
        If an integer, represents the index of the IC to pick.
        Multiple ICs can be selected using a list of int or a slice.
        The indices are 0-indexed, so ``picks=1`` will pick the second
        IC: ``ICA001``. ``None`` will pick the first 5 components.
    axes : list of Axes | None
        List of five matplotlib axes to use in plotting: [topomap_axis,
        image_axis, erp_axis, spectrum_axis, variance_axis]. If None a new
        figure with relevant axes is created. Defaults to None.
    dB : bool
        Whether to plot spectrum in dB. Defaults to True.
    plot_std : bool | float
        Whether to plot standard deviation/confidence intervals in ERP/ERF and
        spectrum plots.
        Defaults to True, which plots one standard deviation above/below for
        the spectrum. If set to float allows to control how many standard
        deviations are plotted for the spectrum. For example 2.5 will plot 2.5
        standard deviation above/below.
        For the ERP/ERF, by default, plot the 95 percent parametric confidence
        interval is calculated. To change this, use ``ci`` in ``ts_args`` in
        ``image_args`` (see below).
    log_scale : bool
        Whether to use a logarithmic frequency axis to plot the spectrum.
        Defaults to ``False``.

        .. note::
           You can interactively toggle this setting by pressing :kbd:`L`.

        .. versionadded:: 1.1
    topomap_args : dict | None
        Dictionary of arguments to ``plot_topomap``. If None, doesn't pass any
        additional arguments. Defaults to None.
    image_args : dict | None
        Dictionary of arguments to ``plot_epochs_image``. If None, doesn't pass
        any additional arguments. Defaults to None.
    psd_args : dict | None
        Dictionary of arguments to :meth:`~mne.Epochs.compute_psd`. If
        ``None``, doesn't pass any additional arguments. Defaults to ``None``.
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
    %(reject_by_annotation_raw)s

        .. versionadded:: 0.21.0
    %(verbose)s

    Returns
    -------
    fig : list
        List of matplotlib figures.

    Notes
    -----
    .. versionadded:: 0.13
    """
    return _fast_plot_ica_properties(ica, inst, picks=picks, axes=axes, dB=dB,
                                     plot_std=plot_std, log_scale=log_scale,
                                     topomap_args=topomap_args,
                                     image_args=image_args, psd_args=psd_args,
                                     figsize=figsize, show=show,
                                     reject=reject,
                                     reject_by_annotation=reject_by_annotation,
                                     verbose=verbose, precomputed_data=None)


def _fast_plot_ica_properties(ica, inst, picks=None, axes=None, dB=True,
                              plot_std=True, log_scale=False,
                              topomap_args=None, image_args=None,
                              psd_args=None, figsize=None, show=True,
                              reject='auto', precomputed_data=None,
                              reject_by_annotation=True, *, verbose=None):
    """Display component properties."""
    from ..preprocessing import ICA

    # input checks and defaults
    # -------------------------
    _validate_type(ica, ICA, "ica", "ICA")
    _validate_type(plot_std, (bool, 'numeric'), 'plot_std')
    if isinstance(plot_std, bool):
        num_std = 1. if plot_std else 0.
    else:
        plot_std = True
    num_std = float(plot_std)

    limit = min(5, ica.n_components_) if picks is None else ica.n_components_
    picks = _picks_to_idx(
        ica.n_components_, picks, picks_on="components"
    )[:limit]

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
    _validate_type(dB, (bool, None), "dB")
    _validate_type(log_scale, (bool, None), "log_scale")

    # calculations
    # ------------
    if isinstance(precomputed_data, tuple):
        kind, dropped_indices, epochs_src, data = precomputed_data
    else:
        kind, dropped_indices, epochs_src, data = _prepare_data_ica_properties(
            inst, ica, reject_by_annotation, reject)
    del reject
    ica_data = np.swapaxes(data[:, picks, :], 0, 1)
    dropped_src = ica_data

    # spectrum
    Nyquist = inst.info['sfreq'] / 2.
    lp = inst.info['lowpass']
    if 'fmax' not in psd_args:
        psd_args['fmax'] = min(lp * 1.25, Nyquist)
    plot_lowpass_edge = lp < Nyquist and (psd_args['fmax'] > lp)
    spectrum = epochs_src.compute_psd(picks=picks, **psd_args)
    # we've already restricted picks  ↑↑↑↑↑↑↑↑↑↑↑
    # in the spectrum object, so here we do picks=all  ↓↓↓↓↓↓↓↓↓↓↓
    psds, freqs = spectrum.get_data(return_freqs=True, picks='all', exclude=[])
    # we also pass exclude=[] so that when this is called by right-clicking in
    # a plot_sources() window on an ICA component name that has been marked as
    # bad, we can still get a plot of it.

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
            epoch_var, plot_lowpass_edge, epochs_src,
            set_title_and_labels, plot_std, psd_ylabel, spectrum_std,
            log_scale, topomap_args, image_args, fig, axes, kind,
            dropped_indices)
        all_fig.append(fig)

    plt_show(show)
    return all_fig


def _prepare_data_ica_properties(inst, ica, reject_by_annotation=True,
                                 reject='auto'):
    """Prepare Epochs sources to plot ICA properties.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    inst : instance of Epochs or Raw
        The data to use in plotting properties.
    reject_by_annotation : bool, optional
        [description], by default True
    reject : str, optional
        [description], by default 'auto'

    Returns
    -------
    kind : str
        "Segment" for BaseRaw and "Epochs" for BaseEpochs
    dropped_indices : list
        Dropped epochs indexes.
    epochs_src : instance of Epochs
        Segmented data of ICA sources.
    data : array of shape (n_epochs, n_ica_sources, n_times)
        A view on epochs ICA sources data.
    """
    from ..io.base import BaseRaw
    from ..io import RawArray
    from ..epochs import BaseEpochs

    _validate_type(inst, (BaseRaw, BaseEpochs), "inst", "Raw or Epochs")
    if isinstance(inst, BaseRaw):
        # when auto, delegate reject to the ica
        from ..epochs import make_fixed_length_epochs
        if reject == 'auto':
            reject = ica.reject_
        if reject is None:
            drop_inds = None
            dropped_indices = []
            # break up continuous signal into segments
            epochs_src = make_fixed_length_epochs(
                ica.get_sources(inst),
                duration=2,
                preload=True,
                reject_by_annotation=reject_by_annotation,
                proj=False,
                verbose=False)
        else:
            data = inst.get_data()
            data, drop_inds = _reject_data_segments(data, reject,
                                                    flat=None, decim=None,
                                                    info=inst.info,
                                                    tstep=2.0)
            inst_rejected = RawArray(data, inst.info)
            # break up continuous signal into segments
            epochs_src = make_fixed_length_epochs(
                ica.get_sources(inst_rejected),
                duration=2,
                preload=True,
                reject_by_annotation=reject_by_annotation,
                proj=False,
                verbose=False)
            # getting dropped epochs indexes
            dropped_indices = [(d[0] // len(epochs_src.times)) + 1
                               for d in drop_inds]
        kind = "Segment"
    else:
        drop_inds = None
        epochs_src = ica.get_sources(inst)
        dropped_indices = []
        kind = "Epochs"
    return kind, dropped_indices, epochs_src, epochs_src.get_data()


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

                if annot:
                    line_label += (' – ' + ', '.join(annot))  # Unicode en-dash
            exclude_labels.append(line_label)
        else:
            exclude_labels.append(None)
    label_props = [('k', '-') if lb is None else ('r', '-') for lb in
                   exclude_labels]
    styles = ['-', '--', ':', '-.']
    if labels is not None:
        # differentiate categories by linestyle and components by color
        col_lbs = [it for it in exclude_labels if it is not None]
        cmap = _get_cmap('tab10', len(col_lbs))

        unique_labels = set()
        for label in exclude_labels:
            if label is None:
                continue
            elif ' – ' in label:
                unique_labels.add(label.split(' – ')[1])
            else:
                unique_labels.add('')

        # Determine up to 4 different styles for n categories
        cat_styles = dict(zip(unique_labels,
                              map(lambda ux: styles[int(ux % len(styles))],
                                  range(len(unique_labels)))))
        for label_idx, label in enumerate(exclude_labels):
            if label is not None:
                color = cmap(col_lbs.index(label))
                if ' – ' in label:
                    label_name = label.split(' – ')[1]
                else:
                    label_name = ''
                style = cat_styles[label_name]
                label_props[label_idx] = (color, style)

    for exc_label, ii in zip(exclude_labels, picks):
        color, style = label_props[ii]
        # ensure traces of excluded components are plotted on top
        zorder = 2 if exc_label is None else 10
        lines.extend(ax.plot(times, evoked.data[ii].T, picker=True,
                             zorder=zorder, color=color, linestyle=style,
                             label=exc_label))
        lines[-1].set_pickradius(3.)

    ax.set(title=title, xlim=times[[0, -1]], xlabel='Time (ms)', ylabel='(NA)')
    if len(exclude) > 0:
        plt.legend(loc='best')
    tight_layout(fig=fig)

    texts.append(ax.text(0, 0, '', zorder=3,
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
                    title='ICA component scores', figsize=None,
                    n_cols=None, show=True):
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
        If list, should match the outer shape of ``scores``.
        If 'ecg' or 'eog', the ``labels_`` attributes will be looked up.
        Note that '/' is used internally for sublabels specifying ECG and
        EOG channels.
    axhline : float
        Draw horizontal line to e.g. visualize rejection threshold.
    title : str
        The figure title.
    figsize : tuple of int | None
        The figure size. If None it gets set automatically.
    n_cols : int | None
        Scores are plotted in a grid. This parameter controls how
        many to plot side by side before starting a new row. By
        default, a number will be chosen to make the grid as square as
        possible.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    my_range = np.arange(ica.n_components_)
    if exclude is None:
        exclude = ica.exclude
    exclude = np.unique(exclude)
    if not isinstance(scores[0], (list, np.ndarray)):
        scores = [scores]
    n_scores = len(scores)

    if n_cols is None:
        # prefer more rows.
        n_rows = int(np.ceil(np.sqrt(n_scores)))
        n_cols = (n_scores - 1) // n_rows + 1
    else:
        n_cols = min(n_scores, n_cols)
        n_rows = (n_scores - 1) // n_cols + 1

    if figsize is None:
        figsize = (6.4 * n_cols, 2.7 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    fig.suptitle(title)

    if labels == 'ecg':
        labels = [label for label in ica.labels_ if label.startswith('ecg/')]
        labels.sort(key=lambda label: label.split('/')[1])  # sort by index
        if len(labels) == 0:
            labels = [label for label in ica.labels_ if
                      label.startswith('ecg')]
    elif labels == 'eog':
        labels = [label for label in ica.labels_ if label.startswith('eog/')]
        labels.sort(key=lambda label: label.split('/')[1])  # sort by index
        if len(labels) == 0:
            labels = [label for label in ica.labels_ if
                      label.startswith('eog')]
    elif isinstance(labels, str):
        labels = [labels]
    elif labels is None:
        labels = (None,) * n_scores

    if len(labels) != n_scores:
        raise ValueError('Need as many labels (%i) as scores (%i)'
                         % (len(labels), n_scores))

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

    adjust_top = 0.8 if len(fig.axes) == 1 else 0.9
    fig.subplots_adjust(top=adjust_top)
    fig.canvas.draw()
    plt_show(show)
    return fig


@verbose
def plot_ica_overlay(ica, inst, exclude=None, picks=None, start=None,
                     stop=None, title=None, show=True, n_pca_components=None,
                     *, on_baseline='warn', verbose=None):
    """Overlay of raw and cleaned signals given the unmixing matrix.

    This method helps visualizing signal quality and artifact rejection.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    inst : instance of mne.io.Raw or mne.Evoked
        The signal to plot. If `~mne.io.Raw`, the raw data is displayed before
        and after cleaning. In a second panel, the cross-channel average will
        be displayed. Since dipolar sources will be canceled out, this
        representation is sensitive to artifacts. If `~mne.Evoked`, butterfly
        traces for signals before and after cleaning will be superimposed.
    exclude : array-like of int | None (default)
        The components marked for exclusion. If ``None`` (default), ICA.exclude
        will be used.
    %(picks_base)s all channels that were included during fitting.
    start, stop : float | None
       The first and last time point (in seconds) of the data to plot. If
       ``inst`` is a `~mne.io.Raw` object, ``start=None`` and ``stop=None``
       will be translated into ``start=0.`` and ``stop=3.``, respectively. For
       `~mne.Evoked`, ``None`` refers to the beginning and end of the evoked
       signal.
    %(title_none)s
    %(show)s
    %(n_pca_components_apply)s

        .. versionadded:: 0.22
    %(on_baseline_ica)s
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        The figure.
    """
    # avoid circular imports
    from ..io.base import BaseRaw
    from ..evoked import Evoked
    from ..preprocessing.ica import _check_start_stop

    if ica.current_fit == 'unfitted':
        raise RuntimeError('You need to fit the ICA first')

    _validate_type(inst, (BaseRaw, Evoked), "inst", "Raw or Evoked")
    if title is None:
        title = 'Signals before (red) and after (black) cleaning'
    picks = ica.ch_names if picks is None else picks
    picks = _picks_to_idx(inst.info, picks, exclude=())
    ch_types_used = inst.get_channel_types(picks=picks, unique=True)
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
                            start=start, stop=stop,
                            n_pca_components=n_pca_components)
        data_cln, _ = raw_cln[picks, start_compare:stop_compare]
        fig = _plot_ica_overlay_raw(data=data, data_cln=data_cln,
                                    times=times, title=title,
                                    ch_types_used=ch_types_used, show=show)
    else:
        assert isinstance(inst, Evoked)
        inst = inst.copy().crop(start, stop)
        if picks is not None:
            with inst.info._unlock():
                inst.info['comps'] = []  # can be safely disabled
            inst.pick_channels([inst.ch_names[p] for p in picks])
        evoked_cln = ica.apply(inst.copy(), exclude=exclude,
                               n_pca_components=n_pca_components,
                               on_baseline=on_baseline)
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
    evoked : instance of mne.Evoked
        The Evoked before IC rejection.
    evoked_cln : instance of mne.Evoked
        The Evoked after IC rejection.
    title : str | None
        The title of the figure.
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
    if title is None:
        title = 'Average signal before (red) and after (black) ICA'
    fig.suptitle(title)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    evoked.plot(axes=axes, show=False, time_unit='s', spatial_colors=False)
    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_color('r')
    fig.canvas.draw()
    evoked_cln.plot(axes=axes, show=False, time_unit='s', spatial_colors=False)
    tight_layout(fig=fig)

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()
    plt_show(show)
    return fig


def _plot_sources(ica, inst, picks, exclude, start, stop, show, title, block,
                  show_scrollbars, show_first_samp, time_format,
                  precompute, use_opengl, *, theme=None, overview_mode=None):
    """Plot the ICA components as a RawArray or EpochsArray."""
    from ._figure import _get_browser
    from .. import EpochsArray, BaseEpochs
    from ..io import RawArray, BaseRaw

    # handle defaults / check arg validity
    is_raw = isinstance(inst, BaseRaw)
    is_epo = isinstance(inst, BaseEpochs)
    sfreq = inst.info['sfreq']
    color = _handle_default('color', (0., 0., 0.))
    units = _handle_default('units', None)
    scalings = (_compute_scalings(None, inst) if is_raw else
                _handle_default('scalings_plot_raw'))
    scalings['misc'] = 5.
    scalings['whitened'] = 1.
    unit_scalings = _handle_default('scalings', None)

    # data
    if is_raw:
        data = ica._transform_raw(inst, 0, len(inst.times))[picks]
    else:
        data = ica._transform_epochs(inst, concatenate=True)[picks]

    # events
    if is_epo:
        event_id_rev = {v: k for k, v in inst.event_id.items()}
        event_nums = inst.events[:, 2]
        event_color_dict = _make_event_color_dict(None, inst.events,
                                                  inst.event_id)

    # channel properties / trace order / picks
    ch_names = list(ica._ica_names)  # copy
    ch_types = ['misc' for _ in picks]

    # add EOG/ECG channels if present
    eog_chs = pick_types(inst.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(inst.info, meg=False, ecg=True, ref_meg=False)
    for eog_idx in eog_chs:
        ch_names.append(inst.ch_names[eog_idx])
        ch_types.append('eog')
    for ecg_idx in ecg_chs:
        ch_names.append(inst.ch_names[ecg_idx])
        ch_types.append('ecg')
    extra_picks = np.concatenate((eog_chs, ecg_chs)).astype(int)
    if len(extra_picks):
        if is_raw:
            eog_ecg_data, _ = inst[extra_picks, :]
        else:
            eog_ecg_data = np.concatenate(inst.get_data(extra_picks), axis=1)
        data = np.append(data, eog_ecg_data, axis=0)
    picks = np.concatenate(
        (picks, ica.n_components_ + np.arange(len(extra_picks))))
    ch_order = np.arange(len(picks))
    n_channels = min([20, len(picks)])
    ch_names_picked = [ch_names[x] for x in picks]

    # create info
    info = create_info(ch_names_picked, sfreq, ch_types=ch_types)
    with info._unlock():
        info['meas_date'] = inst.info['meas_date']
    info['bads'] = [ch_names[x] for x in exclude if x in picks]
    if is_raw:
        inst_array = RawArray(data, info, inst.first_samp)
        inst_array.set_annotations(inst.annotations)
    else:
        data = data.reshape(-1, len(inst), len(inst.times)).swapaxes(0, 1)
        inst_array = EpochsArray(data, info)

    # handle time dimension
    start = 0 if start is None else start
    _last = inst.times[-1] if is_raw else len(inst.events)
    stop = min(start + 20, _last) if stop is None else stop
    first_time = inst._first_time if show_first_samp else 0
    if is_raw:
        duration = stop - start
        start += first_time
    else:
        n_epochs = stop - start
        total_epochs = len(inst)
        epoch_n_times = len(inst.times)
        n_epochs = min(n_epochs, total_epochs)
        n_times = total_epochs * epoch_n_times
        duration = n_epochs * epoch_n_times / sfreq
        event_times = (np.arange(total_epochs) * epoch_n_times
                       + inst.time_as_index(0)) / sfreq
        # NB: this includes start and end of data:
        boundary_times = np.arange(total_epochs + 1) * epoch_n_times / sfreq
    if duration <= 0:
        raise RuntimeError('Stop must be larger than start.')

    # misc
    bad_color = 'lightgray'
    title = 'ICA components' if title is None else title
    precompute = _handle_precompute(precompute)

    params = dict(inst=inst_array,
                  ica=ica,
                  ica_inst=inst,
                  info=info,
                  # channels and channel order
                  ch_names=np.array(ch_names_picked),
                  ch_types=np.array(ch_types),
                  ch_order=ch_order,
                  picks=picks,
                  n_channels=n_channels,
                  picks_data=list(),
                  # time
                  t_start=start if is_raw else boundary_times[start],
                  duration=duration,
                  n_times=inst.n_times if is_raw else n_times,
                  first_time=first_time,
                  time_format=time_format,
                  decim=1,
                  # events
                  event_times=None if is_raw else event_times,
                  # preprocessing
                  projs=list(),
                  projs_on=np.array([], dtype=bool),
                  apply_proj=False,
                  remove_dc=True,  # for EOG/ECG
                  filter_coefs=None,
                  filter_bounds=None,
                  noise_cov=None,
                  # scalings
                  scalings=scalings,
                  units=units,
                  unit_scalings=unit_scalings,
                  # colors
                  ch_color_bad=bad_color,
                  ch_color_dict=color,
                  # display
                  butterfly=False,
                  clipping=None,
                  scrollbars_visible=show_scrollbars,
                  scalebars_visible=False,
                  window_title=title,
                  precompute=precompute,
                  use_opengl=use_opengl,
                  theme=theme,
                  overview_mode=overview_mode,
                  )
    if is_epo:
        params.update(n_epochs=n_epochs,
                      boundary_times=boundary_times,
                      event_id_rev=event_id_rev,
                      event_color_dict=event_color_dict,
                      event_nums=event_nums,
                      epoch_color_bad=(1, 0, 0),
                      epoch_colors=None,
                      xlabel='Epoch number')

    fig = _get_browser(show=show, block=block, **params)

    return fig
