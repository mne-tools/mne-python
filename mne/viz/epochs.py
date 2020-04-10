"""Functions to plot epochs data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Stefan Repplinger <stefan.repplinger@ovgu.de>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

from collections import Counter
from functools import partial
from copy import deepcopy
import warnings

import numpy as np

from ..defaults import _handle_default
from ..utils import verbose, logger, warn, fill_doc, check_version
from ..io.meas_info import create_info, _validate_type

from ..io.pick import (pick_types, channel_type, _get_channel_types,
                       _picks_to_idx, _DATA_CH_TYPES_SPLIT,
                       _DATA_CH_TYPES_ORDER_DEFAULT, _VALID_CHANNEL_TYPES)
from ..time_frequency import psd_multitaper
from .utils import (tight_layout, figure_nobar, _toggle_proj, _toggle_options,
                    _prepare_mne_browse, _setup_vmin_vmax, _channels_changed,
                    _plot_raw_onscroll, _onclick_help, plt_show, _check_cov,
                    _compute_scalings, DraggableColorbar, _setup_cmap,
                    _handle_decim, _setup_plot_projector, _set_ax_label_style,
                    _set_title_multiple_electrodes, _make_combine_callable,
                    _get_figsize_from_config, _toggle_scrollbars,
                    _check_psd_fmax)
from .misc import _handle_event_colors


@fill_doc
def plot_epochs_image(epochs, picks=None, sigma=0., vmin=None,
                      vmax=None, colorbar=True, order=None, show=True,
                      units=None, scalings=None, cmap=None, fig=None,
                      axes=None, overlay_times=None, combine=None,
                      group_by=None, evoked=True, ts_args=None, title=None,
                      clear=False):
    """Plot Event Related Potential / Fields image.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    %(picks_good_data)s
        ``picks`` interacts with ``group_by`` and ``combine`` to determine the
        number of figures generated; see Notes.
    sigma : float
        The standard deviation of a Gaussian smoothing window applied along
        the epochs axis of the image. If 0, no smoothing is applied.
        Defaults to 0.
    vmin : None | float | callable
        The min value in the image (and the ER[P/F]). The unit is µV for
        EEG channels, fT for magnetometers and fT/cm for gradiometers.
        If vmin is None and multiple plots are returned, the limit is
        equalized within channel types.
        Hint: to specify the lower limit of the data, use
        ``vmin=lambda data: data.min()``.
    vmax : None | float | callable
        The max value in the image (and the ER[P/F]). The unit is µV for
        EEG channels, fT for magnetometers and fT/cm for gradiometers.
        If vmin is None and multiple plots are returned, the limit is
        equalized within channel types.
    colorbar : bool
        Display or not a colorbar.
    order : None | array of int | callable
        If not ``None``, order is used to reorder the epochs along the y-axis
        of the image. If it is an array of :class:`int`, its length should
        match the number of good epochs. If it is a callable it should accept
        two positional parameters (``times`` and ``data``, where
        ``data.shape == (len(good_epochs), len(times))``) and return an
        :class:`array <numpy.ndarray>` of indices that will sort ``data`` along
        its first axis.
    show : bool
        Show figure if True.
    units : dict | None
        The units of the channel types used for axes labels. If None,
        defaults to ``units=dict(eeg='µV', grad='fT/cm', mag='fT')``.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting.
        If None, defaults to ``scalings=dict(eeg=1e6, grad=1e13, mag=1e15,
        eog=1e6)``.
    cmap : None | colormap | (colormap, bool) | 'interactive'
        Colormap. If tuple, the first value indicates the colormap to use and
        the second value is a boolean defining interactivity. In interactive
        mode the colors are adjustable by clicking and dragging the colorbar
        with left and right mouse button. Left mouse button moves the scale up
        and down and right mouse button adjusts the range. Hitting space bar
        resets the scale. Up and down arrows can be used to change the
        colormap. If 'interactive', translates to ('RdBu_r', True).
        If None, "RdBu_r" is used, unless the data is all positive, in which
        case "Reds" is used.
    fig : Figure | None
        :class:`~matplotlib.figure.Figure` instance to draw the image to.
        Figure must contain the correct number of axes for drawing the epochs
        image, the evoked response, and a colorbar (depending on values of
        ``evoked`` and ``colorbar``). If ``None`` a new figure is created.
        Defaults to ``None``.
    axes : list of Axes | dict of list of Axes | None
        List of :class:`~matplotlib.axes.Axes` objects in which to draw the
        image, evoked response, and colorbar (in that order). Length of list
        must be 1, 2, or 3 (depending on values of ``colorbar`` and ``evoked``
        parameters). If a :class:`dict`, each entry must be a list of Axes
        objects with the same constraints as above. If both ``axes`` and
        ``group_by`` are dicts, their keys must match. Providing non-``None``
        values for both ``fig`` and ``axes``  results in an error. Defaults to
        ``None``.
    overlay_times : array_like, shape (n_epochs,) | None
        Times (in seconds) at which to draw a line on the corresponding row of
        the image (e.g., a reaction time associated with each epoch). Note that
        ``overlay_times`` should be ordered to correspond with the
        :class:`~mne.Epochs` object (i.e., ``overlay_times[0]`` corresponds to
        ``epochs[0]``, etc).
    %(combine)s
        If callable, the callable must accept one positional input (data of
        shape ``(n_epochs, n_channels, n_times)``) and return an
        :class:`array <numpy.ndarray>` of shape ``(n_epochs, n_times)``. For
        example::

            combine = lambda data: np.median(data, axis=1)

        If ``combine`` is ``None``, channels are combined by computing GFP,
        unless ``group_by`` is also ``None`` and ``picks`` is a list of
        specific channels (not channel types), in which case no combining is
        performed and each channel gets its own figure. See Notes for further
        details. Defaults to ``None``.
    group_by : None | dict
        Specifies which channels are aggregated into a single figure, with
        aggregation method determined by the ``combine`` parameter. If not
        ``None``, one :class:`~matplotlib.figure.Figure` is made per dict
        entry; the dict key will be used as the figure title and the dict
        values must be lists of picks (either channel names or integer indices
        of ``epochs.ch_names``). For example::

            group_by=dict(Left_ROI=[1, 2, 3, 4], Right_ROI=[5, 6, 7, 8])

        Note that within a dict entry all channels must have the same type.
        ``group_by`` interacts with ``picks`` and ``combine`` to determine the
        number of figures generated; see Notes. Defaults to ``None``.
    evoked : bool
        Draw the ER[P/F] below the image or not.
    ts_args : None | dict
        Arguments passed to a call to `plot_compare_evokeds` to style
        the evoked plot below the image. Defaults to an empty dictionary,
        meaning `plot_compare_evokeds` will be called with default parameters.
    title : None | str
        If :class:`str`, will be plotted as figure title. Otherwise, the
        title will indicate channel(s) or channel type being plotted. Defaults
        to ``None``.
    clear : bool
        Whether to clear the axes before plotting (if ``fig`` or ``axes`` are
        provided). Defaults to ``False``.

    Returns
    -------
    figs : list of Figure
        One figure per channel, channel type, or group, depending on values of
        ``picks``, ``group_by``, and ``combine``. See Notes.

    Notes
    -----
    You can control how channels are aggregated into one figure or plotted in
    separate figures through a combination of the ``picks``, ``group_by``, and
    ``combine`` parameters. If ``group_by`` is a :class:`dict`, the result is
    one :class:`~matplotlib.figure.Figure` per dictionary key (for any valid
    values of ``picks`` and ``combine``). If ``group_by`` is ``None``, the
    number and content of the figures generated depends on the values of
    ``picks`` and ``combine``, as summarized in this table:

    .. cssclass:: table-bordered
    .. rst-class:: midvalign

    +----------+----------------------------+------------+-------------------+
    | group_by | picks                      | combine    | result            |
    +==========+============================+============+===================+
    |          | None, int, list of int,    | None,      |                   |
    | dict     | ch_name, list of ch_names, | string, or | 1 figure per      |
    |          | ch_type, list of ch_types  | callable   | dict key          |
    +----------+----------------------------+------------+-------------------+
    |          | None,                      | None,      |                   |
    |          | ch_type,                   | string, or | 1 figure per      |
    |          | list of ch_types           | callable   | ch_type           |
    | None     +----------------------------+------------+-------------------+
    |          | int,                       | None       | 1 figure per pick |
    |          | ch_name,                   +------------+-------------------+
    |          | list of int,               | string or  | 1 figure          |
    |          | list of ch_names           | callable   |                   |
    +----------+----------------------------+------------+-------------------+
    """
    from scipy.ndimage import gaussian_filter1d
    from .. import EpochsArray

    _validate_type(group_by, (dict, None), 'group_by')

    units = _handle_default('units', units)
    scalings = _handle_default('scalings', scalings)
    if set(units) != set(scalings):
        raise ValueError('Scalings and units must have the same keys.')

    # is picks a channel type (or None)?
    picks, picked_types = _picks_to_idx(epochs.info, picks, return_kind=True)
    ch_types = _get_channel_types(epochs.info, picks)

    # `combine` defaults to 'gfp' unless picks are specific channels and
    # there was no group_by passed
    combine_given = combine is not None
    if combine is None and (group_by is not None or picked_types):
        combine = 'gfp'
    # convert `combine` into callable (if None or str)
    combine_func = _make_combine_callable(combine)

    # handle ts_args (params for the evoked time series)
    ts_args = dict() if ts_args is None else ts_args
    manual_ylims = 'ylim' in ts_args
    if combine is not None:
        ts_args['show_sensors'] = False
    vlines = [0] if (epochs.times[0] < 0 < epochs.times[-1]) else []
    ts_defaults = dict(colors={'cond': 'k'}, title='', show=False,
                       truncate_yaxis='auto', truncate_xaxis=False,
                       vlines=vlines, legend=False)
    ts_defaults.update(**ts_args)
    ts_args = ts_defaults.copy()

    # construct a group_by dict if one wasn't supplied
    if group_by is None:
        if picked_types:
            # one fig per ch_type
            group_by = {ch_type: picks[np.array(ch_types) == ch_type]
                        for ch_type in set(ch_types)
                        if ch_type in _DATA_CH_TYPES_SPLIT}
        elif combine is None:
            # one fig per pick
            group_by = {epochs.ch_names[pick]: [pick] for pick in picks}
        else:
            # one fig to rule them all
            ch_names = np.array(epochs.ch_names)[picks].tolist()
            key = _set_title_multiple_electrodes(None, combine, ch_names)
            group_by = {key: picks}
    else:
        group_by = deepcopy(group_by)
    # check for heterogeneous sensor type combinations / "combining" 1 channel
    for this_group, these_picks in group_by.items():
        this_ch_type = np.array(ch_types)[np.in1d(picks, these_picks)]
        if len(set(this_ch_type)) > 1:
            types = ', '.join(set(this_ch_type))
            raise ValueError('Cannot combine sensors of different types; "{}" '
                             'contains types {}.'.format(this_group, types))
        # now we know they're all the same type...
        group_by[this_group] = dict(picks=these_picks, ch_type=this_ch_type[0],
                                    title=title)

        # are they trying to combine a single channel?
        if len(these_picks) < 2 and combine_given:
            warn('Only one channel in group "{}"; cannot combine by method '
                 '"{}".'.format(this_group, combine))

    # check for compatible `fig` / `axes`; instantiate figs if needed; add
    # fig(s) and axes into group_by
    group_by = _validate_fig_and_axes(fig, axes, group_by, evoked, colorbar,
                                      clear=clear)

    # prepare images in advance to get consistent vmin/vmax.
    # At the same time, create a subsetted epochs object for each group
    data = epochs.get_data()
    vmin_vmax = {ch_type: dict(images=list(), norm=list())
                 for ch_type in set(ch_types)}
    for this_group, this_group_dict in group_by.items():
        these_picks = this_group_dict['picks']
        this_ch_type = this_group_dict['ch_type']
        this_ch_info = [epochs.info['chs'][n] for n in these_picks]
        these_ch_names = np.array(epochs.info['ch_names'])[these_picks]
        this_data = data[:, these_picks]
        # create subsetted epochs object
        this_info = create_info(sfreq=epochs.info['sfreq'],
                                ch_names=list(these_ch_names),
                                ch_types=[this_ch_type] * len(these_picks))
        this_info['chs'] = this_ch_info
        this_epochs = EpochsArray(this_data, this_info, tmin=epochs.times[0])
        # apply scalings (only to image, not epochs object), combine channels
        this_image = combine_func(this_data * scalings[this_ch_type])
        # handle `order`. NB: this can potentially yield different orderings
        # in each figure!
        this_image, overlay_times = _order_epochs(this_image, epochs.times,
                                                  order, overlay_times)
        this_norm = np.all(this_image > 0)
        # apply smoothing
        if sigma > 0.:
            this_image = gaussian_filter1d(this_image, sigma=sigma, axis=0,
                                           mode='nearest')
        # update the group_by and vmin_vmax dicts
        group_by[this_group].update(image=this_image, epochs=this_epochs,
                                    norm=this_norm)
        vmin_vmax[this_ch_type]['images'].append(this_image)
        vmin_vmax[this_ch_type]['norm'].append(this_norm)

    # compute overall vmin/vmax for images
    for ch_type, this_vmin_vmax_dict in vmin_vmax.items():
        image_list = this_vmin_vmax_dict['images']
        image_stack = np.stack(image_list)
        norm = all(this_vmin_vmax_dict['norm'])
        vmin_vmax[ch_type] = _setup_vmin_vmax(image_stack, vmin, vmax, norm)
    del image_stack, vmin, vmax

    # prepare to plot
    auto_ylims = {ch_type: [0., 0.] for ch_type in set(ch_types)}

    # plot
    for this_group, this_group_dict in group_by.items():
        this_ch_type = this_group_dict['ch_type']
        this_axes_dict = this_group_dict['axes']
        vmin, vmax = vmin_vmax[this_ch_type]

        # plot title
        if this_group_dict['title'] is None:
            title = _handle_default('titles').get(this_group, this_group)
            if isinstance(combine, str) and len(title):
                _comb = combine.upper() if combine == 'gfp' else combine
                _comb = 'std. dev.' if _comb == 'std' else _comb
                title += ' ({})'.format(_comb)

        # plot the image
        this_fig = _plot_epochs_image(
            this_group_dict['image'], epochs=this_group_dict['epochs'],
            picks=picks, colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
            style_axes=True, norm=this_group_dict['norm'],
            unit=units[this_ch_type], ax=this_axes_dict, show=False,
            title=title, combine=combine, combine_given=combine_given,
            overlay_times=overlay_times, evoked=evoked, ts_args=ts_args)
        group_by[this_group].update(fig=this_fig)

        # detect ylims across figures
        if evoked and not manual_ylims:
            # ensure get_ylim works properly
            this_axes_dict['evoked'].figure.canvas.draw_idle()
            this_bot, this_top = this_axes_dict['evoked'].get_ylim()
            this_min = min(this_bot, this_top)
            this_max = max(this_bot, this_top)
            curr_min, curr_max = auto_ylims[ch_type]
            auto_ylims[this_ch_type] = [min(curr_min, this_min),
                                        max(curr_max, this_max)]

    # equalize ylims across figures (does not adjust ticks)
    if evoked:
        for this_group_dict in group_by.values():
            ax = this_group_dict['axes']['evoked']
            ch_type = this_group_dict['ch_type']
            if not manual_ylims:
                args = auto_ylims[ch_type]
                func = max
                if 'invert_y' in ts_args:
                    args = args[::-1]
                    func = min
                ax.set_ylim(*args)
                yticks = np.array(ax.get_yticks())
                top_tick = func(yticks)
                ax.spines['left'].set_bounds(top_tick, args[0])
    plt_show(show)

    # impose deterministic order of returned objects
    return_order = np.array(sorted(group_by))
    are_ch_types = np.in1d(return_order, _VALID_CHANNEL_TYPES)
    if any(are_ch_types):
        return_order = np.concatenate((return_order[are_ch_types],
                                       return_order[~are_ch_types]))
    return [group_by[group]['fig'] for group in return_order]


def _validate_fig_and_axes(fig, axes, group_by, evoked, colorbar, clear=False):
    """Check user-provided fig/axes compatibility with plot_epochs_image."""
    from matplotlib.pyplot import figure, Axes, subplot2grid

    n_axes = 1 + int(evoked) + int(colorbar)
    ax_names = ('image', 'evoked', 'colorbar')
    ax_names = np.array(ax_names)[np.where([True, evoked, colorbar])]
    prefix = 'Since evoked={} and colorbar={}, '.format(evoked, colorbar)

    # got both fig and axes
    if fig is not None and axes is not None:
        raise ValueError('At least one of "fig" or "axes" must be None; got '
                         'fig={}, axes={}.'.format(fig, axes))

    # got fig=None and axes=None: make fig(s) and axes
    if fig is None and axes is None:
        axes = dict()
        colspan = 9 if colorbar else 10
        rowspan = 2 if evoked else 3
        shape = (3, 10)
        for this_group in group_by:
            this_fig = figure()
            this_fig.canvas.set_window_title(this_group)
            kwargs = dict()
            if check_version('matplotlib', '2.2'):
                kwargs['fig'] = this_fig  # unavailable on earlier mpl
            subplot2grid(shape, (0, 0), colspan=colspan, rowspan=rowspan,
                         **kwargs)
            if evoked:
                subplot2grid(shape, (2, 0), colspan=colspan, rowspan=1,
                             **kwargs)
            if colorbar:
                subplot2grid(shape, (0, 9), colspan=1, rowspan=rowspan,
                             **kwargs)
            axes[this_group] = this_fig.axes

    # got a Figure instance
    if fig is not None:
        # If we're re-plotting into a fig made by a previous call to
        # `plot_image`, be forgiving of presence/absence of sensor inset axis.
        if len(fig.axes) not in (n_axes, n_axes + 1):
            raise ValueError('{}"fig" must contain {} axes, got {}.'
                             ''.format(prefix, n_axes, len(fig.axes)))
        if len(list(group_by)) != 1:
            raise ValueError('When "fig" is not None, "group_by" can only '
                             'have one group (got {}: {}).'
                             .format(len(group_by), ', '.join(group_by)))
        key = list(group_by)[0]
        if clear:  # necessary if re-plotting into previous figure
            _ = [ax.clear() for ax in fig.axes]
            if len(fig.axes) > n_axes:  # get rid of sensor inset
                fig.axes[-1].remove()
            fig.canvas.set_window_title(key)
        axes = {key: fig.axes}

    # got an Axes instance, be forgiving (if evoked and colorbar are False)
    if isinstance(axes, Axes):
        axes = [axes]

    # got an ndarray; be forgiving
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()

    # got a list of axes, make it a dict
    if isinstance(axes, list):
        if len(axes) != n_axes:
            raise ValueError('{}"axes" must be length {}, got {}.'
                             ''.format(prefix, n_axes, len(axes)))
        # for list of axes to work, must be only one group
        if len(list(group_by)) != 1:
            raise ValueError('When axes is a list, can only plot one group '
                             '(got {} groups: {}).'
                             .format(len(group_by), ', '.join(group_by)))
        key = list(group_by)[0]
        axes = {key: axes}

    # got a dict of lists of axes, make it dict of dicts
    if isinstance(axes, dict):
        # in theory a user could pass a dict of axes but *NOT* pass a group_by
        # dict, but that is forbidden in the docstring so it shouldn't happen.
        # The next test could fail in that case because we've constructed a
        # group_by dict and the user won't have known what keys we chose.
        if set(axes) != set(group_by):
            raise ValueError('If "axes" is a dict its keys ({}) must match '
                             'the keys in "group_by" ({}).'
                             .format(list(axes), list(group_by)))
        for this_group, this_axes_list in axes.items():
            if len(this_axes_list) != n_axes:
                raise ValueError('{}each value in "axes" must be a list of {} '
                                 'axes, got {}.'.format(prefix, n_axes,
                                                        len(this_axes_list)))
            # NB: next line assumes all axes in each list are in same figure
            group_by[this_group]['fig'] = this_axes_list[0].get_figure()
            group_by[this_group]['axes'] = {key: axis for key, axis in
                                            zip(ax_names, this_axes_list)}
    return group_by


def _order_epochs(data, times, order=None, overlay_times=None):
    """Sort epochs image data (2D). Helper for plot_epochs_image."""
    n_epochs = len(data)

    if overlay_times is not None:
        if len(overlay_times) != n_epochs:
            raise ValueError('size of overlay_times parameter ({}) does not '
                             'match the number of epochs ({}).'
                             .format(len(overlay_times), n_epochs))
        overlay_times = np.array(overlay_times)
        times_min = np.min(overlay_times)
        times_max = np.max(overlay_times)
        if ((times_min < times[0]) or (times_max > times[-1])):
            warn('Some values in overlay_times fall outside of the epochs '
                 'time interval (between %s s and %s s)'
                 % (times[0], times[-1]))

    if callable(order):
        order = order(times, data)

    if order is not None:
        if len(order) != n_epochs:
            raise ValueError('If order is a {}, its length ({}) must match '
                             'the length of the data ({}).'
                             .format(type(order).__name__, len(order),
                                     n_epochs))
        order = np.asarray(order)
        data = data[order]
        if overlay_times is not None:
            overlay_times = overlay_times[order]

    return data, overlay_times


def _plot_epochs_image(image, style_axes=True, epochs=None, picks=None,
                       vmin=None, vmax=None, colorbar=False, show=False,
                       unit=None, cmap=None, ax=None, overlay_times=None,
                       title=None, evoked=False, ts_args=None, combine=None,
                       combine_given=False, norm=False):
    """Plot epochs image. Helper function for plot_epochs_image."""
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    tmin = epochs.times[0]
    tmax = epochs.times[-1]

    ax_im = ax['image']
    fig = ax_im.get_figure()

    # draw the image
    cmap = _setup_cmap(cmap, norm=norm)
    n_epochs = len(image)
    extent = [1e3 * tmin, 1e3 * tmax, 0, n_epochs]
    im = ax_im.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap[0], aspect='auto',
                      origin='lower', interpolation='nearest', extent=extent)

    # optional things
    if style_axes:
        ax_im.set_title(title)
        ax_im.set_ylabel('Epochs')
        ax_im.axis('auto')
        ax_im.axis('tight')
        ax_im.axvline(0, color='k', linewidth=1, linestyle='--')

    if overlay_times is not None:
        ax_im.plot(1e3 * overlay_times, 0.5 + np.arange(n_epochs), 'k',
                   linewidth=2)
        ax_im.set_xlim(1e3 * tmin, 1e3 * tmax)

    # draw the evoked
    if evoked:
        from . import plot_compare_evokeds
        pass_combine = (combine if combine_given else None)
        _picks = [0] if len(picks) == 1 else None  # prevent applying GFP
        plot_compare_evokeds({'cond': list(epochs.iter_evoked(copy=False))},
                             picks=_picks, axes=ax['evoked'],
                             combine=pass_combine, **ts_args)
        ax['evoked'].set_xlim(tmin, tmax)  # don't multiply by 1e3 here
        ax_im.set_xticks([])

    # draw the colorbar
    if colorbar:
        from matplotlib.pyplot import colorbar as cbar
        this_colorbar = cbar(im, cax=ax['colorbar'])
        this_colorbar.ax.set_ylabel(unit, rotation=270, labelpad=12)
        if cmap[1]:
            ax_im.CB = DraggableColorbar(this_colorbar, im)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            tight_layout(fig=fig)

    # finish
    plt_show(show)
    return fig


def plot_drop_log(drop_log, threshold=0, n_max_plot=20, subject='Unknown subj',
                  color=(0.8, 0.8, 0.8), width=0.8, ignore=('IGNORED',),
                  show=True):
    """Show the channel stats based on a drop_log from Epochs.

    Parameters
    ----------
    drop_log : list of list
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
    ignore : list
        The drop reasons to ignore.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from ..epochs import _drop_log_stats
    percent = _drop_log_stats(drop_log, ignore)
    if percent < threshold:
        logger.info('Percent dropped epochs < supplied threshold; not '
                    'plotting drop log.')
        return
    scores = Counter([ch for d in drop_log for ch in d if ch not in ignore])
    ch_names = np.array(list(scores.keys()))
    counts = np.array(list(scores.values()))
    # init figure, handle easy case (no drops)
    fig, ax = plt.subplots()
    ax.set_title('{}: {:.1f}%'.format(subject, percent))
    if len(ch_names) == 0:
        ax.text(0.5, 0.5, 'No drops', ha='center', fontsize=14)
        return fig
    # count epochs that aren't fully caught by `ignore`
    n_used = sum([any(ch not in ignore for ch in d) or len(d) == 0
                  for d in drop_log])
    # calc plot values
    n_bars = min(n_max_plot, len(ch_names))
    x = np.arange(n_bars)
    y = 100 * counts / n_used
    order = np.flipud(np.argsort(y))
    ax.bar(x, y[order[:n_bars]], color=color, width=width, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names[order[:n_bars]], rotation=45, size=10,
                       horizontalalignment='right')
    ax.set_ylabel('% of epochs rejected')
    ax.grid(axis='y')
    tight_layout(pad=1, fig=fig)
    plt_show(show)
    return fig


def _draw_epochs_axes(epoch_idx, good_ch_idx, bad_ch_idx, data, times, axes,
                      title_str, axes_handler):
    """Handle drawing epochs axes."""
    this = axes_handler[0]
    for ii, data_, ax in zip(epoch_idx, data, axes):
        for l, d in zip(ax.lines, data_[good_ch_idx]):
            l.set_data(times, d)
        if bad_ch_idx is not None:
            bad_lines = [ax.lines[k] for k in bad_ch_idx]
            for l, d in zip(bad_lines, data_[bad_ch_idx]):
                l.set_data(times, d)
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks(list())
        ax.set_xticks(list())
        if vars(ax)[this]['reject'] is True:
            #  memorizing reject
            for l in ax.lines:
                l.set_color((0.8, 0.8, 0.8))
            ax.get_figure().canvas.draw()
        else:
            #  forgetting previous reject
            for k in axes_handler:
                if k == this:
                    continue
                if vars(ax).get(k, {}).get('reject', None) is True:
                    for l in ax.lines[:len(good_ch_idx)]:
                        l.set_color('k')
                    if bad_ch_idx is not None:
                        for l in ax.lines[-len(bad_ch_idx):]:
                            l.set_color('r')
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Handle epochs navigation click."""
    import matplotlib.pyplot as plt
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            p['epochs'].drop(p['reject_idx'])
        plt.close(p['fig'])
        plt.close(event.inaxes.get_figure())

    if here is not None:
        p['idx_handler'].rotate(here)
        p['axes_handler'].rotate(here)
        this_idx = p['idx_handler'][0]
        _draw_epochs_axes(this_idx, p['good_ch_idx'], p['bad_ch_idx'],
                          p['data'][this_idx],
                          p['times'], p['axes'], p['title_str'],
                          p['axes_handler'])
        # XXX don't ask me why
        p['axes'][0].get_figure().canvas.draw()


def _epochs_axes_onclick(event, params):
    """Handle epochs axes click."""
    reject_color = (0.8, 0.8, 0.8)
    ax = event.inaxes
    if event.inaxes is None:
        return
    p = params
    here = vars(ax)[p['axes_handler'][0]]
    if here.get('reject', None) is False:
        idx = here['idx']
        if idx not in p['reject_idx']:
            p['reject_idx'].append(idx)
            for l in ax.lines:
                l.set_color(reject_color)
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            good_lines = [ax.lines[k] for k in p['good_ch_idx']]
            for l in good_lines:
                l.set_color('k')
            if p['bad_ch_idx'] is not None:
                bad_lines = ax.lines[-len(p['bad_ch_idx']):]
                for l in bad_lines:
                    l.set_color('r')
            here['reject'] = False
    ax.get_figure().canvas.draw()


@fill_doc
def plot_epochs(epochs, picks=None, scalings=None, n_epochs=20, n_channels=20,
                title=None, events=None, event_colors=None, order=None,
                show=True, block=False, decim='auto', noise_cov=None,
                butterfly=False, show_scrollbars=True, epoch_colors=None,
                event_id=None):
    """Visualize epochs.

    Bad epochs can be marked with a left click on top of the epoch. Bad
    channels can be selected by clicking the channel name on the left side of
    the main axes. Calling this function drops all the selected bad epochs as
    well as bad epochs marked beforehand with rejection parameters.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object.
    %(picks_good_data)s
    scalings : dict | 'auto' | None
        Scaling factors for the traces. If any fields in scalings are 'auto',
        the scaling factor is set to match the 99.5th percentile of a subset of
        the corresponding data. If scalings == 'auto', all scalings fields are
        set to 'auto'. If any fields are 'auto' and data is not preloaded,
        a subset of epochs up to 100mb will be loaded. If None, defaults to::

            dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                 emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4,
                 whitened=10.)

    n_epochs : int
        The number of epochs per view. Defaults to 20.
    n_channels : int
        The number of channels per view. Defaults to 20.
    title : str | None
        The title of the window. If None, epochs name will be displayed.
        Defaults to None.
    events : None, array, shape (n_events, 3)
        Events to show with vertical bars. If events are provided, the epoch
        numbers are not shown to prevent overlap. You can toggle epoch
        numbering through options (press 'o' key). You can use
        :func:`mne.viz.plot_events` as a legend for the colors. By default, the
        coloring scheme is the same.

        .. warning::  If the epochs have been resampled, the events no longer
            align with the data.

        .. versionadded:: 0.14.0
    event_colors : None, dict
        Dictionary of event_id value and its associated color. If None,
        colors are automatically drawn from a default list (cycled through if
        number of events longer than list of default colors). Uses the same
        coloring scheme as :func:`mne.viz.plot_events`.

        .. versionadded:: 0.14.0
    order : array of str | None
        Order in which to plot channel types.

        .. versionadded:: 0.18.0
    show : bool
        Show figure if True. Defaults to True.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on an epoch.
        Defaults to False.
    decim : int | 'auto'
        Amount to decimate the data during display for speed purposes.
        You should only decimate if the data are sufficiently low-passed,
        otherwise aliasing can occur. The 'auto' mode (default) uses
        the decimation that results in a sampling rate at least three times
        larger than ``info['lowpass']`` (e.g., a 40 Hz lowpass will result in
        at least a 120 Hz displayed sample rate).

        .. versionadded:: 0.15.0
    noise_cov : instance of Covariance | str | None
        Noise covariance used to whiten the data while plotting.
        Whitened data channels are scaled by ``scalings['whitened']``,
        and their channel names are shown in italic.
        Can be a string to load a covariance from disk.
        See also :meth:`mne.Evoked.plot_white` for additional inspection
        of noise covariance properties when whitening evoked data.
        For data processed with SSS, the effective dependence between
        magnetometers and gradiometers may introduce differences in scaling,
        consider using :meth:`mne.Evoked.plot_white`.

        .. versionadded:: 0.16.0
    butterfly : bool
        Whether to directly call the butterfly view.

        .. versionadded:: 0.18.0
    %(show_scrollbars)s
    epoch_colors : list of (n_epochs) list (of n_channels) | None
        Colors to use for individual epochs. If None, use default colors.
    event_id : dict | None
        Dictionary of event labels (e.g. 'aud_l') as keys and associated event
        integers as values. Useful when ``events`` contains event numbers not
        present in ``epochs.event_id`` (e.g., because of event subselection).
        Values in ``event_id`` will take precedence over those in
        ``epochs.event_id`` when there are overlapping keys.

        .. versionadded:: 0.20

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.

    Notes
    -----
    The arrow keys (up/down/left/right) can be used to navigate between
    channels and epochs and the scaling can be adjusted with - and + (or =)
    keys, but this depends on the backend matplotlib is configured to use
    (e.g., mpl.use(``TkAgg``) should work). Full screen mode can be toggled
    with f11 key. The amount of epochs and channels per view can be adjusted
    with home/end and page down/page up keys. These can also be set through
    options dialog by pressing ``o`` key. ``h`` key plots a histogram of
    peak-to-peak values along with the used rejection thresholds. Butterfly
    plot can be toggled with ``b`` key. Right mouse click adds a vertical line
    to the plot. Click 'help' button at bottom left corner of the plotter to
    view all the options.

    .. versionadded:: 0.10.0
    """
    epochs.drop_bad()
    scalings = _compute_scalings(scalings, epochs)
    scalings = _handle_default('scalings_plot_raw', scalings)
    decim, data_picks = _handle_decim(epochs.info.copy(), decim, None)
    projs = epochs.info['projs']
    noise_cov = _check_cov(noise_cov, epochs.info)

    params = dict(epochs=epochs, info=epochs.info.copy(), t_start=0.,
                  bad_color=(0.8, 0.8, 0.8), histogram=None, decim=decim,
                  data_picks=data_picks, noise_cov=noise_cov,
                  use_noise_cov=noise_cov is not None,
                  show_scrollbars=show_scrollbars,
                  epoch_colors=epoch_colors)
    params['label_click_fun'] = partial(_pick_bad_channels, params=params)
    _prepare_mne_browse_epochs(params, projs, n_channels, n_epochs, scalings,
                               title, picks, events=events, order=order,
                               event_colors=event_colors, butterfly=butterfly,
                               event_id=event_id)
    _prepare_projectors(params)

    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)

    return params['fig']


@verbose
def plot_epochs_psd(epochs, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                    proj=False, bandwidth=None, adaptive=False, low_bias=True,
                    normalization='length', picks=None, ax=None, color='black',
                    xscale='linear', area_mode='std', area_alpha=0.33,
                    dB=True, estimate='auto', show=True, n_jobs=1,
                    average=False, line_alpha=None, spatial_colors=True,
                    sphere=None, verbose=None):
    """%(plot_psd_doc)s.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object.
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    tmin : float | None
        Start time to consider.
    tmax : float | None
        End time to consider.
    proj : bool
        Apply projection.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
    %(plot_psd_picks_good_data)s
    ax : instance of Axes | None
        Axes to plot into. If None, axes will be created.
    %(plot_psd_color)s
    %(plot_psd_xscale)s
    %(plot_psd_area_mode)s
    %(plot_psd_area_alpha)s
    %(plot_psd_dB)s
    %(plot_psd_estimate)s
    %(show)s
    %(n_jobs)s
    %(plot_psd_average)s
    %(plot_psd_line_alpha)s
    %(plot_psd_spatial_colors)s
    %(topomap_sphere_auto)s
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure with frequency spectra of the data channels.
    """
    from .utils import _set_psd_plot_params, _plot_psd
    fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
        make_label, xlabels_list = \
        _set_psd_plot_params(epochs.info, proj, picks, ax, area_mode)
    _check_psd_fmax(epochs, fmax)
    del ax
    psd_list = list()
    for picks in picks_list:
        # Multitaper used for epochs instead of Welch, because Welch chunks
        # the data; epoched data are by nature already chunked, however.
        psd, freqs = psd_multitaper(epochs, picks=picks, fmin=fmin,
                                    fmax=fmax, tmin=tmin, tmax=tmax,
                                    bandwidth=bandwidth, adaptive=adaptive,
                                    low_bias=low_bias,
                                    normalization=normalization, proj=proj,
                                    n_jobs=n_jobs)
        psd_list.append(np.mean(psd, axis=0))

    fig = _plot_psd(epochs, fig, freqs, psd_list, picks_list, titles_list,
                    units_list, scalings_list, ax_list, make_label, color,
                    area_mode, area_alpha, dB, estimate, average,
                    spatial_colors, xscale, line_alpha, sphere, xlabels_list)
    plt_show(show)
    return fig


def _prepare_mne_browse_epochs(params, projs, n_channels, n_epochs, scalings,
                               title, picks, events=None, event_colors=None,
                               order=None, butterfly=False, info=None,
                               event_id=None):
    """Set up the mne_browse_epochs window."""
    import matplotlib as mpl
    from matplotlib.collections import LineCollection
    from matplotlib.colors import colorConverter
    epochs = params['epochs']
    info = info or epochs.info
    orig_epoch_times, epochs_events = epochs.times, epochs.events
    name = epochs._name
    del epochs

    # Reorganize channels
    picks = _picks_to_idx(info, picks)
    picks = sorted(picks)
    # channel type string for every channel
    types = _get_channel_types(info, picks)
    # list of unique channel types
    unique_types = _get_channel_types(info, unique=True)
    if order is None:
        order = _DATA_CH_TYPES_ORDER_DEFAULT
    inds = [pick_idx for order_type in order
            for pick_idx, ch_type in zip(picks, types)
            if order_type == ch_type]
    if len(unique_types) > len(order):
        ch_missing = unique_types - set(order)
        raise RuntimeError('%s are in picks but not in order.'
                           ' Please specify all channel types picked.' %
                           (str(ch_missing)))
    types = sorted(types, key=order.index)
    if not len(inds) == len(picks):
        raise RuntimeError('Some channels not classified. Please'
                           ' check your picks')
    ch_names = [params['info']['ch_names'][idx] for idx in inds]
    _validate_type(params['epoch_colors'], (list, None), 'epoch_colors')
    if params['epoch_colors'] is not None:
        if len(params['epoch_colors']) != len(params['epochs'].events):
            raise ValueError('epoch_colors must be list of len(epochs.events).'
                             ' Got %s' % len(params['epoch_colors']))
        for epoch_idx in range(len(params['epoch_colors'])):
            these_colors = params['epoch_colors'][epoch_idx]
            _validate_type(these_colors, list,
                           'epoch_colors[%s]' % (epoch_idx,))
            if len(these_colors) != len(params['epochs'].ch_names):
                raise ValueError('epoch_colors for the %dth epoch '
                                 'has length %d, expected %d.'
                                 % (epoch_idx, len(these_colors),
                                    len(params['epochs'].ch_names)))
            params['epoch_colors'][epoch_idx] = \
                [these_colors[idx] for idx in inds]

    # set up plotting
    n_epochs = min(n_epochs, len(epochs_events))
    duration = len(orig_epoch_times) * n_epochs
    n_channels = min(n_channels, len(picks))
    if title is None:
        title = name
        if title is None or len(title) == 0:
            title = ''
    color = _handle_default('color', None)

    figsize = _get_figsize_from_config()
    params['fig'] = figure_nobar(facecolor='w', figsize=figsize, dpi=80)
    params['fig'].canvas.set_window_title(title or 'Epochs')
    _prepare_mne_browse(params, xlabel='Epochs')
    ax = params['ax']
    ax_hscroll = params['ax_hscroll']
    ax_vscroll = params['ax_vscroll']

    # add secondary x axis for annotations / event labels
    ax2 = ax.twiny()
    ax2.set_zorder(-1)
    ax2.set_axes_locator(ax.get_axes_locator())
    # set axis lims
    ax.axis([0, duration, 0, 200])
    ax2.axis([0, duration, 0, 200])

    # populate vertical and horizontal scrollbars
    ax_vscroll.add_patch(mpl.patches.Rectangle((0, 0), 1, len(picks),
                                               facecolor='w', zorder=3))
    for ci in range(len(picks)):
        if ch_names[ci] in params['info']['bads']:
            this_color = params['bad_color']
        else:
            this_color = color[types[ci]]
        ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                   facecolor=this_color,
                                                   edgecolor=this_color,
                                                   zorder=4))

    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       edgecolor='w', facecolor='w', zorder=5)
    ax_vscroll.add_patch(vsel_patch)

    ax_vscroll.set_ylim(len(types), 0)
    ax_vscroll.set_title('Ch.')

    # populate colors list
    type_colors = [colorConverter.to_rgba(color[c]) for c in types]
    colors = list()
    for color_idx in range(len(type_colors)):
        colors.append([type_colors[color_idx]] * len(epochs_events))
    lines = list()
    n_times = len(orig_epoch_times)

    for ch_idx in range(n_channels):
        if len(colors) - 1 < ch_idx:
            break
        lc = LineCollection(list(), antialiased=True, linewidths=0.5,
                            zorder=3, picker=True)
        lc.set_pickradius(3.)
        ax.add_collection(lc)
        lines.append(lc)

    data = np.zeros((params['info']['nchan'],
                     len(orig_epoch_times) * n_epochs))

    ylim = (25., 0.)  # Hardcoded 25 because butterfly has max 5 rows (5*5=25).
    # make shells for plotting traces
    offset = ylim[0] / n_channels
    offsets = np.arange(n_channels) * offset + (offset / 2.)

    times = np.arange(len(orig_epoch_times) * len(epochs_events))
    epoch_times = np.arange(0, len(times), n_times)

    ax.set_yticks(offsets)
    ax.set_ylim(ylim)
    ticks = epoch_times + 0.5 * n_times
    ax.set_xticks(ticks)
    ax2.set_xticks(ticks[:n_epochs])
    labels = list(range(0, len(ticks)))  # epoch numbers
    ax.set_xticklabels(labels)
    xlim = epoch_times[-1] + len(orig_epoch_times)
    ax_hscroll.set_xlim(0, xlim)
    vertline_t = ax_hscroll.text(0, 1, '', color='y', va='bottom', ha='right')

    # fit horizontal scroll bar ticks
    hscroll_ticks = np.arange(0, xlim, xlim / 7.0)
    hscroll_ticks = np.append(hscroll_ticks, epoch_times[-1])
    hticks = list()
    for tick in hscroll_ticks:
        hticks.append(epoch_times.flat[np.abs(epoch_times - tick).argmin()])
    hlabels = [x // n_times for x in hticks]
    ax_hscroll.set_xticks(hticks)
    ax_hscroll.set_xticklabels(hlabels)

    for epoch_idx in range(len(epoch_times)):
        ax_hscroll.add_patch(mpl.patches.Rectangle((epoch_idx * n_times, 0),
                                                   n_times, 1, facecolor='w',
                                                   edgecolor='w', alpha=0.6))
    hsel_patch = mpl.patches.Rectangle((0, 0), duration, 1,
                                       edgecolor='k',
                                       facecolor=(0.75, 0.75, 0.75),
                                       alpha=0.25, linewidth=1, clip_on=False)
    ax_hscroll.add_patch(hsel_patch)
    text = ax.text(0, 0, 'blank', zorder=3, verticalalignment='baseline',
                   ha='left', fontweight='bold')
    text.set_visible(False)

    epoch_nr = True
    if events is not None:
        event_set = set(events[:, 2])
        ev_id = params['epochs'].event_id if event_id is None else event_id
        event_colors = _handle_event_colors(event_colors, event_set, ev_id)
        epoch_nr = False  # epoch number off by default to avoid overlap
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)

    params.update({'ax': ax,
                   'ax2': ax2,
                   'ax_hscroll': ax_hscroll,
                   'ax_vscroll': ax_vscroll,
                   'vsel_patch': vsel_patch,
                   'hsel_patch': hsel_patch,
                   'lines': lines,  # vertical lines for segmentation
                   'projs': projs,
                   'ch_names': ch_names,
                   'n_channels': n_channels,
                   'n_epochs': n_epochs,
                   'scalings': scalings,
                   'duration': duration,
                   'ch_start': 0,
                   'colors': colors,
                   'def_colors': type_colors,  # don't change at runtime
                   'picks': picks,
                   'bads': np.array(list(), dtype=int),
                   'data': data,
                   'times': times,
                   'epoch_times': epoch_times,
                   'offsets': offsets,
                   'labels': labels,
                   'scale_factor': 1.0,
                   'butterfly_scale': 1.0,
                   'fig_proj': None,
                   'types': np.array(types),
                   'inds': inds,
                   'vert_lines': list(),
                   'vertline_t': vertline_t,
                   'butterfly': butterfly,
                   'text': text,
                   'fig_options': None,
                   'settings': [True, True, epoch_nr, True],
                   'image_plot': None,
                   'events': events,
                   'event_colors': event_colors,
                   'ev_lines': list(),
                   'ev_texts': list(),
                   'ann': list(),  # list for butterfly view annotations
                   'order': order,
                   'ch_types': unique_types})

    params['plot_fun'] = partial(_plot_traces, params=params)

    # Plot epoch_colors
    if params['epoch_colors'] is not None:
        for epoch_idx, epoch_color in enumerate(params['epoch_colors']):
            for ch_idx in range(len(params['ch_names'])):
                if epoch_color[ch_idx] is not None:
                    params['colors'][ch_idx][epoch_idx] = \
                        colorConverter.to_rgba(epoch_color[ch_idx])

            # plot on horizontal patch if all colors are same
            if epoch_color.count(epoch_color[0]) == len(epoch_color):
                params['ax_hscroll'].patches[epoch_idx].set_color(
                    epoch_color[0])
                params['ax_hscroll'].patches[epoch_idx].set_zorder(3)
                params['ax_hscroll'].patches[epoch_idx].set_edgecolor('w')

    # callbacks
    callback_scroll = partial(_plot_onscroll, params=params)
    params['fig'].canvas.mpl_connect('scroll_event', callback_scroll)
    callback_click = partial(_mouse_click, params=params)
    params['fig'].canvas.mpl_connect('button_press_event', callback_click)
    callback_key = partial(_plot_onkey, params=params)
    params['fig'].canvas.mpl_connect('key_press_event', callback_key)
    params['fig'].canvas.mpl_connect('pick_event', partial(_onpick,
                                                           params=params))
    params['callback_key'] = callback_key
    # Draw event lines for the first time.
    _plot_vert_lines(params)


def _prepare_projectors(params):
    """Set up the projectors for epochs browser."""
    import matplotlib as mpl
    epochs = params['epochs']
    projs = params['projs']
    if len(projs) > 0 and not epochs.proj:
        # set up proj button
        ax_button = params['fig'].add_axes(params['proj_button_pos'])
        ax_button.set_axes_locator(params['proj_button_locator'])
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
        params['opt_button'] = opt_button
        params['apply_proj'] = epochs.proj

    # As here code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_epochs_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    callback_proj('none')


def _plot_traces(params):
    """Plot concatenated epochs."""
    params['text'].set_visible(False)
    ax = params['ax']
    butterfly = params['butterfly']
    offsets = params['offsets']
    lines = params['lines']
    epochs = params['epochs']

    if butterfly:
        ch_start = 0
        n_channels = len(params['picks'])
        data = params['data'] * params['butterfly_scale']
        _prepare_butterfly(params)
    else:
        ch_start = params['ch_start']
        n_channels = params['n_channels']
        data = params['data'] * params['scale_factor']

    n_times = len(epochs.times)
    tick_list = list()
    start_idx = int(params['t_start'] / n_times)
    end = params['t_start'] + params['duration']
    end_idx = int(end / n_times)
    xlabels = params['labels'][start_idx:]
    event_ids = params['epochs'].events[:, 2]
    params['ax2'].set_xticklabels(event_ids[start_idx:])
    ax.set_xticklabels(xlabels)
    ylabels = ax.yaxis.get_ticklabels()
    # do the plotting
    for line_idx in range(n_channels):
        ch_idx = line_idx + ch_start
        if line_idx >= len(lines):
            break
        elif ch_idx < len(params['ch_names']):
            if butterfly:
                # determine offsets for signal traces
                ch_type = params['types'][ch_idx]
                chan_types_split = sorted(set(params['ch_types']) &
                                          set(_DATA_CH_TYPES_SPLIT),
                                          key=params['order'].index)
                ylim = ax.get_ylim()[0]
                ticks = np.arange(
                    0, ylim, ylim / (4 * max(len(chan_types_split), 1)))
                offset_pos = np.arange(2, len(chan_types_split) * 4, 4)
                if ch_type in chan_types_split:
                    offset = ticks[offset_pos[chan_types_split.index(ch_type)]]
                else:
                    lines[line_idx].set_segments(list())
                    offset = None
            else:
                tick_list += [params['ch_names'][ch_idx]]
                offset = offsets[line_idx]
            if offset is None:
                continue

            if params['inds'][ch_idx] in params['data_picks']:
                this_decim = params['decim']
            else:
                this_decim = 1
            this_data = data[ch_idx]

            # subtraction here gets correct orientation for flipped ylim
            ydata = offset - this_data
            xdata = params['times'][:params['duration']]
            num_epochs = np.min([params['n_epochs'], len(epochs.events)])

            segments = np.split(np.array((xdata, ydata)).T, num_epochs)
            segments = [segment[::this_decim] for segment in segments]

            ch_name = params['ch_names'][ch_idx]
            if ch_name in params['info']['bads']:
                if not butterfly:
                    this_color = params['bad_color']
                    ylabels[line_idx].set_color(this_color)
                this_color = np.tile((params['bad_color']), (num_epochs, 1))
                for bad_idx in params['bads']:
                    if bad_idx < start_idx or bad_idx >= end_idx:
                        continue
                    this_color[bad_idx - start_idx] = (1., 0., 0.)
                lines[line_idx].set_zorder(2)
            else:
                this_color = params['colors'][ch_idx][start_idx:end_idx]
                lines[line_idx].set_zorder(3)
                if not butterfly:
                    ylabels[line_idx].set_color('black')
            lines[line_idx].set_segments(segments)
            lines[line_idx].set_color(this_color)
        else:
            lines[line_idx].set_segments(list())

    # finalize plot
    ax.set_xlim(params['times'][0], params['times'][0] + params['duration'],
                False)
    params['ax2'].set_xlim(params['times'][0],
                           params['times'][0] + params['duration'], False)
    if butterfly:
        # compute labels for ticks surrounding the trace offset
        factor = -1. / params['butterfly_scale']
        scalings_default = _handle_default('scalings')
        chan_types_split = sorted(set(params['types']) &
                                  set(_DATA_CH_TYPES_SPLIT),
                                  key=params['order'].index)
        ylim = ax.get_ylim()[0]
        ticks = np.arange(
            0, ylim + 1, ylim / (4 * max(len(chan_types_split), 1)))
        offset_pos = np.arange(2, (len(chan_types_split) * 4) + 1, 4)
        ax.set_yticks(ticks)
        labels = [''] * 20
        labels = [0 if idx in range(2, len(labels), 4) else label
                  for idx, label in enumerate(labels)]
        for idx_chan, chan_type in enumerate(chan_types_split):
            tick_top, tick_bottom = 1 + idx_chan * 4, 3 + idx_chan * 4
            offset = ticks[offset_pos[idx_chan]]
            for tick_pos in [tick_top, tick_bottom]:
                tickoffset_diff = ticks[tick_pos] - offset
                labels[tick_pos] = (tickoffset_diff *
                                    params['scalings'][chan_type] *
                                    factor * scalings_default[chan_type])
        # Heuristic to turn floats to ints where possible (e.g. -500.0 to -500)
        for li, label in enumerate(labels):
            if isinstance(label, float) and float(str(label)) != round(label):
                labels[li] = round(label, 2)
        ax.set_yticklabels(labels, fontsize=12, color='black')
    else:
        ax.set_yticklabels(tick_list, fontsize=12)
        _set_ax_label_style(ax, params)

    if params['events'] is not None:  # vertical lines for events.
        _ = _draw_event_lines(params)

    params['vsel_patch'].set_y(ch_start)
    params['fig'].canvas.draw()
    # XXX This is a hack to make sure this figure gets drawn last
    # so that when matplotlib goes to calculate bounds we don't get a
    # CGContextRef error on the MacOSX backend :(
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _plot_update_epochs_proj(params, bools=None):
    """Deal with proj changed."""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    epochs = params['epochs']
    n_epochs = params['n_epochs']
    params['projector'], params['whitened_ch_names'] = _setup_plot_projector(
        params['info'], params['noise_cov'], True, params['use_noise_cov'])
    start = int(params['t_start'] / len(epochs.times))
    end = start + n_epochs
    if epochs.preload:
        data = np.concatenate(epochs.get_data()[start:end], axis=1)
    else:
        # this is faster than epochs.get_data()[start:end] when not preloaded
        data = np.concatenate(epochs[start:end].get_data(), axis=1)

    if params['projector'] is not None:
        data = np.dot(params['projector'], data)
    types = params['types']
    for pick, ind in enumerate(params['inds']):
        ch_name = params['info']['ch_names'][ind]
        if ch_name in params['whitened_ch_names'] and \
                ch_name not in params['info']['bads']:
            norm = params['scalings']['whitened']
        else:
            norm = params['scalings'][types[pick]]
        params['data'][pick] = data[ind] / norm
    params['plot_fun']()


def _handle_picks(epochs):
    """Handle picks."""
    if any('ICA' in k for k in epochs.ch_names):
        picks = pick_types(epochs.info, misc=True, ref_meg=False,
                           exclude=[])
    else:
        picks = pick_types(epochs.info, meg=True, eeg=True, eog=True, ecg=True,
                           seeg=True, ecog=True, ref_meg=False, fnirs=True,
                           exclude=[])
    return picks


def _plot_window(value, params):
    """Deal with horizontal shift of the viewport."""
    max_times = len(params['times']) - params['duration']
    if value > max_times:
        value = len(params['times']) - params['duration']
    if value < 0:
        value = 0
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)
        params['plot_update_proj_callback'](params)


def _plot_vert_lines(params):
    """Plot vertical lines."""
    ax = params['ax']
    while len(ax.lines) > 0:
        ax.lines.pop()
    params['vert_lines'] = list()
    params['ev_lines'] = list()
    params['vertline_t'].set_text('')
    epochs = params['epochs']

    # draw event lines
    tzero_already_drawn = False
    if params['events'] is not None:
        tzero_already_drawn = _draw_event_lines(params)
    # draw zero lines
    if params['settings'][3] and not tzero_already_drawn:
        t_zero = np.where(epochs.times == 0.)[0]
        if len(t_zero) == 1:  # not True if tmin > 0
            for event_idx in range(len(epochs.events)):
                pos = [event_idx * len(epochs.times) + t_zero[0],
                       event_idx * len(epochs.times) + t_zero[0]]
                ax.plot(pos, ax.get_ylim(), 'g', zorder=0, alpha=0.4)
    # draw boundaries between epochs
    for epoch_idx in range(len(epochs.events)):
        pos = [epoch_idx * len(epochs.times), epoch_idx * len(epochs.times)]
        ax.plot(pos, ax.get_ylim(), color='black', linestyle='--', zorder=2)


def _pick_bad_epochs(event, params):
    """Select / drop bad epochs."""
    if 'ica' in params:
        pos = (event.xdata, event.ydata)
        _pick_bad_channels(pos, params)
        return
    n_times = len(params['epochs'].times)
    start_idx = int(params['t_start'] / n_times)
    xdata = event.xdata
    xlim = event.inaxes.get_xlim()
    epoch_idx = start_idx + int(xdata / (xlim[1] / params['n_epochs']))
    total_epochs = len(params['epochs'].events)
    if epoch_idx > total_epochs - 1:
        return
    # remove bad epoch
    if epoch_idx in params['bads']:
        params['bads'] = params['bads'][(params['bads'] != epoch_idx)]
        for ch_idx in range(len(params['ch_names'])):
            params['colors'][ch_idx][epoch_idx] = params['def_colors'][ch_idx]
        params['ax_hscroll'].patches[epoch_idx].set_color('w')
        params['ax_hscroll'].patches[epoch_idx].set_zorder(2)
        params['plot_fun']()
        return
    # add bad epoch
    params['bads'] = np.append(params['bads'], epoch_idx)
    params['ax_hscroll'].patches[epoch_idx].set_color((1., 0., 0., 1.))
    params['ax_hscroll'].patches[epoch_idx].set_zorder(3)
    params['ax_hscroll'].patches[epoch_idx].set_edgecolor('w')
    for ch_idx in range(len(params['ch_names'])):
        params['colors'][ch_idx][epoch_idx] = (1., 0., 0., 1.)
    params['plot_fun']()


def _pick_bad_channels(pos, params):
    """Select bad channels."""
    text, ch_idx = _label2idx(params, pos)
    if text is None:
        return
    if text in params['info']['bads']:
        while text in params['info']['bads']:
            params['info']['bads'].remove(text)
        color = params['def_colors'][ch_idx]
        params['ax_vscroll'].patches[ch_idx + 1].set_color(color)
    else:
        params['info']['bads'].append(text)
        color = params['bad_color']
        params['ax_vscroll'].patches[ch_idx + 1].set_color(color)
    if 'ica' in params:
        params['plot_fun']()
    else:
        params['plot_update_proj_callback'](params)


def _plot_onscroll(event, params):
    """Handle scroll events."""
    if event.key == 'control':
        if event.step < 0:
            event.key = '-'
        else:
            event.key = '+'
        _plot_onkey(event, params)
        return
    if params['butterfly']:
        return
    _plot_raw_onscroll(event, params, len(params['ch_names']))


def _mouse_click(event, params):
    """Handle mouse click events."""
    from matplotlib.pyplot import fignum_exists
    if event.inaxes is None:
        if params['butterfly'] or not params['settings'][0]:
            return
        ax = params['ax']
        ylim = ax.get_ylim()
        pos = ax.transData.inverted().transform((event.x, event.y))
        if pos[0] > 0 or pos[1] < 0 or pos[1] > ylim[0]:
            return
        if event.button == 1:  # left click
            params['label_click_fun'](pos)
        elif event.button == 3:  # right click
            if 'ica' not in params:
                _, ch_idx = _label2idx(params, pos)
                if ch_idx is None:
                    return
                if channel_type(params['info'], ch_idx) not in ['mag', 'grad',
                                                                'eeg', 'eog']:
                    logger.info('Event related fields / potentials only '
                                'available for MEG and EEG channels.')
                    return
                # check if the figure was already closed
                if (params['image_plot'] is not None and
                        not fignum_exists(params['image_plot'].number)):
                    params['image_plot'] = None
                fig = plot_epochs_image(params['epochs'],
                                        picks=params['inds'][ch_idx],
                                        fig=params['image_plot'],
                                        clear=True)[0]
                params['image_plot'] = fig
    elif event.button == 1:  # left click
        # vertical scroll bar changed
        if event.inaxes == params['ax_vscroll']:
            if params['butterfly']:
                return
            # Don't let scrollbar go outside vertical scrollbar limits
            # XXX: floating point exception on some machines if this happens.
            ch_start = min(
                max(int(event.ydata) - params['n_channels'] // 2, 0),
                len(params['ch_names']) - params['n_channels'])

            if params['ch_start'] != ch_start:
                params['ch_start'] = ch_start
                params['plot_fun']()
        # horizontal scroll bar changed
        elif event.inaxes == params['ax_hscroll']:
            # find the closest epoch time
            times = params['epoch_times']
            offset = 0.5 * params['n_epochs'] * len(params['epochs'].times)
            xdata = times.flat[np.abs(times - (event.xdata - offset)).argmin()]
            _plot_window(xdata, params)
        # main axes
        elif event.inaxes == params['ax']:
            _pick_bad_epochs(event, params)

    elif event.inaxes == params['ax'] and event.button == 2:  # middle click
        params['fig'].canvas.draw()
        if params['fig_proj'] is not None:
            params['fig_proj'].canvas.draw()
    elif event.inaxes == params['ax'] and event.button == 3:  # right click
        n_times = len(params['epochs'].times)
        xdata = int(event.xdata % n_times)
        prev_xdata = 0
        if len(params['vert_lines']) > 0:
            prev_xdata = params['vert_lines'][0][0].get_data()[0][0]
            while len(params['vert_lines']) > 0:
                params['ax'].lines.remove(params['vert_lines'][0][0])
                params['vert_lines'].pop(0)
        if prev_xdata == xdata:  # lines removed
            params['vertline_t'].set_text('')
            params['plot_fun']()
            return
        ylim = params['ax'].get_ylim()
        for epoch_idx in range(params['n_epochs']):  # plot lines
            pos = [epoch_idx * n_times + xdata, epoch_idx * n_times + xdata]
            params['vert_lines'].append(params['ax'].plot(pos, ylim, 'y',
                                                          zorder=5))
        params['vertline_t'].set_text('%0.3f' % params['epochs'].times[xdata])
        params['plot_fun']()


def _plot_onkey(event, params):
    """Handle key presses."""
    import matplotlib.pyplot as plt
    if event.key == 'down':
        if params['butterfly']:
            return
        params['ch_start'] += params['n_channels']
        _channels_changed(params, len(params['ch_names']))
    elif event.key == 'up':
        if params['butterfly']:
            return
        params['ch_start'] -= params['n_channels']
        _channels_changed(params, len(params['ch_names']))
    elif event.key == 'left':
        sample = params['t_start'] - params['duration']
        sample = np.max([0, sample])
        _plot_window(sample, params)
    elif event.key == 'right':
        sample = params['t_start'] + params['duration']
        sample = np.min([sample, params['times'][-1] - params['duration']])
        times = params['epoch_times']
        xdata = times.flat[np.abs(times - sample).argmin()]
        _plot_window(xdata, params)
    elif event.key == '-':
        if params['butterfly']:
            params['butterfly_scale'] /= 1.1
        else:
            params['scale_factor'] /= 1.1
        params['plot_fun']()
    elif event.key in ['+', '=']:
        if params['butterfly']:
            params['butterfly_scale'] *= 1.1
        else:
            params['scale_factor'] *= 1.1
        params['plot_fun']()
    elif event.key == 'f11':
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    elif event.key == 'pagedown':
        if params['n_channels'] == 1 or params['butterfly']:
            return
        n_channels = params['n_channels'] - 1
        ylim = params['ax'].get_ylim()
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['n_channels'] = n_channels
        params['ax'].collections.pop()
        params['ax'].set_yticks(params['offsets'])
        params['lines'].pop()
        params['vsel_patch'].set_height(n_channels)
        params['plot_fun']()
    elif event.key == 'pageup':
        if params['butterfly']:
            return
        from matplotlib.collections import LineCollection
        n_channels = params['n_channels'] + 1
        ylim = params['ax'].get_ylim()
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['n_channels'] = n_channels
        lc = LineCollection(list(), antialiased=True, linewidths=0.5,
                            zorder=3, picker=True)
        lc.set_pickradius(3.)
        params['ax'].add_collection(lc)
        params['ax'].set_yticks(params['offsets'])
        params['lines'].append(lc)
        params['vsel_patch'].set_height(n_channels)
        params['plot_fun']()
    elif event.key == 'home':
        n_epochs = params['n_epochs'] - 1
        if n_epochs <= 0:
            return
        n_times = len(params['epochs'].times)
        ticks = params['epoch_times'] + 0.5 * n_times
        params['ax2'].set_xticks(ticks[:n_epochs])
        params['n_epochs'] = n_epochs
        params['duration'] -= n_times
        params['hsel_patch'].set_width(params['duration'])
        params['data'] = params['data'][:, :-n_times]
        params['plot_update_proj_callback'](params)
    elif event.key == 'end':
        n_epochs = params['n_epochs'] + 1
        n_times = len(params['epochs'].times)
        if n_times * n_epochs > len(params['times']):
            return
        ticks = params['epoch_times'] + 0.5 * n_times
        params['ax2'].set_xticks(ticks[:n_epochs])
        params['n_epochs'] = n_epochs
        if len(params['vert_lines']) > 0:
            ax = params['ax']
            pos = params['vert_lines'][0][0].get_data()[0] + params['duration']
            params['vert_lines'].append(ax.plot(pos, ax.get_ylim(), 'y',
                                                zorder=4))
        params['duration'] += n_times
        if params['t_start'] + params['duration'] > len(params['times']):
            params['t_start'] -= n_times
            params['hsel_patch'].set_x(params['t_start'])
        params['hsel_patch'].set_width(params['duration'])
        params['data'] = np.zeros((len(params['data']), params['duration']))
        params['plot_update_proj_callback'](params)
    elif event.key == 'b':
        params['butterfly'] = not params['butterfly']
        if params['fig_options'] is not None:
            plt.close(params['fig_options'])
            params['fig_options'] = None
        _prepare_butterfly(params)
        params['plot_fun']()
    elif event.key == 'w':
        params['use_noise_cov'] = not params['use_noise_cov']
        _plot_update_epochs_proj(params)
        _plot_traces(params)
    elif event.key == 'o':
        if not params['butterfly']:
            _open_options(params)
    elif event.key == 'h':
        _plot_histogram(params)
    elif event.key == '?':
        _onclick_help(event, params)
    elif event.key == 'escape':
        plt.close(params['fig'])
    elif event.key == 'z':
        # zen mode: remove scrollbars and buttons
        _toggle_scrollbars(params)


def _prepare_butterfly(params):
    """Set up butterfly plot."""
    from matplotlib.collections import LineCollection
    import matplotlib as mpl
    if params['butterfly']:
        units = _handle_default('units')
        chan_types = sorted(set(params['types']) & set(params['order']),
                            key=params['order'].index)
        if len(chan_types) < 1:
            return
        params['ax_vscroll'].set_visible(False)
        ax = params['ax']
        labels = ax.yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(True)
        offsets = np.arange(0, ax.get_ylim()[0],
                            ax.get_ylim()[0] / (4 * len(chan_types)))
        ticks = offsets
        ticks = [ticks[x] if x < len(ticks) else 0 for x in range(20)]
        ax.set_yticks(ticks)
        used_types = 0
        params['offsets'] = [ticks[2]]
        # checking which annotations are displayed and removing them
        ann = params['ann']
        annotations = [child for child in params['ax2'].get_children()
                       if isinstance(child, mpl.text.Annotation)]
        for annote in annotations:
            annote.remove()
        ann[:] = list()
        assert len(params['ann']) == 0
        titles = _handle_default('titles')
        for chan_type in chan_types:
            unit = units[chan_type]
            pos = (0, 1 - (ticks[2 + 4 * used_types] / ax.get_ylim()[0]))
            ann.append(params['ax2'].annotate(
                '%s (%s)' % (titles[chan_type], unit), xy=pos,
                xytext=(-70, 0), ha='left', size=12, va='center',
                xycoords='axes fraction', rotation=90,
                textcoords='offset points'))
            used_types += 1
        while len(params['lines']) < len(params['picks']):
            lc = LineCollection(list(), antialiased=True, linewidths=.5,
                                zorder=3, picker=True)
            lc.set_pickradius(3.)
            ax.add_collection(lc)
            params['lines'].append(lc)
    else:  # change back to default view
        labels = params['ax'].yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][0])
        params['ax_vscroll'].set_visible(True)
        while len(params['ax2'].texts) > 0:
            params['ax2'].texts.pop()
        n_channels = params['n_channels']
        while len(params['lines']) > n_channels:
            params['ax'].collections.pop()
            params['lines'].pop()
        ylim = (25., 0.)
        params['ax'].set_ylim(ylim)
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['ax'].set_yticks(params['offsets'])


def _onpick(event, params):
    """Add a channel name on click."""
    if event.mouseevent.button != 2 or not params['butterfly']:
        return  # text label added with a middle mouse button
    lidx = np.where([l is event.artist for l in params['lines']])[0][0]
    text = params['text']
    text.set_x(event.mouseevent.xdata)
    text.set_y(event.mouseevent.ydata)
    text.set_text(params['ch_names'][lidx])
    text.set_visible(True)
    # do NOT redraw here, since for butterfly plots hundreds of lines could
    # potentially be picked -- use _mouse_click (happens once per click)
    # to do the drawing


def _close_event(event, params):
    """Drop selected bad epochs (called on closing of the plot)."""
    params['epochs'].drop(params['bads'])
    params['epochs'].info['bads'] = params['info']['bads']
    logger.info('Channels marked as bad: %s' % params['epochs'].info['bads'])


def _update_channels_epochs(event, params):
    """Change the amount of channels and epochs per view."""
    from matplotlib.collections import LineCollection
    # Channels
    n_channels = int(np.around(params['channel_slider'].val))
    offset = params['ax'].get_ylim()[0] / n_channels
    params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
    while len(params['lines']) > n_channels:
        params['ax'].collections.pop()
        params['lines'].pop()
    while len(params['lines']) < n_channels:
        lc = LineCollection(list(), linewidths=0.5, antialiased=True,
                            zorder=3, picker=True)
        lc.set_pickradius(3.)
        params['ax'].add_collection(lc)
        params['lines'].append(lc)
    params['ax'].set_yticks(params['offsets'])
    params['vsel_patch'].set_height(n_channels)
    params['n_channels'] = n_channels

    # Epochs
    n_epochs = int(np.around(params['epoch_slider'].val))
    n_times = len(params['epochs'].times)
    ticks = params['epoch_times'] + 0.5 * n_times
    params['ax2'].set_xticks(ticks[:n_epochs])
    params['n_epochs'] = n_epochs
    params['duration'] = n_times * n_epochs
    params['hsel_patch'].set_width(params['duration'])
    params['data'] = np.zeros((len(params['data']), params['duration']))
    if params['t_start'] + n_times * n_epochs > len(params['times']):
        params['t_start'] = len(params['times']) - n_times * n_epochs
        params['hsel_patch'].set_x(params['t_start'])
    params['plot_update_proj_callback'](params)


def _toggle_labels(label, params):
    """Toggle axis labels."""
    if label == 'Channel names visible':
        params['settings'][0] = not params['settings'][0]
        labels = params['ax'].yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][0])
    elif label == 'Event-id visible':
        params['settings'][1] = not params['settings'][1]
        labels = params['ax2'].xaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][1])
    elif label == 'Epoch-id visible':
        params['settings'][2] = not params['settings'][2]
        labels = params['ax'].xaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][2])
    elif label == 'Zeroline visible':
        params['settings'][3] = not params['settings'][3]
        _plot_vert_lines(params)
    params['fig'].canvas.draw()
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _open_options(params):
    """Open the option window."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if params['fig_options'] is not None:
        # turn off options dialog
        plt.close(params['fig_options'])
        params['fig_options'] = None
        return
    width = 10
    height = 3
    fig_options = figure_nobar(figsize=(width, height), dpi=80)
    fig_options.canvas.set_window_title('View settings')
    params['fig_options'] = fig_options
    ax_channels = plt.axes([0.15, 0.1, 0.65, 0.1])
    ax_epochs = plt.axes([0.15, 0.25, 0.65, 0.1])
    ax_button = plt.axes([0.85, 0.1, 0.1, 0.25])
    ax_check = plt.axes([0.15, 0.4, 0.4, 0.55])
    plt.axis('off')
    params['update_button'] = mpl.widgets.Button(ax_button, 'Update')
    params['channel_slider'] = mpl.widgets.Slider(ax_channels, 'Channels', 1,
                                                  len(params['ch_names']),
                                                  valfmt='%0.0f',
                                                  valinit=params['n_channels'])
    params['epoch_slider'] = mpl.widgets.Slider(ax_epochs, 'Epochs', 1,
                                                len(params['epoch_times']),
                                                valfmt='%0.0f',
                                                valinit=params['n_epochs'])
    params['checkbox'] = mpl.widgets.CheckButtons(ax_check,
                                                  ['Channel names visible',
                                                   'Event-id visible',
                                                   'Epoch-id visible',
                                                   'Zeroline visible'],
                                                  actives=params['settings'])
    update = partial(_update_channels_epochs, params=params)
    params['update_button'].on_clicked(update)
    labels_callback = partial(_toggle_labels, params=params)
    params['checkbox'].on_clicked(labels_callback)
    close_callback = partial(_settings_closed, params=params)
    params['fig_options'].canvas.mpl_connect('close_event', close_callback)
    try:
        params['fig_options'].canvas.draw()
        params['fig_options'].show(warn=False)
        if params['fig_proj'] is not None:
            params['fig_proj'].canvas.draw()
    except Exception:
        pass


def _settings_closed(events, params):
    """Handle close event from settings dialog."""
    params['fig_options'] = None


def _plot_histogram(params):
    """Plot histogram of peak-to-peak values."""
    import matplotlib.pyplot as plt
    epochs = params['epochs']
    p2p = np.ptp(epochs.get_data(), axis=2)
    types = list()
    data = list()
    if 'eeg' in params['types']:
        eegs = np.array([p2p.T[i] for i,
                         x in enumerate(params['types']) if x == 'eeg'])
        data.append(eegs.ravel())
        types.append('eeg')
    if 'mag' in params['types']:
        mags = np.array([p2p.T[i] for i,
                         x in enumerate(params['types']) if x == 'mag'])
        data.append(mags.ravel())
        types.append('mag')
    if 'grad' in params['types']:
        grads = np.array([p2p.T[i] for i,
                          x in enumerate(params['types']) if x == 'grad'])
        data.append(grads.ravel())
        types.append('grad')
    params['histogram'] = plt.figure()
    scalings = _handle_default('scalings')
    units = _handle_default('units')
    titles = _handle_default('titles')
    colors = _handle_default('color')
    for idx in range(len(types)):
        ax = plt.subplot(len(types), 1, idx + 1)
        plt.xlabel(units[types[idx]])
        plt.ylabel('Count')
        color = colors[types[idx]]
        rej = None
        if epochs.reject is not None and types[idx] in epochs.reject:
            rej = epochs.reject[types[idx]] * scalings[types[idx]]
            rng = [0., rej * 1.1]
        else:
            rng = None
        plt.hist(data[idx] * scalings[types[idx]], bins=100, color=color,
                 range=rng)
        if rej is not None:
            ax.plot((rej, rej), (0, ax.get_ylim()[1]), color='r')
        plt.title(titles[types[idx]])
    params['histogram'].suptitle('Peak-to-peak histogram', y=0.99)
    params['histogram'].subplots_adjust(hspace=0.6)
    try:
        params['histogram'].show(warn=False)
    except Exception:
        pass
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()
    plt.tight_layout(h_pad=0.7, pad=2)


def _label2idx(params, pos):
    """Handle click on labels (returns channel name and idx)."""
    labels = params['ax'].yaxis.get_ticklabels()
    offsets = np.array(params['offsets']) + params['offsets'][0]
    line_idx = np.searchsorted(offsets, pos[1])
    text = labels[line_idx].get_text()
    if len(text) == 0:
        return None, None
    ch_idx = params['ch_start'] + line_idx
    return text, ch_idx


def _draw_event_lines(params):
    """Draw event lines."""
    includes_tzero = False
    epochs = params['epochs']
    n_times = len(epochs.times)
    start_idx = int(params['t_start'] / n_times)
    color = params['event_colors']
    ax = params['ax']
    for ev_line in params['ev_lines']:
        ax.lines.remove(ev_line)  # clear the view first
    for ev_text in params['ev_texts']:
        ax.texts.remove(ev_text)
    params['ev_texts'] = list()
    params['ev_lines'] = list()
    t_zero = np.where(epochs.times == 0.)[0]  # idx of 0s
    if len(t_zero) == 0:
        t_zero = epochs.times[0] * -1 * epochs.info['sfreq']  # if tmin > 0
    end = params['n_epochs'] + start_idx
    samp_times = params['events'][:, 0]
    for idx, event in enumerate(epochs.events[start_idx:end]):
        event_mask = ((event[0] - t_zero < samp_times) &
                      (samp_times < event[0] + n_times - t_zero))
        for ev in params['events'][event_mask]:
            if ev[0] == event[0]:
                includes_tzero = True
            pos = [idx * n_times + ev[0] - event[0] + t_zero,
                   idx * n_times + ev[0] - event[0] + t_zero]
            kwargs = {} if ev[2] not in color else {'color': color[ev[2]]}
            params['ev_lines'].append(ax.plot(pos, ax.get_ylim(),
                                              zorder=3, **kwargs)[0])
            params['ev_texts'].append(ax.text(pos[0], ax.get_ylim()[0],
                                              ev[2], color=color[ev[2]],
                                              ha='center', va='top'))
    return includes_tzero
