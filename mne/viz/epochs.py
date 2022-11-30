"""Functions to plot epochs data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Stefan Repplinger <stefan.repplinger@ovgu.de>
#          Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

from collections import Counter
from copy import deepcopy
import warnings

import numpy as np

from .raw import _setup_channel_selections
from ..fixes import _sharex
from ..defaults import _handle_default
from ..utils import legacy, verbose, logger, warn, fill_doc, _check_option
from ..io.meas_info import create_info, _validate_type

from ..io.pick import (_get_channel_types, _picks_to_idx, _DATA_CH_TYPES_SPLIT,
                       _VALID_CHANNEL_TYPES)
from .utils import (tight_layout, _setup_vmin_vmax, plt_show,
                    _check_cov, _handle_precompute,
                    _compute_scalings, DraggableColorbar, _setup_cmap,
                    _handle_decim, _set_title_multiple_electrodes,
                    _make_combine_callable, _set_window_title,
                    _make_event_color_dict, _get_channel_plotting_order)


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
        Arguments passed to a call to `~mne.viz.plot_compare_evokeds` to style
        the evoked plot below the image. Defaults to an empty dictionary,
        meaning `~mne.viz.plot_compare_evokeds` will be called with default
        parameters.
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
                       truncate_yaxis=False, truncate_xaxis=False,
                       vlines=vlines, legend=False)
    ts_defaults.update(**ts_args)
    ts_args = ts_defaults.copy()

    # construct a group_by dict if one wasn't supplied
    if group_by is None:
        if picked_types:
            # one fig per ch_type
            group_by = {ch_type: picks[np.array(ch_types) == ch_type]
                        for ch_type in set(ch_types)
                        if ch_type in _DATA_CH_TYPES_SPLIT + ('ref_meg',)}
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
        with this_info._unlock():
            this_info['chs'] = this_ch_info
        this_epochs = EpochsArray(this_data, this_info, tmin=epochs.times[0])
        # apply scalings (only to image, not epochs object), combine channels
        this_image = combine_func(this_data * scalings[this_ch_type])
        # handle `order`. NB: this can potentially yield different orderings
        # in each figure!
        this_image, _overlay_times = _order_epochs(this_image, epochs.times,
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
                title += f' ({_comb})'

        # plot the image
        this_fig = _plot_epochs_image(
            this_group_dict['image'], epochs=this_group_dict['epochs'],
            picks=picks, colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
            style_axes=True, norm=this_group_dict['norm'],
            unit=units[this_ch_type], ax=this_axes_dict, show=False,
            title=title, combine=combine, combine_given=combine_given,
            overlay_times=_overlay_times, evoked=evoked, ts_args=ts_args)
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
                if 'invert_y' in ts_args:
                    args = args[::-1]
                ax.set_ylim(*args)
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
            _set_window_title(this_fig, this_group)
            subplot2grid(shape, (0, 0), colspan=colspan, rowspan=rowspan,
                         fig=this_fig)
            if evoked:
                subplot2grid(shape, (2, 0), colspan=colspan, rowspan=1,
                             fig=this_fig)
            if colorbar:
                subplot2grid(shape, (0, 9), colspan=1, rowspan=rowspan,
                             fig=this_fig)
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
            _set_window_title(fig, key)
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
            raise ValueError(
                f'size of overlay_times parameter ({len(overlay_times)}) does '
                f'not match the number of epochs ({n_epochs}).')
        overlay_times = np.array(overlay_times)
        times_min = np.min(overlay_times)
        times_max = np.max(overlay_times)
        if (times_min < times[0]) or (times_max > times[-1]):
            warn('Some values in overlay_times fall outside of the epochs '
                 f'time interval (between {times[0]} s and {times[-1]} s)')

    if callable(order):
        order = order(times, data)

    if order is not None:
        if len(order) != n_epochs:
            raise ValueError(f'If order is a {type(order).__name__}, its '
                             f'length ({len(order)}) must match the length of '
                             f'the data ({n_epochs}).')
        order = np.array(order)
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
    from matplotlib.ticker import AutoLocator

    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    tmin = epochs.times[0]
    tmax = epochs.times[-1]

    ax_im = ax['image']
    fig = ax_im.get_figure()

    # draw the image
    cmap = _setup_cmap(cmap, norm=norm)
    n_epochs = len(image)
    extent = [tmin, tmax, 0, n_epochs]
    im = ax_im.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap[0], aspect='auto',
                      origin='lower', interpolation='nearest', extent=extent)

    # optional things
    if style_axes:
        ax_im.set_title(title)
        ax_im.set_ylabel('Epochs')
        if not evoked:
            ax_im.set_xlabel('Time (s)')
        ax_im.axis('auto')
        ax_im.axis('tight')
        ax_im.axvline(0, color='k', linewidth=1, linestyle='--')

    if overlay_times is not None:
        ax_im.plot(overlay_times, 0.5 + np.arange(n_epochs), 'k',
                   linewidth=2)
        ax_im.set_xlim(tmin, tmax)
    # draw the evoked
    if evoked:
        from . import plot_compare_evokeds
        pass_combine = (combine if combine_given else None)
        _picks = [0] if len(picks) == 1 else None  # prevent applying GFP
        plot_compare_evokeds({'cond': list(epochs.iter_evoked(copy=False))},
                             picks=_picks, axes=ax['evoked'],
                             combine=pass_combine, **ts_args)
        ax['evoked'].set_xlim(tmin, tmax)
        ax['evoked'].lines[0].set_clip_on(True)
        ax['evoked'].collections[0].set_clip_on(True)
        _sharex(ax['evoked'], ax_im)
        # fix the axes for proper updating during interactivity
        loc = ax_im.xaxis.get_major_locator()
        ax['evoked'].xaxis.set_major_locator(loc)
        ax['evoked'].yaxis.set_major_locator(AutoLocator())

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


def plot_drop_log(drop_log, threshold=0, n_max_plot=20, subject=None,
                  color='lightgray', width=0.8, ignore=('IGNORED',),
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
    subject : str | None
        The subject name to use in the title of the plot. If ``None``, do not
        display a subject name.

        .. versionchanged:: 0.23
           Added support for ``None``.

        .. versionchanged:: 1.0
           Defaults to ``None``.
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
    absolute = len([x for x in drop_log if len(x)
                    if not any(y in ignore for y in x)])
    n_epochs_before_drop = len([x for x in drop_log
                                if not any(y in ignore for y in x)])

    scores = Counter([ch for d in drop_log for ch in d if ch not in ignore])
    ch_names = np.array(list(scores.keys()))
    counts = np.array(list(scores.values()))
    # init figure, handle easy case (no drops)
    fig, ax = plt.subplots()
    title = (f'{absolute} of {n_epochs_before_drop} epochs removed '
             f'({percent:.1f}%)')
    if subject is not None:
        title = f'{subject}: {title}'
    ax.set_title(title)
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
    ax.set_ylabel('% of epochs removed')
    ax.grid(axis='y')
    tight_layout(pad=1, fig=fig)
    plt_show(show)
    return fig


@fill_doc
def plot_epochs(epochs, picks=None, scalings=None, n_epochs=20, n_channels=20,
                title=None, events=None, event_color=None,
                order=None, show=True, block=False, decim='auto',
                noise_cov=None, butterfly=False, show_scrollbars=True,
                show_scalebars=True, epoch_colors=None, event_id=None,
                group_by='type', precompute=None, use_opengl=None, *,
                theme=None, overview_mode=None):
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
    %(scalings)s
    n_epochs : int
        The number of epochs per view. Defaults to 20.
    n_channels : int
        The number of channels per view. Defaults to 20.
    title : str | None
        The title of the window. If None, epochs name will be displayed.
        Defaults to None.
    events : None | array, shape (n_events, 3)
        Events to show with vertical bars. You can use `~mne.viz.plot_events`
        as a legend for the colors. By default, the coloring scheme is the
        same. Defaults to ``None``.

        .. warning::  If the epochs have been resampled, the events no longer
            align with the data.

        .. versionadded:: 0.14.0
    %(event_color)s
        Defaults to ``None``.
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
    %(show_scalebars)s

        .. versionadded:: 0.24.0
    epoch_colors : list of (n_epochs) list (of n_channels) | None
        Colors to use for individual epochs. If None, use default colors.
    event_id : dict | None
        Dictionary of event labels (e.g. 'aud_l') as keys and associated event
        integers as values. Useful when ``events`` contains event numbers not
        present in ``epochs.event_id`` (e.g., because of event subselection).
        Values in ``event_id`` will take precedence over those in
        ``epochs.event_id`` when there are overlapping keys.

        .. versionadded:: 0.20
    %(group_by_browse)s
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
    The arrow keys (up/down/left/right) can be used to navigate between
    channels and epochs and the scaling can be adjusted with - and + (or =)
    keys, but this depends on the backend matplotlib is configured to use
    (e.g., mpl.use(``TkAgg``) should work). Full screen mode can be toggled
    with f11 key. The amount of epochs and channels per view can be adjusted
    with home/end and page down/page up keys. ``h`` key plots a histogram of
    peak-to-peak values along with the used rejection thresholds. Butterfly
    plot can be toggled with ``b`` key. Left mouse click adds a vertical line
    to the plot. Click 'help' button at bottom left corner of the plotter to
    view all the options.

    %(notes_2d_backend)s

    .. versionadded:: 0.10.0
    """
    from ._figure import _get_browser

    epochs.drop_bad()
    info = epochs.info.copy()
    sfreq = info['sfreq']
    projs = info['projs']
    projs_on = np.full_like(projs, epochs.proj, dtype=bool)
    if not epochs.proj:
        with info._unlock():
            info['projs'] = list()

    # handle defaults / check arg validity
    color = _handle_default('color', None)
    scalings = _compute_scalings(scalings, epochs)
    scalings = _handle_default('scalings_plot_raw', scalings)
    if scalings['whitened'] == 'auto':
        scalings['whitened'] = 1.
    units = _handle_default('units', None)
    unit_scalings = _handle_default('scalings', None)
    decim, picks_data = _handle_decim(epochs.info.copy(), decim, None)
    noise_cov = _check_cov(noise_cov, epochs.info)
    event_id_rev = {v: k for k, v in (event_id or {}).items()}
    _check_option('group_by', group_by,
                  ('selection', 'position', 'original', 'type'))
    # validate epoch_colors
    _validate_type(epoch_colors, (list, None), 'epoch_colors')
    if epoch_colors is not None:
        if len(epoch_colors) != len(epochs.events):
            msg = ('epoch_colors must have length equal to the number of '
                   f'epochs ({len(epochs)}); got length {len(epoch_colors)}.')
            raise ValueError(msg)
        for ix, this_colors in enumerate(epoch_colors):
            _validate_type(this_colors, list, f'epoch_colors[{ix}]')
            if len(this_colors) != len(epochs.ch_names):
                msg = (f'epoch colors for epoch {ix} has length '
                       f'{len(this_colors)}, expected {len(epochs.ch_names)}.')
                raise ValueError(msg)

    # handle time dimension
    n_epochs = min(n_epochs, len(epochs))
    n_times = len(epochs) * len(epochs.times)
    duration = n_epochs * len(epochs.times) / sfreq
    # NB: this includes start and end of data:
    boundary_times = np.arange(len(epochs) + 1) * len(epochs.times) / sfreq

    # events
    if events is not None:
        event_nums = events[:, 2]
        event_samps = events[:, 0]
        epoch_n_samps = len(epochs.times)
        # handle overlapping epochs (each event may show up in multiple places)
        boundaries = (epochs.events[:, [0]] + np.array([-1, 1])
                      * epochs.time_as_index(0))
        in_bounds = np.logical_and(boundaries[:, [0]] <= event_samps,
                                   event_samps < boundaries[:, [1]])
        event_ixs = [np.nonzero(a)[0] for a in in_bounds.T]
        warned = False
        event_times = list()
        event_numbers = list()
        for samp, num, _ixs in zip(event_samps, event_nums, event_ixs):
            relevant_epoch_events = epochs.events[:, 0][_ixs]
            if len(relevant_epoch_events) > 1 and not warned:
                logger.info('You seem to have overlapping epochs. Some event '
                            'lines may be duplicated in the plot.')
                warned = True
            offsets = samp - relevant_epoch_events + epochs.time_as_index(0)
            this_event_times = (_ixs * epoch_n_samps + offsets) / sfreq
            event_times.extend(this_event_times)
            event_numbers.extend([num] * len(_ixs))
        event_nums = np.array(event_numbers)
        event_times = np.array(event_times)
    else:
        event_nums = None
        event_times = None
    event_color_dict = _make_event_color_dict(event_color, events, event_id)

    # determine trace order
    picks = _picks_to_idx(info, picks)
    n_channels = min(n_channels, len(picks))
    ch_names = np.array(epochs.ch_names)
    ch_types = np.array(epochs.get_channel_types())
    order = _get_channel_plotting_order(order, ch_types, picks)
    selections = None
    if group_by in ('selection', 'position'):
        selections = _setup_channel_selections(epochs, group_by, order)
        order = np.concatenate(list(selections.values()))
        default_selection = list(selections)[0]
        n_channels = len(selections[default_selection])

    # generate window title
    if title is None:
        title = epochs._name
        if title is None or len(title) == 0:
            title = 'Epochs'
    elif not isinstance(title, str):
        raise TypeError(f'title must be None or a string, got a {type(title)}')

    precompute = _handle_precompute(precompute)
    params = dict(inst=epochs,
                  info=info,
                  n_epochs=n_epochs,
                  # channels and channel order
                  ch_names=ch_names,
                  ch_types=ch_types,
                  ch_order=order,
                  picks=order[:n_channels],
                  n_channels=n_channels,
                  picks_data=picks_data,
                  group_by=group_by,
                  ch_selections=selections,
                  # time
                  t_start=0,
                  duration=duration,
                  n_times=n_times,
                  first_time=0,
                  time_format='float',
                  decim=decim,
                  boundary_times=boundary_times,
                  # events
                  event_id_rev=event_id_rev,
                  event_color_dict=event_color_dict,
                  event_nums=event_nums,
                  event_times=event_times,
                  # preprocessing
                  projs=projs,
                  projs_on=projs_on,
                  apply_proj=epochs.proj,
                  remove_dc=True,
                  filter_coefs=None,
                  filter_bounds=None,
                  noise_cov=noise_cov,
                  use_noise_cov=noise_cov is not None,
                  # scalings
                  scalings=scalings,
                  units=units,
                  unit_scalings=unit_scalings,
                  # colors
                  ch_color_bad='lightgray',
                  ch_color_dict=color,
                  epoch_color_bad=(1, 0, 0),
                  epoch_colors=epoch_colors,
                  # display
                  butterfly=butterfly,
                  clipping=None,
                  scrollbars_visible=show_scrollbars,
                  scalebars_visible=show_scalebars,
                  window_title=title,
                  xlabel='Epoch number',
                  # pyqtgraph-specific
                  precompute=precompute,
                  use_opengl=use_opengl,
                  theme=theme,
                  overview_mode=overview_mode)

    fig = _get_browser(show=show, block=block, **params)

    return fig


@legacy(alt='Epochs.compute_psd().plot()')
@verbose
def plot_epochs_psd(epochs, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                    proj=False, bandwidth=None, adaptive=False, low_bias=True,
                    normalization='length', picks=None, ax=None, color='black',
                    xscale='linear', area_mode='std', area_alpha=0.33,
                    dB=True, estimate='auto', show=True, n_jobs=None,
                    average=False, line_alpha=None, spatial_colors=True,
                    sphere=None, exclude='bads', verbose=None):
    """%(plot_psd_doc)s.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object.
    %(fmin_fmax_psd)s
    %(tmin_tmax_psd)s
    %(proj_psd)s
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    %(normalization)s
    %(picks_good_data_noref)s
    %(ax_plot_psd)s
    %(color_plot_psd)s
    %(xscale_plot_psd)s
    %(area_mode_plot_psd)s
    %(area_alpha_plot_psd)s
    %(dB_plot_psd)s
    %(estimate_plot_psd)s
    %(show)s
    %(n_jobs)s
    %(average_plot_psd)s
    %(line_alpha_plot_psd)s
    %(spatial_colors_psd)s
    %(sphere_topomap_auto)s
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the bad channels
        are excluded. Pass an empty list to plot all channels (including
        channels marked "bad", if any).

        .. versionadded:: 0.24.0
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure with frequency spectra of the data channels.

    Notes
    -----
    %(notes_plot_*_psd_func)s
    """
    fig = epochs.plot_psd(
        fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=picks,
        proj=proj, method='multitaper',
        ax=ax, color=color, xscale=xscale, area_mode=area_mode,
        area_alpha=area_alpha, dB=dB, estimate=estimate, show=show,
        line_alpha=line_alpha, spatial_colors=spatial_colors, sphere=sphere,
        exclude=exclude, n_jobs=n_jobs, average=average, verbose=verbose,
        # these are **method_kw:
        window='hamming', bandwidth=bandwidth, adaptive=adaptive,
        low_bias=low_bias, normalization=normalization)
    return fig
