"""Utility functions for plotting M/EEG data
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import math
from functools import partial
import difflib
import webbrowser
import tempfile
import numpy as np
from copy import deepcopy
from distutils.version import LooseVersion

from ..channels.layout import _auto_topomap_coords
from ..channels.channels import _contains_ch_type
from ..defaults import _handle_default
from ..io import show_fiff, Info
from ..io.pick import channel_type, channel_indices_by_type, pick_channels
from ..utils import verbose, set_config, warn
from ..externals.six import string_types
from ..selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                         _divide_to_regions)


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#473C8B', '#458B74',
          '#CD7F32', '#FF4040', '#ADFF2F', '#8E2323', '#FF1493']


def _setup_vmin_vmax(data, vmin, vmax, norm=False):
    """Aux function to handle vmin and vmax parameters"""
    if vmax is None and vmin is None:
        vmax = np.abs(data).max()
        if norm:
            vmin = 0.
        else:
            vmin = -vmax
    else:
        if callable(vmin):
            vmin = vmin(data)
        elif vmin is None:
            if norm:
                vmin = 0.
            else:
                vmin = np.min(data)
        if callable(vmax):
            vmax = vmax(data)
        elif vmax is None:
            vmax = np.max(data)
    return vmin, vmax


def plt_show(show=True, **kwargs):
    """Helper to show a figure while suppressing warnings"""
    import matplotlib
    import matplotlib.pyplot as plt
    if show and matplotlib.get_backend() != 'agg':
        plt.show(**kwargs)


def tight_layout(pad=1.2, h_pad=None, w_pad=None, fig=None):
    """ Adjust subplot parameters to give specified padding.

    Note. For plotting please use this function instead of plt.tight_layout

    Parameters
    ----------
    pad : float
        padding between the figure edge and the edges of subplots, as a
        fraction of the font-size.
    h_pad : float
        Padding height between edges of adjacent subplots.
        Defaults to `pad_inches`.
    w_pad : float
        Padding width between edges of adjacent subplots.
        Defaults to `pad_inches`.
    fig : instance of Figure
        Figure to apply changes to.
    """
    import matplotlib.pyplot as plt
    fig = plt.gcf() if fig is None else fig

    fig.canvas.draw()
    try:  # see https://github.com/matplotlib/matplotlib/issues/2654
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    except Exception:
        try:
            fig.set_tight_layout(dict(pad=pad, h_pad=h_pad, w_pad=w_pad))
        except Exception:
            warn('Matplotlib function "tight_layout" is not supported.'
                 ' Skipping subplot adjustment.')


def _check_delayed_ssp(container):
    """ Aux function to be used for interactive SSP selection
    """
    if container.proj is True or\
       all(p['active'] for p in container.info['projs']):
        raise RuntimeError('Projs are already applied. Please initialize'
                           ' the data with proj set to False.')
    elif len(container.info['projs']) < 1:
        raise RuntimeError('No projs found in evoked.')


def _validate_if_list_of_axes(axes, obligatory_len=None):
    """ Helper function that validates whether input is a list/array of axes"""
    import matplotlib as mpl
    if obligatory_len is not None and not isinstance(obligatory_len, int):
        raise ValueError('obligatory_len must be None or int, got %d',
                         'instead' % type(obligatory_len))
    if not isinstance(axes, (list, np.ndarray)):
        raise ValueError('axes must be a list or numpy array of matplotlib '
                         'axes objects, got %s instead.' % type(axes))
    if isinstance(axes, np.ndarray) and axes.ndim > 1:
        raise ValueError('if input is a numpy array, it must be '
                         'one-dimensional. The received numpy array has %d '
                         'dimensions however. Try using ravel or flatten '
                         'method of the array.' % axes.ndim)
    is_correct_type = np.array([isinstance(x, mpl.axes.Axes)
                               for x in axes])
    if not np.all(is_correct_type):
        first_bad = np.where(np.logical_not(is_correct_type))[0][0]
        raise ValueError('axes must be a list or numpy array of matplotlib '
                         'axes objects while one of the list elements is '
                         '%s.' % type(axes[first_bad]))
    if obligatory_len is not None and not len(axes) == obligatory_len:
        raise ValueError('axes must be a list/array of length %d, while the'
                         ' length is %d' % (obligatory_len, len(axes)))


def mne_analyze_colormap(limits=[5, 10, 15], format='mayavi'):
    """Return a colormap similar to that used by mne_analyze

    Parameters
    ----------
    limits : list (or array) of length 3 or 6
        Bounds for the colormap, which will be mirrored across zero if length
        3, or completely specified (and potentially asymmetric) if length 6.
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
    # Ensure limits is an array
    limits = np.asarray(limits, dtype='float')

    if len(limits) != 3 and len(limits) != 6:
        raise ValueError('limits must have 3 or 6 elements')
    if len(limits) == 3 and any(limits < 0.):
        raise ValueError('if 3 elements, limits must all be non-negative')
    if any(np.diff(limits) <= 0):
        raise ValueError('limits must be monotonically increasing')
    if format == 'matplotlib':
        from matplotlib import colors
        if len(limits) == 3:
            limits = (np.concatenate((-np.flipud(limits), limits)) +
                      limits[-1]) / (2 * limits[-1])
        else:
            limits = (limits - np.min(limits)) / np.max(limits -
                                                        np.min(limits))

        cdict = {'red': ((limits[0], 0.0, 0.0),
                         (limits[1], 0.0, 0.0),
                         (limits[2], 0.5, 0.5),
                         (limits[3], 0.5, 0.5),
                         (limits[4], 1.0, 1.0),
                         (limits[5], 1.0, 1.0)),
                 'green': ((limits[0], 1.0, 1.0),
                           (limits[1], 0.0, 0.0),
                           (limits[2], 0.5, 0.5),
                           (limits[3], 0.5, 0.5),
                           (limits[4], 0.0, 0.0),
                           (limits[5], 1.0, 1.0)),
                 'blue': ((limits[0], 1.0, 1.0),
                          (limits[1], 1.0, 1.0),
                          (limits[2], 0.5, 0.5),
                          (limits[3], 0.5, 0.5),
                          (limits[4], 0.0, 0.0),
                          (limits[5], 0.0, 0.0))}
        return colors.LinearSegmentedColormap('mne_analyze', cdict)
    elif format == 'mayavi':
        if len(limits) == 3:
            limits = np.concatenate((-np.flipud(limits), [0], limits)) /\
                limits[-1]
        else:
            limits = np.concatenate((limits[:3], [0], limits[3:]))
            limits /= np.max(np.abs(limits))
        r = np.array([0, 0, 0, 0, 1, 1, 1])
        g = np.array([1, 0, 0, 0, 0, 0, 1])
        b = np.array([1, 1, 1, 0, 0, 0, 0])
        a = np.array([1, 1, 0, 0, 0, 1, 1])
        xp = (np.arange(256) - 128) / 128.0
        colormap = np.r_[[np.interp(xp, limits, 255 * c)
                          for c in [r, g, b, a]]].T
        return colormap
    else:
        raise ValueError('format must be either matplotlib or mayavi')


def _toggle_options(event, params):
    """Toggle options (projectors) dialog"""
    import matplotlib.pyplot as plt
    if len(params['projs']) > 0:
        if params['fig_proj'] is None:
            _draw_proj_checkbox(event, params, draw_current_state=False)
        else:
            # turn off options dialog
            plt.close(params['fig_proj'])
            del params['proj_checks']
            params['fig_proj'] = None


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
    if 'proj_bools' not in params:
        compute_proj = True
    elif not np.array_equal(bools, params['proj_bools']):
        compute_proj = True

    # if projectors changed, update plots
    if compute_proj is True:
        params['plot_update_proj_callback'](params, bools)


def _get_help_text(params):
    """Aux function for customizing help dialogs text."""
    text, text2 = list(), list()

    text.append(u'\u2190 : \n')
    text.append(u'\u2192 : \n')
    text.append(u'\u2193 : \n')
    text.append(u'\u2191 : \n')
    text.append(u'- : \n')
    text.append(u'+ or = : \n')
    text.append(u'Home : \n')
    text.append(u'End : \n')
    if 'fig_selection' not in params:
        text.append(u'Page down : \n')
        text.append(u'Page up : \n')

    text.append(u'F11 : \n')
    text.append(u'? : \n')
    text.append(u'Esc : \n\n')
    text.append(u'Mouse controls\n')
    text.append(u'click on data :\n')

    text2.append('Navigate left\n')
    text2.append('Navigate right\n')

    text2.append('Scale down\n')
    text2.append('Scale up\n')

    text2.append('Toggle full screen mode\n')
    text2.append('Open help box\n')
    text2.append('Quit\n\n\n')
    if 'raw' in params:
        text2.insert(4, 'Reduce the time shown per view\n')
        text2.insert(5, 'Increase the time shown per view\n')
        text.append(u'click elsewhere in the plot :\n')
        if 'ica' in params:
            text.append(u'click component name :\n')
            text2.insert(2, 'Navigate components down\n')
            text2.insert(3, 'Navigate components up\n')
            text2.insert(8, 'Reduce the number of components per view\n')
            text2.insert(9, 'Increase the number of components per view\n')
            text2.append('Mark bad channel\n')
            text2.append('Vertical line at a time instant\n')
            text2.append('Show topography for the component\n')
        else:
            text.append(u'click channel name :\n')
            text2.insert(2, 'Navigate channels down\n')
            text2.insert(3, 'Navigate channels up\n')
            if 'fig_selection' not in params:
                text2.insert(8, 'Reduce the number of channels per view\n')
                text2.insert(9, 'Increase the number of channels per view\n')
            text2.append('Mark bad channel\n')
            text2.append('Vertical line at a time instant\n')
            text2.append('Mark bad channel\n')

    elif 'epochs' in params:
        text.append(u'right click :\n')
        text2.insert(4, 'Reduce the number of epochs per view\n')
        text2.insert(5, 'Increase the number of epochs per view\n')
        if 'ica' in params:
            text.append(u'click component name :\n')
            text2.insert(2, 'Navigate components down\n')
            text2.insert(3, 'Navigate components up\n')
            text2.insert(8, 'Reduce the number of components per view\n')
            text2.insert(9, 'Increase the number of components per view\n')
            text2.append('Mark component for exclusion\n')
            text2.append('Vertical line at a time instant\n')
            text2.append('Show topography for the component\n')
        else:
            text.append(u'click channel name :\n')
            text.append(u'right click channel name :\n')
            text2.insert(2, 'Navigate channels down\n')
            text2.insert(3, 'Navigate channels up\n')
            text2.insert(8, 'Reduce the number of channels per view\n')
            text2.insert(9, 'Increase the number of channels per view\n')
            text.insert(10, u'b : \n')
            text2.insert(10, 'Toggle butterfly plot on/off\n')
            text.insert(11, u'h : \n')
            text2.insert(11, 'Show histogram of peak-to-peak values\n')
            text2.append('Mark bad epoch\n')
            text2.append('Vertical line at a time instant\n')
            text2.append('Mark bad channel\n')
            text2.append('Plot ERP/ERF image\n')
            text.append(u'middle click :\n')
            text2.append('Show channel name (butterfly plot)\n')
        text.insert(11, u'o : \n')
        text2.insert(11, 'View settings (orig. view only)\n')

    return ''.join(text), ''.join(text2)


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

    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 1.5 * nrow + 1))
    axes = [axes] if ncol == nrow == 1 else axes.flatten()
    for ax in axes[n_cells:]:  # hide unused axes
        # XXX: Previously done by ax.set_visible(False), but because of mpl
        # bug, we just hide the frame.
        from .topomap import _hide_frame
        _hide_frame(ax)
    return fig, axes


def _draw_proj_checkbox(event, params, draw_current_state=True):
    """Toggle options (projectors) dialog"""
    from matplotlib import widgets
    projs = params['projs']
    # turn on options dialog

    labels = [p['desc'] for p in projs]
    actives = ([p['active'] for p in projs] if draw_current_state else
               [True] * len(params['projs']))

    width = max([len(p['desc']) for p in projs]) / 6.0 + 0.5
    height = len(projs) / 6.0 + 0.5
    fig_proj = figure_nobar(figsize=(width, height))
    fig_proj.canvas.set_window_title('SSP projection vectors')
    params['fig_proj'] = fig_proj  # necessary for proper toggling
    ax_temp = fig_proj.add_axes((0, 0, 1, 1), frameon=False)

    proj_checks = widgets.CheckButtons(ax_temp, labels=labels, actives=actives)
    # change already-applied projectors to red
    for ii, p in enumerate(projs):
        if p['active'] is True:
            for x in proj_checks.lines[ii]:
                x.set_color('r')
    # make minimal size
    # pass key presses from option dialog over

    proj_checks.on_clicked(partial(_toggle_proj, params=params))
    params['proj_checks'] = proj_checks

    # this should work for non-test cases
    try:
        fig_proj.canvas.draw()
        fig_proj.show(warn=False)
    except Exception:
        pass


def _layout_figure(params):
    """Function for setting figure layout. Shared with raw and epoch plots"""
    size = params['fig'].get_size_inches() * params['fig'].dpi
    scroll_width = 25
    hscroll_dist = 25
    vscroll_dist = 10
    l_border = 100
    r_border = 10
    t_border = 35
    b_border = 40

    # only bother trying to reset layout if it's reasonable to do so
    if size[0] < 2 * scroll_width or size[1] < 2 * scroll_width + hscroll_dist:
        return

    # convert to relative units
    scroll_width_x = scroll_width / size[0]
    scroll_width_y = scroll_width / size[1]
    vscroll_dist /= size[0]
    hscroll_dist /= size[1]
    l_border /= size[0]
    r_border /= size[0]
    t_border /= size[1]
    b_border /= size[1]
    # main axis (traces)
    ax_width = 1.0 - scroll_width_x - l_border - r_border - vscroll_dist
    ax_y = hscroll_dist + scroll_width_y + b_border
    ax_height = 1.0 - ax_y - t_border

    pos = [l_border, ax_y, ax_width, ax_height]

    params['ax'].set_position(pos)
    if 'ax2' in params:
        params['ax2'].set_position(pos)
    params['ax'].set_position(pos)
    # vscroll (channels)
    pos = [ax_width + l_border + vscroll_dist, ax_y,
           scroll_width_x, ax_height]
    params['ax_vscroll'].set_position(pos)
    # hscroll (time)
    pos = [l_border, b_border, ax_width, scroll_width_y]
    params['ax_hscroll'].set_position(pos)
    if 'ax_button' in params:
        # options button
        pos = [l_border + ax_width + vscroll_dist, b_border,
               scroll_width_x, scroll_width_y]
        params['ax_button'].set_position(pos)
    if 'ax_help_button' in params:
        pos = [l_border - vscroll_dist - scroll_width_x * 2, b_border,
               scroll_width_x * 2, scroll_width_y]
        params['ax_help_button'].set_position(pos)
    params['fig'].canvas.draw()


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
        f = open(fname_out, 'wb')
    else:
        f = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.html')
        fname_out = f.name
    with f as fid:
        fid.write(diff.encode('utf-8'))
    if show is True:
        webbrowser.open_new_tab(fname_out)
    return fname_out


def figure_nobar(*args, **kwargs):
    """Make matplotlib figure with no toolbar"""
    from matplotlib import rcParams, pyplot as plt
    old_val = rcParams['toolbar']
    try:
        rcParams['toolbar'] = 'none'
        fig = plt.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        cbs = list(fig.canvas.callbacks.callbacks['key_press_event'].keys())
        for key in cbs:
            fig.canvas.callbacks.disconnect(key)
    except Exception as ex:
        raise ex
    finally:
        rcParams['toolbar'] = old_val
    return fig


def _helper_raw_resize(event, params):
    """Helper for resizing"""
    size = ','.join([str(s) for s in params['fig'].get_size_inches()])
    set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
    _layout_figure(params)


def _plot_raw_onscroll(event, params, len_channels=None):
    """Interpret scroll events"""
    if 'fig_selection' in params:
        _change_channel_group(event.step, params)
        return
    if len_channels is None:
        len_channels = len(params['inds'])
    orig_start = params['ch_start']
    if event.step < 0:
        params['ch_start'] = min(params['ch_start'] + params['n_channels'],
                                 len_channels - params['n_channels'])
    else:  # event.key == 'up':
        params['ch_start'] = max(params['ch_start'] - params['n_channels'], 0)
    if orig_start != params['ch_start']:
        _channels_changed(params, len_channels)


def _channels_changed(params, len_channels):
    """Helper function for dealing with the vertical shift of the viewport."""
    if params['ch_start'] + params['n_channels'] > len_channels:
        params['ch_start'] = len_channels - params['n_channels']
    if params['ch_start'] < 0:
        params['ch_start'] = 0
    params['plot_fun']()


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


def _radio_clicked(label, params):
    """Callback for radio buttons in selection dialog."""
    from .evoked import _rgb
    labels = [l._text for l in params['fig_selection'].radio.labels]
    idx = labels.index(label)
    params['fig_selection'].radio._active_idx = idx
    channels = params['selections'][label]
    ax_topo = params['fig_selection'].get_axes()[1]
    types = np.array([], dtype=int)
    for this_type in ('mag', 'grad', 'eeg', 'seeg', 'ecog', 'hbo', 'hbr'):
        if this_type in params['types']:
            types = np.concatenate(
                [types, np.where(np.array(params['types']) == this_type)[0]])
    colors = np.zeros((len(types), 4))  # alpha = 0 by default
    locs3d = np.array([ch['loc'][:3] for ch in params['info']['chs']])
    x, y, z = locs3d.T
    color_vals = _rgb(params['info'], x, y, z)
    for color_idx, pick in enumerate(types):
        if pick in channels:  # set color and alpha = 1
            colors[color_idx] = np.append(color_vals[pick], 1.)
    ax_topo.collections[0]._facecolors = colors
    params['fig_selection'].canvas.draw()

    nchan = sum([len(params['selections'][l]) for l in labels[:idx]])
    params['vsel_patch'].set_y(nchan)
    n_channels = len(channels)
    params['n_channels'] = n_channels
    params['inds'] = channels
    for line in params['lines'][n_channels:]:  # To remove lines from view.
        line.set_xdata([])
        line.set_ydata([])
    if n_channels > 0:  # Can be 0 with lasso selector.
        _setup_browser_offsets(params, n_channels)
    params['plot_fun']()


def _set_radio_button(idx, params):
    """Helper for setting radio button."""
    # XXX: New version of matplotlib has this implemented for radio buttons,
    # This function is for compatibility with old versions of mpl.
    radio = params['fig_selection'].radio
    radio.circles[radio._active_idx].set_facecolor((1., 1., 1., 1.))
    radio.circles[idx].set_facecolor((0., 0., 1., 1.))
    _radio_clicked(radio.labels[idx]._text, params)


def _change_channel_group(step, params):
    """Deal with change of channel group."""
    radio = params['fig_selection'].radio
    idx = radio._active_idx
    if step < 0:
        if idx < len(radio.labels) - 1:
            _set_radio_button(idx + 1, params)
    else:
        if idx > 0:
            _set_radio_button(idx - 1, params)
    return


def _handle_change_selection(event, params):
    """Helper for handling clicks on vertical scrollbar using selections."""
    radio = params['fig_selection'].radio
    ydata = event.ydata
    labels = [label._text for label in radio.labels]
    offset = 0
    for idx, label in enumerate(labels):
        nchans = len(params['selections'][label])
        offset += nchans
        if ydata < offset:
            _set_radio_button(idx, params)
            return


def _plot_raw_onkey(event, params):
    """Interpret key presses"""
    import matplotlib.pyplot as plt
    if event.key == 'escape':
        plt.close(params['fig'])
    elif event.key == 'down':
        if 'fig_selection' in params.keys():
            _change_channel_group(-1, params)
            return
        params['ch_start'] += params['n_channels']
        _channels_changed(params, len(params['inds']))
    elif event.key == 'up':
        if 'fig_selection' in params.keys():
            _change_channel_group(1, params)
            return
        params['ch_start'] -= params['n_channels']
        _channels_changed(params, len(params['inds']))
    elif event.key == 'right':
        value = params['t_start'] + params['duration']
        _plot_raw_time(value, params)
        params['update_fun']()
        params['plot_fun']()
    elif event.key == 'left':
        value = params['t_start'] - params['duration']
        _plot_raw_time(value, params)
        params['update_fun']()
        params['plot_fun']()
    elif event.key in ['+', '=']:
        params['scale_factor'] *= 1.1
        params['plot_fun']()
    elif event.key == '-':
        params['scale_factor'] /= 1.1
        params['plot_fun']()
    elif event.key == 'pageup' and 'fig_selection' not in params:
        n_channels = params['n_channels'] + 1
        _setup_browser_offsets(params, n_channels)
        _channels_changed(params, len(params['inds']))
    elif event.key == 'pagedown' and 'fig_selection' not in params:
        n_channels = params['n_channels'] - 1
        if n_channels == 0:
            return
        _setup_browser_offsets(params, n_channels)
        if len(params['lines']) > n_channels:  # remove line from view
            params['lines'][n_channels].set_xdata([])
            params['lines'][n_channels].set_ydata([])
        _channels_changed(params, len(params['inds']))
    elif event.key == 'home':
        duration = params['duration'] - 1.0
        if duration <= 0:
            return
        params['duration'] = duration
        params['hsel_patch'].set_width(params['duration'])
        params['update_fun']()
        params['plot_fun']()
    elif event.key == 'end':
        duration = params['duration'] + 1.0
        if duration > params['raw'].times[-1]:
            duration = params['raw'].times[-1]
        params['duration'] = duration
        params['hsel_patch'].set_width(params['duration'])
        params['update_fun']()
        params['plot_fun']()
    elif event.key == '?':
        _onclick_help(event, params)
    elif event.key == 'f11':
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()


def _mouse_click(event, params):
    """Vertical select callback"""
    if event.button != 1:
        return
    if event.inaxes is None:
        if params['n_channels'] > 100:
            return
        ax = params['ax']
        ylim = ax.get_ylim()
        pos = ax.transData.inverted().transform((event.x, event.y))
        if pos[0] > params['t_start'] or pos[1] < 0 or pos[1] > ylim[0]:
            return
        params['label_click_fun'](pos)
    # vertical scrollbar changed
    if event.inaxes == params['ax_vscroll']:
        if 'fig_selection' in params.keys():
            _handle_change_selection(event, params)
        else:
            ch_start = max(int(event.ydata) - params['n_channels'] // 2, 0)
            if params['ch_start'] != ch_start:
                params['ch_start'] = ch_start
                params['plot_fun']()
    # horizontal scrollbar changed
    elif event.inaxes == params['ax_hscroll']:
        _plot_raw_time(event.xdata - params['duration'] / 2, params)
        params['update_fun']()
        params['plot_fun']()

    elif event.inaxes == params['ax']:
        params['pick_bads_fun'](event)


def _handle_topomap_bads(ch_name, params):
    """Helper for coloring channels in selection topomap when selecting bads"""
    for type in ('mag', 'grad', 'eeg', 'seeg', 'hbo', 'hbr'):
        if type in params['types']:
            types = np.where(np.array(params['types']) == type)[0]
            break
    color_ind = np.where(np.array(
        params['info']['ch_names'])[types] == ch_name)[0]
    if len(color_ind) > 0:
        sensors = params['fig_selection'].axes[1].collections[0]
        this_color = sensors._edgecolors[color_ind][0]
        if all(this_color == [1., 0., 0., 1.]):  # is red
            sensors._edgecolors[color_ind] = [0., 0., 0., 1.]
        else:  # is black
            sensors._edgecolors[color_ind] = [1., 0., 0., 1.]
        params['fig_selection'].canvas.draw()


def _find_channel_idx(ch_name, params):
    """Helper for finding all indices when using selections."""
    indices = list()
    offset = 0
    labels = [l._text for l in params['fig_selection'].radio.labels]
    for label in labels:
        if label == 'Custom':
            continue  # Custom selection not included as it shifts the indices.
        selection = params['selections'][label]
        hits = np.where(np.array(params['raw'].ch_names)[selection] == ch_name)
        for idx in hits[0]:
            indices.append(offset + idx)
        offset += len(selection)
    return indices


def _select_bads(event, params, bads):
    """Helper for selecting bad channels onpick. Returns updated bads list."""
    # trade-off, avoid selecting more than one channel when drifts are present
    # however for clean data don't click on peaks but on flat segments
    def f(x, y):
        return y(np.mean(x), x.std() * 2)
    lines = event.inaxes.lines
    for line in lines:
        ydata = line.get_ydata()
        if not isinstance(ydata, list) and not np.isnan(ydata).any():
            ymin, ymax = f(ydata, np.subtract), f(ydata, np.add)
            if ymin <= event.ydata <= ymax:
                this_chan = vars(line)['ch_name']
                if this_chan in params['info']['ch_names']:
                    if 'fig_selection' in params:
                        ch_idx = _find_channel_idx(this_chan, params)
                        _handle_topomap_bads(this_chan, params)
                    else:
                        ch_idx = [params['ch_start'] + lines.index(line)]

                    if this_chan not in bads:
                        bads.append(this_chan)
                        color = params['bad_color']
                        line.set_zorder(-1)
                    else:
                        while this_chan in bads:
                            bads.remove(this_chan)
                        color = vars(line)['def_color']
                        line.set_zorder(0)
                    line.set_color(color)
                    for idx in ch_idx:
                        params['ax_vscroll'].patches[idx].set_color(color)
                    break
    else:
        x = np.array([event.xdata] * 2)
        params['ax_vertline'].set_data(x, np.array(params['ax'].get_ylim()))
        params['ax_hscroll_vertline'].set_data(x, np.array([0., 1.]))
        params['vertline_t'].set_text('%0.3f' % x[0])

    return bads


def _onclick_help(event, params):
    """Function for drawing help window"""
    import matplotlib.pyplot as plt
    text, text2 = _get_help_text(params)

    width = 6
    height = 5

    fig_help = figure_nobar(figsize=(width, height), dpi=80)
    fig_help.canvas.set_window_title('Help')
    ax = plt.subplot2grid((8, 5), (0, 0), colspan=5)
    ax.set_title('Keyboard shortcuts')
    plt.axis('off')
    ax1 = plt.subplot2grid((8, 5), (1, 0), rowspan=7, colspan=2)
    ax1.set_yticklabels(list())
    plt.text(0.99, 1, text, fontname='STIXGeneral', va='top', weight='bold',
             ha='right')
    plt.axis('off')

    ax2 = plt.subplot2grid((8, 5), (1, 2), rowspan=7, colspan=3)
    ax2.set_yticklabels(list())
    plt.text(0, 1, text2, fontname='STIXGeneral', va='top')
    plt.axis('off')

    tight_layout(fig=fig_help)
    # this should work for non-test cases
    try:
        fig_help.canvas.draw()
        fig_help.show(warn=False)
    except Exception:
        pass


def _setup_browser_offsets(params, n_channels):
    """Aux function for computing viewport height and adjusting offsets."""
    ylim = [n_channels * 2 + 1, 0]
    offset = ylim[0] / n_channels
    params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
    params['n_channels'] = n_channels
    params['ax'].set_yticks(params['offsets'])
    params['ax'].set_ylim(ylim)
    params['vsel_patch'].set_height(n_channels)
    line = params['ax_vertline']
    line.set_data(line._x, np.array(params['ax'].get_ylim()))


class ClickableImage(object):

    """
    Display an image so you can click on it and store x/y positions.

    Takes as input an image array (can be any array that works with imshow,
    but will work best with images.  Displays the image and lets you
    click on it.  Stores the xy coordinates of each click, so now you can
    superimpose something on top of it.

    Upon clicking, the x/y coordinate of the cursor will be stored in
    self.coords, which is a list of (x, y) tuples.

    Parameters
    ----------
    imdata : ndarray
        The image that you wish to click on for 2-d points.
    **kwargs : dict
        Keyword arguments. Passed to ax.imshow.

    Notes
    -----
    .. versionadded:: 0.9.0

    """

    def __init__(self, imdata, **kwargs):
        """Display the image for clicking."""
        from matplotlib.pyplot import figure
        self.coords = []
        self.imdata = imdata
        self.fig = figure()
        self.ax = self.fig.add_subplot(111)
        self.ymax = self.imdata.shape[0]
        self.xmax = self.imdata.shape[1]
        self.im = self.ax.imshow(imdata, aspect='auto',
                                 extent=(0, self.xmax, 0, self.ymax),
                                 picker=True, **kwargs)
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('pick_event', self.onclick)
        plt_show()

    def onclick(self, event):
        """Mouse click handler.

        Parameters
        ----------
        event : matplotlib event object
            The matplotlib object that we use to get x/y position.
        """
        mouseevent = event.mouseevent
        self.coords.append((mouseevent.xdata, mouseevent.ydata))

    def plot_clicks(self, **kwargs):
        """Plot the x/y positions stored in self.coords.

        Parameters
        ----------
        **kwargs : dict
            Arguments are passed to imshow in displaying the bg image.
        """
        from matplotlib.pyplot import subplots
        f, ax = subplots()
        ax.imshow(self.imdata, extent=(0, self.xmax, 0, self.ymax), **kwargs)
        xlim, ylim = [ax.get_xlim(), ax.get_ylim()]
        xcoords, ycoords = zip(*self.coords)
        ax.scatter(xcoords, ycoords, c='r')
        ann_text = np.arange(len(self.coords)).astype(str)
        for txt, coord in zip(ann_text, self.coords):
            ax.annotate(txt, coord, fontsize=20, color='r')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt_show()

    def to_layout(self, **kwargs):
        """Turn coordinates into an MNE Layout object.

        Normalizes by the image you used to generate clicks

        Parameters
        ----------
        **kwargs : dict
            Arguments are passed to generate_2d_layout
        """
        from ..channels.layout import generate_2d_layout
        coords = np.array(self.coords)
        lt = generate_2d_layout(coords, bg_image=self.imdata, **kwargs)
        return lt


def _fake_click(fig, ax, point, xform='ax', button=1, kind='press'):
    """Helper to fake a click at a relative point within axes."""
    if xform == 'ax':
        x, y = ax.transAxes.transform_point(point)
    elif xform == 'data':
        x, y = ax.transData.transform_point(point)
    else:
        raise ValueError('unknown transform')
    if kind == 'press':
        func = partial(fig.canvas.button_press_event, x=x, y=y, button=button)
    elif kind == 'release':
        func = partial(fig.canvas.button_release_event, x=x, y=y,
                       button=button)
    elif kind == 'motion':
        func = partial(fig.canvas.motion_notify_event, x=x, y=y)
    try:
        func(guiEvent=None)
    except Exception:  # for old MPL
        func()


def add_background_image(fig, im, set_ratios=None):
    """Add a background image to a plot.

    Adds the image specified in `im` to the
    figure `fig`. This is generally meant to
    be done with topo plots, though it could work
    for any plot.

    Note: This modifies the figure and/or axes
    in place.

    Parameters
    ----------
    fig : plt.figure
        The figure you wish to add a bg image to.
    im : array, shape (M, N, {3, 4})
        A background image for the figure. This must be a valid input to
        `matplotlib.pyplot.imshow`. Defaults to None.
    set_ratios : None | str
        Set the aspect ratio of any axes in fig
        to the value in set_ratios. Defaults to None,
        which does nothing to axes.

    Returns
    -------
    ax_im : instance of the created matplotlib axis object
        corresponding to the image you added.

    Notes
    -----
    .. versionadded:: 0.9.0

    """
    if im is None:
        # Don't do anything and return nothing
        return None
    if set_ratios is not None:
        for ax in fig.axes:
            ax.set_aspect(set_ratios)

    ax_im = fig.add_axes([0, 0, 1, 1], label='background')
    ax_im.imshow(im, aspect='auto')
    ax_im.set_zorder(-1)
    return ax_im


def _find_peaks(evoked, npeaks):
    """Helper function for finding peaks from evoked data
    Returns ``npeaks`` biggest peaks as a list of time points.
    """
    from scipy.signal import argrelmax
    gfp = evoked.data.std(axis=0)
    order = len(evoked.times) // 30
    if order < 1:
        order = 1
    peaks = argrelmax(gfp, order=order, axis=0)[0]
    if len(peaks) > npeaks:
        max_indices = np.argsort(gfp[peaks])[-npeaks:]
        peaks = np.sort(peaks[max_indices])
    times = evoked.times[peaks]
    if len(times) == 0:
        times = [evoked.times[gfp.argmax()]]
    return times


def _process_times(inst, times, n_peaks=None, few=False):
    """Helper to return a list of times for topomaps"""
    if isinstance(times, string_types):
        if times == "peaks":
            if n_peaks is None:
                n_peaks = 3 if few else 7
            times = _find_peaks(inst, n_peaks)
        elif times == "auto":
            if n_peaks is None:
                n_peaks = 5 if few else 10
            times = np.linspace(inst.times[0], inst.times[-1], n_peaks)
        else:
            raise ValueError("Got an unrecognized method for `times`. Only "
                             "'peaks' and 'auto' are supported (or directly "
                             "passing numbers).")
    elif np.isscalar(times):
        times = [times]

    times = np.array(times)

    if times.ndim != 1:
        raise ValueError('times must be 1D, got %d dimensions' % times.ndim)
    if len(times) > 20:
        raise RuntimeError('Too many plots requested. Please pass fewer '
                           'than 20 time instants.')

    return times


def plot_sensors(info, kind='topomap', ch_type=None, title=None,
                 show_names=False, ch_groups=None, axes=None, block=False,
                 show=True):
    """Plot sensors positions.

    Parameters
    ----------
    info : Instance of Info
        Info structure containing the channel locations.
    kind : str
        Whether to plot the sensors as 3d, topomap or as an interactive
        sensor selection dialog. Available options 'topomap', '3d', 'select'.
        If 'select', a set of channels can be selected interactively by using
        lasso selector or clicking while holding control key. The selected
        channels are returned along with the figure instance. Defaults to
        'topomap'.
    ch_type : None | str
        The channel type to plot. Available options 'mag', 'grad', 'eeg',
        'seeg', 'ecog', 'all'. If ``'all'``, all the available mag, grad, eeg,
        seeg and ecog channels are plotted. If None (default), then channels
        are chosen in the order given above.
    title : str | None
        Title for the figure. If None (default), equals to
        ``'Sensor positions (%s)' % ch_type``.
    show_names : bool
        Whether to display all channel names. Defaults to False.
    ch_groups : 'position' | array of shape (ch_groups, picks) | None
        Channel groups for coloring the sensors. If None (default), default
        coloring scheme is used. If 'position', the sensors are divided
        into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
        array, the channels are divided by picks given in the array.

        .. versionadded:: 0.13.0

    axes : instance of Axes | instance of Axes3D | None
        Axes to draw the sensors to. If ``kind='3d'``, axes must be an instance
        of Axes3D. If None (default), a new axes will be created.

        .. versionadded:: 0.13.0

    block : bool
        Whether to halt program execution until the figure is closed. Defaults
        to False.

        .. versionadded:: 0.13.0

    show : bool
        Show figure if True. Defaults to True.

    Returns
    -------
    fig : instance of matplotlib figure
        Figure containing the sensor topography.
    selection : list
        A list of selected channels. Only returned if ``kind=='select'``.

    See Also
    --------
    mne.viz.plot_layout

    Notes
    -----
    This function plots the sensor locations from the info structure using
    matplotlib. For drawing the sensors using mayavi see
    :func:`mne.viz.plot_trans`.

    .. versionadded:: 0.12.0

    """
    from .evoked import _rgb
    if kind not in ['topomap', '3d', 'select']:
        raise ValueError("Kind must be 'topomap', '3d' or 'select'. Got %s." %
                         kind)
    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info not %s' % type(info))
    ch_indices = channel_indices_by_type(info)
    allowed_types = ['mag', 'grad', 'eeg', 'seeg', 'ecog']
    if ch_type is None:
        for this_type in allowed_types:
            if _contains_ch_type(info, this_type):
                ch_type = this_type
                break
        picks = ch_indices[ch_type]
    elif ch_type == 'all':
        picks = list()
        for this_type in allowed_types:
            picks += ch_indices[this_type]
    elif ch_type in allowed_types:
        picks = ch_indices[ch_type]
    else:
        raise ValueError("ch_type must be one of %s not %s!" % (allowed_types,
                                                                ch_type))

    if len(picks) == 0:
        raise ValueError('Could not find any channels of type %s.' % ch_type)

    pos = np.asarray([ch['loc'][:3] for ch in info['chs']])[picks]
    ch_names = np.array(info['ch_names'])[picks]
    bads = [idx for idx, name in enumerate(ch_names) if name in info['bads']]
    if ch_groups is None:
        def_colors = _handle_default('color')
        colors = ['red' if i in bads else def_colors[channel_type(info, pick)]
                  for i, pick in enumerate(picks)]
    else:
        if ch_groups in ['position', 'selection']:
            if ch_groups == 'position':
                ch_groups = _divide_to_regions(info, add_stim=False)
                ch_groups = list(ch_groups.values())
            else:
                ch_groups, color_vals = list(), list()
                for selection in _SELECTIONS + _EEG_SELECTIONS:
                    channels = pick_channels(
                        info['ch_names'], read_selection(selection, info=info))
                    ch_groups.append(channels)
            color_vals = np.ones((len(ch_groups), 4))
            for idx, ch_group in enumerate(ch_groups):
                color_picks = [np.where(picks == ch)[0][0] for ch in ch_group
                               if ch in picks]
                if len(color_picks) == 0:
                    continue
                x, y, z = pos[color_picks].T
                color = np.mean(_rgb(info, x, y, z), axis=0)
                color_vals[idx, :3] = color  # mean of spatial color
        else:
            import matplotlib.pyplot as plt
            colors = np.linspace(0, 1, len(ch_groups))
            color_vals = [plt.cm.jet(colors[i]) for i in range(len(ch_groups))]
        if not isinstance(ch_groups, (np.ndarray, list)):
            raise ValueError("ch_groups must be None, 'position', "
                             "'selection', or an array. Got %s." % ch_groups)
        colors = np.zeros((len(picks), 4))
        for pick_idx, pick in enumerate(picks):
            for ind, value in enumerate(ch_groups):
                if pick in value:
                    colors[pick_idx] = color_vals[ind]
                    break
    if kind in ('topomap', 'select'):
        pos = _auto_topomap_coords(info, picks, True)

    title = 'Sensor positions (%s)' % ch_type if title is None else title
    fig = _plot_sensors(pos, colors, bads, ch_names, title, show_names, axes,
                        show, kind == 'select', block=block)
    if kind == 'select':
        return fig, fig.lasso.selection
    return fig


def _onpick_sensor(event, fig, ax, pos, ch_names, show_names):
    """Callback for picked channel in plot_sensors."""
    if event.mouseevent.key == 'control' and fig.lasso is not None:
        for ind in event.ind:
            fig.lasso.select_one(ind)

        return
    if show_names:
        return  # channel names already visible
    ind = event.ind[0]  # Just take the first sensor.
    ch_name = ch_names[ind]

    this_pos = pos[ind]

    # XXX: Bug in matplotlib won't allow setting the position of existing
    # text item, so we create a new one.
    ax.texts.pop(0)
    if len(this_pos) == 3:
        ax.text(this_pos[0], this_pos[1], this_pos[2], ch_name)
    else:
        ax.text(this_pos[0], this_pos[1], ch_name)
    fig.canvas.draw()


def _close_event(event, fig):
    fig.lasso.disconnect()


def _plot_sensors(pos, colors, bads, ch_names, title, show_names, ax, show,
                  select, block):
    """Helper function for plotting sensors."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from .topomap import _check_outlines, _draw_outlines
    edgecolors = np.repeat('black', len(colors))
    edgecolors[bads] = 'red'
    if ax is None:
        fig = plt.figure()
        if pos.shape[1] == 3:
            Axes3D(fig)
            ax = fig.gca(projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if pos.shape[1] == 3:
        ax.text(0, 0, 0, '', zorder=1)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], picker=True, c=colors,
                   s=75, edgecolor=edgecolors, linewidth=2)

        ax.azim = 90
        ax.elev = 0
    else:
        ax.text(0, 0, '', zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None,
                            hspace=None)
        pos, outlines = _check_outlines(pos, 'head')
        _draw_outlines(ax, outlines)

        pts = ax.scatter(pos[:, 0], pos[:, 1], picker=True, c=colors, s=75,
                         edgecolor=edgecolors, linewidth=2)
        if select:
            fig.lasso = SelectFromCollection(ax, pts, ch_names)

    connect_picker = True
    if show_names:
        for idx in range(len(pos)):
            this_pos = pos[idx]
            if pos.shape[1] == 3:
                ax.text(this_pos[0], this_pos[1], this_pos[2], ch_names[idx])
            else:
                ax.text(this_pos[0], this_pos[1], ch_names[idx])
        connect_picker = select
    if connect_picker:
        picker = partial(_onpick_sensor, fig=fig, ax=ax, pos=pos,
                         ch_names=ch_names, show_names=show_names)
        fig.canvas.mpl_connect('pick_event', picker)

    fig.suptitle(title)
    closed = partial(_close_event, fig=fig)
    fig.canvas.mpl_connect('close_event', closed)
    plt_show(show, block=block)
    return fig


def _compute_scalings(scalings, inst):
    """Compute scalings for each channel type automatically.

    Parameters
    ----------
    scalings : dict
        The scalings for each channel type. If any values are
        'auto', this will automatically compute a reasonable
        scaling for that channel type. Any values that aren't
        'auto' will not be changed.
    inst : instance of Raw or Epochs
        The data for which you want to compute scalings. If data
        is not preloaded, this will read a subset of times / epochs
        up to 100mb in size in order to compute scalings.

    Returns
    -------
    scalings : dict
        A scalings dictionary with updated values
    """
    from ..io.base import _BaseRaw
    from ..epochs import _BaseEpochs
    if not isinstance(inst, (_BaseRaw, _BaseEpochs)):
        raise ValueError('Must supply either Raw or Epochs')
    if scalings is None:
        # If scalings is None just return it and do nothing
        return scalings

    ch_types = channel_indices_by_type(inst.info)
    ch_types = dict([(i_type, i_ixs)
                     for i_type, i_ixs in ch_types.items() if len(i_ixs) != 0])
    if scalings == 'auto':
        # If we want to auto-compute everything
        scalings = dict((i_type, 'auto') for i_type in ch_types.keys())
    if not isinstance(scalings, dict):
        raise ValueError('scalings must be a dictionary of ch_type: val pairs,'
                         ' not type %s ' % type(scalings))
    scalings = deepcopy(scalings)

    if inst.preload is False:
        if isinstance(inst, _BaseRaw):
            # Load a window of data from the center up to 100mb in size
            n_times = 1e8 // (len(inst.ch_names) * 8)
            n_times = np.clip(n_times, 1, inst.n_times)
            n_secs = n_times / float(inst.info['sfreq'])
            time_middle = np.mean(inst.times)
            tmin = np.clip(time_middle - n_secs / 2., inst.times.min(), None)
            tmax = np.clip(time_middle + n_secs / 2., None, inst.times.max())
            data = inst._read_segment(tmin, tmax)
        elif isinstance(inst, _BaseEpochs):
            # Load a random subset of epochs up to 100mb in size
            n_epochs = 1e8 // (len(inst.ch_names) * len(inst.times) * 8)
            n_epochs = int(np.clip(n_epochs, 1, len(inst)))
            ixs_epochs = np.random.choice(range(len(inst)), n_epochs, False)
            inst = inst.copy()[ixs_epochs].load_data()
    else:
        data = inst._data
    if isinstance(inst, _BaseEpochs):
        data = inst._data.reshape([len(inst.ch_names), -1])
    # Iterate through ch types and update scaling if ' auto'
    for key, value in scalings.items():
        if value != 'auto':
            continue
        if key not in ch_types.keys():
            raise ValueError("Sensor {0} doesn't exist in data".format(key))
        this_data = data[ch_types[key]]
        scale_factor = np.percentile(this_data.ravel(), [0.5, 99.5])
        scale_factor = np.max(np.abs(scale_factor))
        scalings[key] = scale_factor
    return scalings


class DraggableColorbar(object):
    """Class for enabling interactive colorbar.
    See http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
    """  # noqa
    def __init__(self, cbar, mappable):
        import matplotlib.pyplot as plt
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if
                             hasattr(getattr(plt.cm, i), 'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)
        self.lims = (self.cbar.norm.vmin, self.cbar.norm.vmax)
        self.connect()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)
        self.scroll = self.cbar.patch.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)

    def on_press(self, event):
        """Callback for button press."""
        if event.inaxes != self.cbar.ax:
            return
        self.press = event.y

    def key_press(self, event):
        """Callback for key press."""
        if event.key == 'down':
            self.index += 1
        elif event.key == 'up':
            self.index -= 1
        elif event.key == ' ':  # space key resets scale
            self.cbar.norm.vmin = self.lims[0]
            self.cbar.norm.vmax = self.lims[1]
        else:
            return
        if self.index < 0:
            self.index = len(self.cycle) - 1
        elif self.index >= len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        """Callback for mouse movements."""
        if self.press is None:
            return
        if event.inaxes != self.cbar.ax:
            return
        yprev = self.press
        dy = event.y - yprev
        self.press = event.y
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button == 1:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax -= (perc * scale) * np.sign(dy)
        elif event.button == 3:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax += (perc * scale) * np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_release(self, event):
        """Callback for release."""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_scroll(self, event):
        """Callback for scroll."""
        scale = 1.1 if event.step < 0 else 1. / 1.1
        self.cbar.norm.vmin *= scale
        self.cbar.norm.vmax *= scale
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


class SelectFromCollection(object):
    """Select channels from a matplotlib collection using `LassoSelector`.

    Selected channels are saved in the ``selection`` attribute. This tool
    highlights selected points by fading other points out (i.e., reducing their
    alpha values).

    Notes:
    This tool selects collection objects based on their *origins*
    (i.e., `offsets`). Emits mpl event 'lasso_event' when selection is ready.

    Parameters
    ----------
    ax : Instance of Axes
        Axes to interact with.

    collection : Instance of matplotlib collection
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
        Defaults to 0.3.
    """

    def __init__(self, ax, collection, ch_names, alpha_other=0.3):
        import matplotlib as mpl
        if LooseVersion(mpl.__version__) < LooseVersion('1.2.1'):
            raise ImportError('Interactive selection not possible for '
                              'matplotlib versions < 1.2.1. Upgrade '
                              'matplotlib.')
        from matplotlib.widgets import LassoSelector
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.ch_names = ch_names
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)
        self.fc[:, -1] = self.alpha_other  # deselect in the beginning

        self.lasso = LassoSelector(ax, onselect=self.on_select,
                                   lineprops={'color': 'red', 'linewidth': .5})
        self.selection = list()

    def on_select(self, verts):
        """Callback for selecting a subset from the collection."""
        from matplotlib.path import Path
        if len(verts) <= 3:  # Seems to be a good way to exclude single clicks.
            return

        path = Path(verts)
        inds = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        if self.canvas._key == 'control':  # Appending selection.
            sels = [np.where(self.ch_names == c)[0][0] for c in self.selection]
            inters = set(inds) - set(sels)
            inds = list(inters.union(set(sels) - set(inds)))

        while len(self.selection) > 0:
            self.selection.pop(0)
        self.selection.extend(self.ch_names[inds])
        self.fc[:, -1] = self.alpha_other
        self.fc[inds, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.canvas.callbacks.process('lasso_event')

    def select_one(self, ind):
        """Helper for selecting/deselecting one sensor."""
        ch_name = self.ch_names[ind]
        if ch_name in self.selection:
            sel_ind = self.selection.index(ch_name)
            self.selection.pop(sel_ind)
            this_alpha = self.alpha_other
        else:
            self.selection.append(ch_name)
            this_alpha = 1
        self.fc[ind, -1] = this_alpha
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.canvas.callbacks.process('lasso_event')

    def disconnect(self):
        """Method for disconnecting the lasso selector."""
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
