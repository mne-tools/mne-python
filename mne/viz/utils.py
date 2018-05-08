"""Utility functions for plotting M/EEG data."""
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
from itertools import cycle
from warnings import catch_warnings

from ..channels.layout import _auto_topomap_coords
from ..channels.channels import _contains_ch_type
from ..defaults import _handle_default
from ..io import show_fiff, Info
from ..io.pick import (channel_type, channel_indices_by_type, pick_channels,
                       _pick_data_channels, _DATA_CH_TYPES_SPLIT,
                       pick_info, _picks_by_type)
from ..io.proc_history import _get_rank_sss
from ..io.proj import setup_proj
from ..utils import logger, verbose, set_config, warn

from ..externals.six import string_types
from ..selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                         _divide_to_regions)
from ..annotations import Annotations, _sync_onset


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#473C8B', '#458B74',
          '#CD7F32', '#FF4040', '#ADFF2F', '#8E2323', '#FF1493']

_channel_type_prettyprint = {'eeg': "EEG channel",  'grad': "Gradiometer",
                             'mag': "Magnetometer", 'seeg': "sEEG channel",
                             'eog': "EOG channel", 'ecg': "ECG sensor",
                             'emg': "EMG sensor", 'ecog': "ECoG channel",
                             'misc': "miscellaneous sensor"}


def _setup_vmin_vmax(data, vmin, vmax, norm=False):
    """Handle vmin and vmax parameters."""
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


def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    if show and matplotlib.get_backend() != 'agg':
        (fig or plt).show(**kwargs)


def tight_layout(pad=1.2, h_pad=None, w_pad=None, fig=None):
    """Adjust subplot parameters to give specified padding.

    .. note:: For plotting please use this function instead of
              ``plt.tight_layout``.

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
        with catch_warnings(record=True) as ws:
            fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    except Exception:
        try:
            with catch_warnings(record=True) as ws:
                fig.set_tight_layout(dict(pad=pad, h_pad=h_pad, w_pad=w_pad))
        except Exception:
            warn('Matplotlib function "tight_layout" is not supported.'
                 ' Skipping subplot adjustment.')
            return
    for w in ws:
        w = str(w.message) if hasattr(w, 'message') else w.get_message()
        if not w.startswith('This figure includes Axes'):
            warn(w)


def _check_delayed_ssp(container):
    """Handle interactive SSP selection."""
    if container.proj is True or\
       all(p['active'] for p in container.info['projs']):
        raise RuntimeError('Projs are already applied. Please initialize'
                           ' the data with proj set to False.')
    elif len(container.info['projs']) < 1:
        raise RuntimeError('No projs found in evoked.')


def _validate_if_list_of_axes(axes, obligatory_len=None):
    """Validate whether input is a list/array of axes."""
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
    """Return a colormap similar to that used by mne_analyze.

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
    """Toggle options (projectors) dialog."""
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
    """Perform operations when proj boxes clicked."""
    # read options if possible
    if 'proj_checks' in params:
        bools = [x[0].get_visible() for x in params['proj_checks'].lines]
        for bi, (b, p) in enumerate(zip(bools, params['projs'])):
            # see if they tried to deactivate an active one
            if not b and p['active']:
                bools[bi] = True
    else:
        proj = params.get('apply_proj', True)
        bools = [proj] * len(params['projs'])

    compute_proj = False
    if 'proj_bools' not in params:
        compute_proj = True
    elif not np.array_equal(bools, params['proj_bools']):
        compute_proj = True

    # if projectors changed, update plots
    if compute_proj is True:
        params['plot_update_proj_callback'](params, bools)


def _get_help_text(params):
    """Customize help dialogs text."""
    text, text2 = list(), list()

    text.append(u'\u2190 : \n')  # left arrow
    text.append(u'\u2192 : \n')  # right arrow
    text.append(u'\u2193 : \n')  # down arrow
    text.append(u'\u2191 : \n')  # up arrow
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
            text.insert(6, u'a : \n')
            text2.insert(6, 'Toggle annotation mode\n')
            text.insert(7, u'b : \n')
            text2.insert(7, 'Toggle butterfly plot on/off\n')
            if 'fig_selection' not in params:
                text2.insert(10, 'Reduce the number of channels per view\n')
                text2.insert(11, 'Increase the number of channels per view\n')
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
    import matplotlib.pyplot as plt
    if n_cells == 1:
        nrow = ncol = 1
    elif n_cells <= max_col:
        nrow, ncol = 1, n_cells
    else:
        nrow, ncol = int(math.ceil(n_cells / float(max_col))), max_col

    fig, axes = plt.subplots(nrow, ncol, figsize=(1.3 * ncol + 1,
                                                  1.5 * nrow + 1))
    axes = [axes] if ncol == nrow == 1 else axes.flatten()
    for ax in axes[n_cells:]:  # hide unused axes
        # XXX: Previously done by ax.set_visible(False), but because of mpl
        # bug, we just hide the frame.
        from .topomap import _hide_frame
        _hide_frame(ax)
    return fig, axes


def _draw_proj_checkbox(event, params, draw_current_state=True):
    """Toggle options (projectors) dialog."""
    from matplotlib import widgets
    projs = params['projs']
    # turn on options dialog

    labels = [p['desc'] for p in projs]
    actives = ([p['active'] for p in projs] if draw_current_state else
               params.get('proj_bools', [params['apply_proj']] * len(projs)))

    width = max([4., max([len(p['desc']) for p in projs]) / 6.0 + 0.5])
    height = len(projs) / 6.0 + 1.5
    fig_proj = figure_nobar(figsize=(width, height))
    fig_proj.canvas.set_window_title('SSP projection vectors')
    params['fig_proj'] = fig_proj  # necessary for proper toggling
    ax_temp = fig_proj.add_axes((0, 0, 1, 0.8), frameon=False)
    ax_temp.set_title('Projectors marked with "X" are active')

    proj_checks = widgets.CheckButtons(ax_temp, labels=labels, actives=actives)
    # make edges around checkbox areas
    [rect.set_edgecolor('0.5') for rect in proj_checks.rectangles]
    [rect.set_linewidth(1.) for rect in proj_checks.rectangles]

    # change already-applied projectors to red
    for ii, p in enumerate(projs):
        if p['active']:
            for x in proj_checks.lines[ii]:
                x.set_color('r')
    # make minimal size
    # pass key presses from option dialog over

    proj_checks.on_clicked(partial(_toggle_proj, params=params))
    params['proj_checks'] = proj_checks
    fig_proj.canvas.mpl_connect('key_press_event', _key_press)

    # this should work for non-test cases
    try:
        fig_proj.canvas.draw()
        plt_show(fig=fig_proj, warn=False)
    except Exception:
        pass


def _layout_figure(params):
    """Set figure layout. Shared with raw and epoch plots."""
    size = params['fig'].get_size_inches() * params['fig'].dpi
    scroll_width = 25
    hscroll_dist = 25
    vscroll_dist = 10
    l_border = 100
    r_border = 10
    t_border = 35
    b_border = 45

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
    """Compare the contents of two fiff files using diff and show_fiff.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

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
    """Make matplotlib figure with no toolbar."""
    from matplotlib import rcParams, pyplot as plt
    old_val = rcParams['toolbar']
    try:
        rcParams['toolbar'] = 'none'
        fig = plt.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        cbs = list(fig.canvas.callbacks.callbacks['key_press_event'].keys())
        for key in cbs:
            fig.canvas.callbacks.disconnect(key)
    finally:
        rcParams['toolbar'] = old_val
    return fig


def _helper_raw_resize(event, params):
    """Resize."""
    size = ','.join([str(s) for s in params['fig'].get_size_inches()])
    set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
    _layout_figure(params)


def _plot_raw_onscroll(event, params, len_channels=None):
    """Interpret scroll events."""
    if 'fig_selection' in params:
        if params['butterfly']:
            return
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
    """Deal with the vertical shift of the viewport."""
    if params['ch_start'] + params['n_channels'] > len_channels:
        params['ch_start'] = len_channels - params['n_channels']
    if params['ch_start'] < 0:
        params['ch_start'] = 0
    params['plot_fun']()


def _plot_raw_time(value, params):
    """Deal with changed time value."""
    info = params['info']
    max_times = params['n_times'] / float(info['sfreq']) + \
        params['first_time'] - params['duration']
    if value > max_times:
        value = params['n_times'] / float(info['sfreq']) + \
            params['first_time'] - params['duration']
    if value < params['first_time']:
        value = params['first_time']
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)


def _radio_clicked(label, params):
    """Handle radio buttons in selection dialog."""
    from .evoked import _rgb

    # First the selection dialog.
    labels = [l._text for l in params['fig_selection'].radio.labels]
    idx = labels.index(label)
    params['fig_selection'].radio._active_idx = idx
    channels = params['selections'][label]
    ax_topo = params['fig_selection'].get_axes()[1]
    types = np.array([], dtype=int)
    for this_type in _DATA_CH_TYPES_SPLIT:
        if this_type in params['types']:
            types = np.concatenate(
                [types, np.where(np.array(params['types']) == this_type)[0]])
    colors = np.zeros((len(types), 4))  # alpha = 0 by default
    locs3d = np.array([ch['loc'][:3] for ch in params['info']['chs']])
    x, y, z = locs3d.T
    color_vals = _rgb(x, y, z)
    for color_idx, pick in enumerate(types):
        if pick in channels:  # set color and alpha = 1
            colors[color_idx] = np.append(color_vals[pick], 1.)
    ax_topo.collections[0]._facecolors = colors
    params['fig_selection'].canvas.draw()

    if params['butterfly']:
        return
    # Then the plotting window.
    params['ax_vscroll'].set_visible(True)
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


def _get_active_radiobutton(radio):
    """Find out active radio button."""
    # XXX: In mpl 1.5 you can do: fig.radio.value_selected
    colors = np.array([np.sum(circle.get_facecolor()) for circle
                       in radio.circles])
    return np.where(colors < 4.0)[0][0]  # return idx where color != white


def _set_annotation_radio_button(idx, params):
    """Set active button."""
    radio = params['fig_annotation'].radio
    for circle in radio.circles:
        circle.set_facecolor('white')
    radio.circles[idx].set_facecolor('#cccccc')
    _annotation_radio_clicked('', radio, params['ax'].selector)


def _set_radio_button(idx, params):
    """Set radio button."""
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
    elif idx > 0:
        _set_radio_button(idx - 1, params)


def _handle_change_selection(event, params):
    """Handle clicks on vertical scrollbar using selections."""
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
    """Interpret key presses."""
    import matplotlib.pyplot as plt
    if event.key == params['close_key']:
        plt.close(params['fig'])
        if params['fig_annotation'] is not None:
            plt.close(params['fig_annotation'])
    elif event.key == 'down':
        if 'fig_selection' in params.keys():
            _change_channel_group(-1, params)
            return
        elif params['butterfly']:
            return
        params['ch_start'] += params['n_channels']
        _channels_changed(params, len(params['inds']))
    elif event.key == 'up':
        if 'fig_selection' in params.keys():
            _change_channel_group(1, params)
            return
        elif params['butterfly']:
            return
        params['ch_start'] -= params['n_channels']
        _channels_changed(params, len(params['inds']))
    elif event.key == 'right':
        value = params['t_start'] + params['duration'] / 4
        _plot_raw_time(value, params)
        params['update_fun']()
        params['plot_fun']()
    elif event.key == 'shift+right':
        value = params['t_start'] + params['duration']
        _plot_raw_time(value, params)
        params['update_fun']()
        params['plot_fun']()
    elif event.key == 'left':
        value = params['t_start'] - params['duration'] / 4
        _plot_raw_time(value, params)
        params['update_fun']()
        params['plot_fun']()
    elif event.key == 'shift+left':
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
    elif event.key == 'a':
        if 'ica' in params.keys():
            return
        if params['fig_annotation'] is None:
            _setup_annotation_fig(params)
        else:
            params['fig_annotation'].canvas.close_event()
    elif event.key == 'b':
        _setup_butterfly(params)
    elif event.key == 'w':
        params['use_noise_cov'] = not params['use_noise_cov']
        params['plot_update_proj_callback'](params, None)


def _setup_annotation_fig(params):
    """Initialize the annotation figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, SpanSelector, Button
    if params['fig_annotation'] is not None:
        params['fig_annotation'].canvas.close_event()
    if params['raw'].annotations is None:
        params['raw'].annotations = Annotations(list(), list(), list())
    annotations = params['raw'].annotations
    labels = list(set(annotations.description))
    labels = np.union1d(labels, params['added_label'])
    fig = figure_nobar(figsize=(4.5, 2.75 + len(labels) * 0.75))
    fig.patch.set_facecolor('white')
    ax = plt.subplot2grid((len(labels) + 2, 2), (0, 0),
                          rowspan=max(len(labels), 1),
                          colspan=2, frameon=False)
    ax.set_title('Labels')
    ax.set_aspect('equal')
    button_ax = plt.subplot2grid((len(labels) + 2, 2), (len(labels), 1),
                                 rowspan=1, colspan=1)
    label_ax = plt.subplot2grid((len(labels) + 2, 2), (len(labels), 0),
                                rowspan=1, colspan=1)
    plt.axis('off')
    text_ax = plt.subplot2grid((len(labels) + 2, 2), (len(labels) + 1, 0),
                               rowspan=1, colspan=2)
    text_ax.text(0.5, 0.9, 'Left click & drag - Create/modify annotation\n'
                           'Right click - Delete annotation\n'
                           'Letter/number keys - Add character\n'
                           'Backspace - Delete character\n'
                           'Esc - Close window/exit annotation mode', va='top',
                 ha='center')
    plt.axis('off')

    annotations_closed = partial(_annotations_closed, params=params)
    fig.canvas.mpl_connect('close_event', annotations_closed)
    fig.canvas.set_window_title('Annotations')
    fig.radio = RadioButtons(ax, labels, activecolor='#cccccc')
    radius = 0.15
    circles = fig.radio.circles
    for circle, label in zip(circles, fig.radio.labels):
        circle.set_edgecolor(params['segment_colors'][label.get_text()])
        circle.set_linewidth(4)
        circle.set_radius(radius / (len(labels)))
        label.set_x(circle.center[0] + (radius + 0.1) / len(labels))
    col = 'r' if len(fig.radio.circles) < 1 else circles[0].get_edgecolor()
    fig.canvas.mpl_connect('key_press_event', partial(
        _change_annotation_description, params=params))
    fig.button = Button(button_ax, 'Add label')
    fig.label = label_ax.text(0.5, 0.5, '"BAD_"', va='center', ha='center')
    fig.button.on_clicked(partial(_onclick_new_label, params=params))
    plt_show(fig=fig)
    params['fig_annotation'] = fig

    ax = params['ax']
    cb_onselect = partial(_annotate_select, params=params)
    selector = SpanSelector(ax, cb_onselect, 'horizontal', minspan=.1,
                            rectprops=dict(alpha=0.5, facecolor=col))
    if len(labels) == 0:
        selector.active = False
    params['ax'].selector = selector
    if LooseVersion(mpl.__version__) < LooseVersion('1.5'):
        # XXX: Hover event messes up callback ids in old mpl.
        warn('Modifying existing annotations is not possible for '
             'matplotlib versions < 1.4. Upgrade matplotlib.')
        return
    hover_callback = partial(_on_hover, params=params)
    params['hover_callback'] = params['fig'].canvas.mpl_connect(
        'motion_notify_event', hover_callback)

    radio_clicked = partial(_annotation_radio_clicked, radio=fig.radio,
                            selector=selector)
    fig.radio.on_clicked(radio_clicked)


def _onclick_new_label(event, params):
    """Add new description on button press."""
    text = params['fig_annotation'].label.get_text()[1:-1]
    params['added_label'].append(text)
    _setup_annotation_colors(params)
    _setup_annotation_fig(params)
    idx = [label.get_text() for label in
           params['fig_annotation'].radio.labels].index(text)
    _set_annotation_radio_button(idx, params)


def _mouse_click(event, params):
    """Handle mouse clicks."""
    if event.button not in (1, 3):
        return
    if event.button == 3:
        if params['fig_annotation'] is None:
            return
        raw = params['raw']
        if np.any([c.contains(event)[0] for c in params['ax'].collections]):
            xdata = event.xdata - params['first_time']
            onset = _sync_onset(raw, raw.annotations.onset)
            ends = onset + raw.annotations.duration
            ann_idx = np.where((xdata > onset) & (xdata < ends))[0]
            raw.annotations.delete(ann_idx)  # only first one deleted
        _remove_segment_line(params)
        _plot_annotations(raw, params)
        params['plot_fun']()
        return

    if event.inaxes is None:  # check if channel label is clicked
        if params['n_channels'] > 100:
            return
        ax = params['ax']
        ylim = ax.get_ylim()
        pos = ax.transData.inverted().transform((event.x, event.y))
        if pos[0] > params['t_start'] or pos[1] < 0 or pos[1] > ylim[0]:
            return
        params['label_click_fun'](pos)
    # vertical scrollbar changed
    elif event.inaxes == params['ax_vscroll']:
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
    """Color channels in selection topomap when selecting bads."""
    for t in _DATA_CH_TYPES_SPLIT:
        if t in params['types']:
            types = np.where(np.array(params['types']) == t)[0]
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
    """Find all indices when using selections."""
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


def _draw_vert_line(xdata, params):
    """Draw vertical line."""
    params['ax_vertline'].set_data(xdata, np.array(params['ax'].get_ylim()))
    params['ax_hscroll_vertline'].set_data(xdata, np.array([0., 1.]))
    params['vertline_t'].set_text('%0.3f' % xdata[0])


def _select_bads(event, params, bads):
    """Select bad channels onpick. Returns updated bads list."""
    # trade-off, avoid selecting more than one channel when drifts are present
    # however for clean data don't click on peaks but on flat segments
    if params['butterfly']:
        _draw_vert_line(np.array([event.xdata] * 2), params)
        return bads

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
        _draw_vert_line(np.array([event.xdata] * 2), params)

    return bads


def _onclick_help(event, params):
    """Draw help window."""
    import matplotlib.pyplot as plt
    text, text2 = _get_help_text(params)

    width = 6
    height = 5

    fig_help = figure_nobar(figsize=(width, height), dpi=80)
    fig_help.canvas.set_window_title('Help')
    params['fig_help'] = fig_help
    ax = plt.subplot2grid((8, 5), (0, 0), colspan=5)
    ax.set_title('Keyboard shortcuts')
    plt.axis('off')
    ax1 = plt.subplot2grid((8, 5), (1, 0), rowspan=7, colspan=2)
    ax1.set_yticklabels(list())
    plt.text(0.99, 1, text, fontname='STIXGeneral', va='top', ha='right')
    plt.axis('off')

    ax2 = plt.subplot2grid((8, 5), (1, 2), rowspan=7, colspan=3)
    ax2.set_yticklabels(list())
    plt.text(0, 1, text2, fontname='STIXGeneral', va='top')
    plt.axis('off')

    fig_help.canvas.mpl_connect('key_press_event', _key_press)

    tight_layout(fig=fig_help)
    # this should work for non-test cases
    try:
        fig_help.canvas.draw()
        plt_show(fig=fig_help, warn=False)
    except Exception:
        pass


def _key_press(event):
    """Handle key press in dialog."""
    import matplotlib.pyplot as plt
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def _setup_browser_offsets(params, n_channels):
    """Compute viewport height and adjust offsets."""
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
    """Display an image so you can click on it and store x/y positions.

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

    def __init__(self, imdata, **kwargs):  # noqa: D102
        """Display the image for clicking."""
        from matplotlib.pyplot import figure
        self.coords = []
        self.imdata = imdata
        self.fig = figure()
        self.ax = self.fig.add_subplot(111)
        self.ymax = self.imdata.shape[0]
        self.xmax = self.imdata.shape[1]
        self.im = self.ax.imshow(imdata,
                                 extent=(0, self.xmax, 0, self.ymax),
                                 picker=True, **kwargs)
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('pick_event', self.onclick)
        plt_show(block=True)

    def onclick(self, event):
        """Handle Mouse clicks.

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
        if len(self.coords) == 0:
            raise ValueError('No coordinates found, make sure you click '
                             'on the image that is first shown.')
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
    """Fake a click at a relative point within axes."""
    if xform == 'ax':
        x, y = ax.transAxes.transform_point(point)
    elif xform == 'data':
        x, y = ax.transData.transform_point(point)
    elif xform == 'pix':
        x, y = point
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
    """Find peaks from evoked data.

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


def _process_times(inst, use_times, n_peaks=None, few=False):
    """Return a list of times for topomaps."""
    if isinstance(use_times, string_types):
        if use_times == 'interactive':
            use_times, n_peaks = 'peaks', 1
        if use_times == 'peaks':
            if n_peaks is None:
                n_peaks = min(3 if few else 7, len(inst.times))
            use_times = _find_peaks(inst, n_peaks)
        elif use_times == 'auto':
            if n_peaks is None:
                n_peaks = min(5 if few else 10, len(use_times))
            use_times = np.linspace(inst.times[0], inst.times[-1], n_peaks)
        else:
            raise ValueError("Got an unrecognized method for `times`. Only "
                             "'peaks', 'auto' and 'interactive' are supported "
                             "(or directly passing numbers).")
    elif np.isscalar(use_times):
        use_times = [use_times]

    use_times = np.array(use_times, float)

    if use_times.ndim != 1:
        raise ValueError('times must be 1D, got %d dimensions'
                         % use_times.ndim)
    if len(use_times) > 20:
        raise RuntimeError('Too many plots requested. Please pass fewer '
                           'than 20 time instants.')

    return use_times


def plot_sensors(info, kind='topomap', ch_type=None, title=None,
                 show_names=False, ch_groups=None, to_sphere=True, axes=None,
                 block=False, show=True):
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
    show_names : bool | array of str
        Whether to display all channel names. If an array, only the channel
        names in the array are shown. Defaults to False.
    ch_groups : 'position' | array of shape (ch_groups, picks) | None
        Channel groups for coloring the sensors. If None (default), default
        coloring scheme is used. If 'position', the sensors are divided
        into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
        array, the channels are divided by picks given in the array.

        .. versionadded:: 0.13.0

    to_sphere : bool
        Whether to project the 3d locations to a sphere. When False, the
        sensor array appears similar as to looking downwards straight above the
        subject's head. Has no effect when kind='3d'. Defaults to True.

        .. versionadded:: 0.14.0

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
    :func:`mne.viz.plot_alignment`.

    .. versionadded:: 0.12.0

    """
    from .evoked import _rgb
    if kind not in ['topomap', '3d', 'select']:
        raise ValueError("Kind must be 'topomap', '3d' or 'select'. Got %s." %
                         kind)
    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info not %s' % type(info))
    ch_indices = channel_indices_by_type(info)
    allowed_types = _DATA_CH_TYPES_SPLIT
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
                color = np.mean(_rgb(x, y, z), axis=0)
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
        pos = _auto_topomap_coords(info, picks, True, to_sphere=to_sphere)

    title = 'Sensor positions (%s)' % ch_type if title is None else title
    fig = _plot_sensors(pos, colors, bads, ch_names, title, show_names, axes,
                        show, kind == 'select', block=block,
                        to_sphere=to_sphere)
    if kind == 'select':
        return fig, fig.lasso.selection
    return fig


def _onpick_sensor(event, fig, ax, pos, ch_names, show_names):
    """Pick a channel in plot_sensors."""
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
    """Listen for sensor plotter close event."""
    if fig.lasso is not None:
        fig.lasso.disconnect()


def _plot_sensors(pos, colors, bads, ch_names, title, show_names, ax, show,
                  select, block, to_sphere):
    """Plot sensors."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from .topomap import _check_outlines, _draw_outlines
    edgecolors = np.repeat('black', len(colors))
    edgecolors[bads] = 'red'
    if ax is None:
        fig = plt.figure(figsize=(max(plt.rcParams['figure.figsize']),) * 2)
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
        ax.xaxis.set_label_text('x')
        ax.yaxis.set_label_text('y')
        ax.zaxis.set_label_text('z')
    else:
        ax.text(0, 0, '', zorder=1)
        # Equal aspect for 3D looks bad, so only use for 2D
        ax.set(xticks=[], yticks=[], aspect='equal')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None,
                            hspace=None)
        if to_sphere:
            pos, outlines = _check_outlines(pos, 'head')
        else:
            pos, outlines = _check_outlines(pos, np.array([0.5, 0.5]),
                                            {'center': (0, 0),
                                             'scale': (4.5, 4.5)})
        _draw_outlines(ax, outlines)

        pts = ax.scatter(pos[:, 0], pos[:, 1], picker=True, c=colors, s=25,
                         edgecolor=edgecolors, linewidth=2, clip_on=False)

        if select:
            fig.lasso = SelectFromCollection(ax, pts, ch_names)
        else:
            fig.lasso = None

        ax.axis("off")  # remove border around figure

    connect_picker = True
    if show_names:
        if isinstance(show_names, (list, np.ndarray)):  # only given channels
            indices = [list(ch_names).index(name) for name in show_names]
        else:  # all channels
            indices = range(len(pos))
        for idx in indices:
            this_pos = pos[idx]
            if pos.shape[1] == 3:
                ax.text(this_pos[0], this_pos[1], this_pos[2], ch_names[idx])
            else:
                ax.text(this_pos[0] + 0.015, this_pos[1], ch_names[idx])
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
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    if not isinstance(inst, (BaseRaw, BaseEpochs)):
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
        if isinstance(inst, BaseRaw):
            # Load a window of data from the center up to 100mb in size
            n_times = 1e8 // (len(inst.ch_names) * 8)
            n_times = np.clip(n_times, 1, inst.n_times)
            n_secs = n_times / float(inst.info['sfreq'])
            time_middle = np.mean(inst.times)
            tmin = np.clip(time_middle - n_secs / 2., inst.times.min(), None)
            tmax = np.clip(time_middle + n_secs / 2., None, inst.times.max())
            data = inst._read_segment(tmin, tmax)
        elif isinstance(inst, BaseEpochs):
            # Load a random subset of epochs up to 100mb in size
            n_epochs = 1e8 // (len(inst.ch_names) * len(inst.times) * 8)
            n_epochs = int(np.clip(n_epochs, 1, len(inst)))
            ixs_epochs = np.random.choice(range(len(inst)), n_epochs, False)
            inst = inst.copy()[ixs_epochs].load_data()
    else:
        data = inst._data
    if isinstance(inst, BaseEpochs):
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


def _setup_cmap(cmap, n_axes=1, norm=False):
    """Set color map interactivity."""
    if cmap == 'interactive':
        cmap = ('Reds' if norm else 'RdBu_r', True)
    elif not isinstance(cmap, tuple):
        if cmap is None:
            cmap = 'Reds' if norm else 'RdBu_r'
        cmap = (cmap, False if n_axes > 2 else True)
    return cmap


def _prepare_joint_axes(n_maps, figsize=None):
    """Prepare axes for topomaps and colorbar in joint plot figure.

    Parameters
    ----------
    n_maps: int
        Number of topomaps to include in the figure
    figsize: tuple
        Figure size, see plt.figsize

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with initialized axes
    main_ax: matplotlib.axes._subplots.AxesSubplot
        Axes in which to put the main plot
    map_ax: list
        List of axes for each topomap
    cbar_ax: matplotlib.axes._subplots.AxesSubplot
        Axes for colorbar next to topomaps
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    main_ax = fig.add_subplot(212)
    ts = n_maps + 2
    map_ax = [plt.subplot(4, ts, x + 2 + ts) for x in range(n_maps)]
    # Position topomap subplots on the second row, starting on the
    # second column
    cbar_ax = plt.subplot(4, 5 * (ts + 1), 10 * (ts + 1))
    # Position colorbar at the very end of a more finely divided
    # second row of subplots
    return fig, main_ax, map_ax, cbar_ax


class DraggableColorbar(object):
    """Enable interactive colorbar.

    See http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
    """  # noqa: E501

    def __init__(self, cbar, mappable):  # noqa: D102
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
        """Handle button press."""
        if event.inaxes != self.cbar.ax:
            return
        self.press = event.y

    def key_press(self, event):
        """Handle key press."""
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
        """Handle mouse movements."""
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
        """Handle release."""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_scroll(self, event):
        """Handle scroll."""
        scale = 1.1 if event.step < 0 else 1. / 1.1
        self.cbar.norm.vmin *= scale
        self.cbar.norm.vmax *= scale
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


class SelectFromCollection(object):
    """Select channels from a matplotlib collection using ``LassoSelector``.

    Selected channels are saved in the ``selection`` attribute. This tool
    highlights selected points by fading other points out (i.e., reducing their
    alpha values).

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

    Notes
    -----
    This tool selects collection objects based on their *origins*
    (i.e., `offsets`). Emits mpl event 'lasso_event' when selection is ready.
    """

    def __init__(self, ax, collection, ch_names,
                 alpha_other=0.3):  # noqa: D102
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
        """Select a subset from the collection."""
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
        """Select or deselect one sensor."""
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
        """Disconnect the lasso selector."""
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def _annotate_select(vmin, vmax, params):
    """Handle annotation span selector."""
    raw = params['raw']
    onset = _sync_onset(raw, vmin, True) - params['first_time']
    duration = vmax - vmin
    active_idx = _get_active_radiobutton(params['fig_annotation'].radio)
    description = params['fig_annotation'].radio.labels[active_idx].get_text()
    if raw.annotations is None:
        raw.annotations = Annotations([], [], [])
    _merge_annotations(onset, onset + duration, description,
                       raw.annotations)
    _plot_annotations(params['raw'], params)
    params['plot_fun']()


def _plot_annotations(raw, params):
    """Set up annotations for plotting in raw browser."""
    if raw.annotations is None:
        return

    while len(params['ax_hscroll'].collections) > 0:
        params['ax_hscroll'].collections.pop()
    segments = list()
    # sort the segments by start time
    ann_order = raw.annotations.onset.argsort(axis=0)
    descriptions = raw.annotations.description[ann_order]

    _setup_annotation_colors(params)
    for idx, onset in enumerate(raw.annotations.onset[ann_order]):
        annot_start = _sync_onset(raw, onset) + params['first_time']
        annot_end = annot_start + raw.annotations.duration[ann_order][idx]
        segments.append([annot_start, annot_end])
        dscr = descriptions[idx]
        params['ax_hscroll'].fill_betweenx(
            (0., 1.), annot_start, annot_end, alpha=0.3,
            color=params['segment_colors'][dscr])
    # Do not adjust half a sample backward (even though this would make it
    # clearer what is included) because this breaks click-drag functionality
    params['segments'] = np.array(segments)
    params['annot_description'] = descriptions


def _setup_annotation_colors(params):
    """Set up colors for annotations."""
    raw = params['raw']
    segment_colors = params.get('segment_colors', dict())
    # sort the segments by start time
    if raw.annotations is not None:
        ann_order = raw.annotations.onset.argsort(axis=0)
        descriptions = raw.annotations.description[ann_order]
    else:
        descriptions = list()
    color_keys = np.union1d(descriptions, params['added_label'])
    color_cycle = cycle(np.delete(COLORS, COLORS.index('r')))  # no red
    for key, color in segment_colors.items():
        assert color in COLORS
        if color != 'r' and key in color_keys:
            next(color_cycle)
    for idx, key in enumerate(color_keys):
        if key in segment_colors:
            continue
        elif key.lower().startswith('bad') or key.lower().startswith('edge'):
            segment_colors[key] = 'r'
        else:
            segment_colors[key] = next(color_cycle)
    params['segment_colors'] = segment_colors


def _annotations_closed(event, params):
    """Clean up on annotation dialog close."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.close(params['fig_annotation'])
    if params['ax'].selector is not None:
        params['ax'].selector.disconnect_events()
        params['ax'].selector = None
    params['fig_annotation'] = None
    if params['segment_line'] is not None:
        params['segment_line'].remove()
        params['segment_line'] = None
    if LooseVersion(mpl.__version__) >= LooseVersion('1.5'):
        params['fig'].canvas.mpl_disconnect(params['hover_callback'])
    params['fig_annotation'] = None
    params['fig'].canvas.draw()


def _on_hover(event, params):
    """Handle hover event."""
    from matplotlib.patheffects import Stroke, Normal
    if (event.button is not None or
            event.inaxes != params['ax'] or event.xdata is None):
        return
    for coll in params['ax'].collections:
        if coll.contains(event)[0]:
            path = coll.get_paths()
            assert len(path) == 1
            path = path[0]
            color = coll.get_edgecolors()[0]
            mn = path.vertices[:, 0].min()
            mx = path.vertices[:, 0].max()
            # left/right line
            x = mn if abs(event.xdata - mn) < abs(event.xdata - mx) else mx
            mask = path.vertices[:, 0] == x
            ylim = params['ax'].get_ylim()

            def drag_callback(x0):
                path.vertices[mask, 0] = x0

            if params['segment_line'] is None:
                modify_callback = partial(_annotation_modify, params=params)
                line = params['ax'].plot([x, x], ylim, color=color,
                                         linewidth=2., picker=5.)[0]
                dl = DraggableLine(line, modify_callback, drag_callback)
                params['segment_line'] = dl
            else:
                params['segment_line'].set_x(x)
                params['segment_line'].drag_callback = drag_callback
            line = params['segment_line'].line
            pe = [Stroke(linewidth=4, foreground=color, alpha=0.5), Normal()]
            line.set_path_effects(pe if line.contains(event)[0] else pe[1:])
            params['vertline_t'].set_text('%.3f' % x)
            params['ax_vertline'].set_data(0,
                                           np.array(params['ax'].get_ylim()))
            params['ax'].selector.active = False
            params['fig'].canvas.draw()
            return
    _remove_segment_line(params)


def _remove_segment_line(params):
    """Remove annotation line from the view."""
    if params['segment_line'] is not None:
        params['segment_line'].remove()
        params['segment_line'] = None
        params['ax'].selector.active = True
        params['vertline_t'].set_text('')


def _annotation_modify(old_x, new_x, params):
    """Modify annotation."""
    raw = params['raw']

    segment = np.array(np.where(params['segments'] == old_x))
    if segment.shape[1] == 0:
        return
    annotations = params['raw'].annotations
    idx = [segment[0][0], segment[1][0]]
    onset = _sync_onset(raw, params['segments'][idx[0]][0], True)
    ann_idx = np.where(annotations.onset == onset - params['first_time'])[0]
    if idx[1] == 0:  # start of annotation
        onset = _sync_onset(raw, new_x, True) - params['first_time']
        duration = annotations.duration[ann_idx] + old_x - new_x
    else:  # end of annotation
        onset = annotations.onset[ann_idx]
        duration = _sync_onset(raw, new_x, True) - onset - params['first_time']

    if duration < 0:
        onset += duration
        duration *= -1.

    _merge_annotations(onset, onset + duration,
                       annotations.description[ann_idx], annotations, ann_idx)
    _plot_annotations(params['raw'], params)
    _remove_segment_line(params)

    params['plot_fun']()


def _merge_annotations(start, stop, description, annotations, current=()):
    """Handle drawn annotations."""
    ends = annotations.onset + annotations.duration
    idx = np.intersect1d(np.where(ends >= start)[0],
                         np.where(annotations.onset <= stop)[0])
    idx = np.intersect1d(idx,
                         np.where(annotations.description == description)[0])
    new_idx = np.setdiff1d(idx, current)  # don't include modified annotation
    end = max(np.append((annotations.onset[new_idx] +
                         annotations.duration[new_idx]), stop))
    onset = min(np.append(annotations.onset[new_idx], start))
    duration = end - onset
    annotations.delete(idx)
    annotations.append(onset, duration, description)


def _change_annotation_description(event, params):
    """Handle keys in annotation dialog."""
    import matplotlib.pyplot as plt
    fig = event.canvas.figure
    text = fig.label.get_text()[1:-1]
    if event.key == 'backspace':
        text = text[:-1]
    elif event.key == 'escape':
        plt.close(fig)
        return
    elif event.key == 'enter':
        _onclick_new_label(event, params)
    elif len(event.key) > 1 or event.key == ';':  # ignore modifier keys
        return
    else:
        text = text + event.key
    fig.label.set_text('"' + text + '"')
    fig.canvas.draw()


def _annotation_radio_clicked(label, radio, selector):
    """Handle annotation radio buttons."""
    idx = _get_active_radiobutton(radio)
    color = radio.circles[idx].get_edgecolor()
    selector.rect.set_color(color)
    selector.rectprops.update(dict(facecolor=color))


def _setup_butterfly(params):
    """Set butterfly view of raw plotter."""
    from .raw import _setup_browser_selection
    if 'ica' in params:
        return
    butterfly = not params['butterfly']
    ax = params['ax']
    params['butterfly'] = butterfly
    if butterfly:
        types = np.array(params['types'])[params['orig_inds']]
        if params['group_by'] in ['type', 'original']:
            inds = params['inds']
            labels = [t for t in _DATA_CH_TYPES_SPLIT + ['eog', 'ecg']
                      if t in types] + ['misc']
            ticks = np.arange(5, 5 * (len(labels) + 1), 5)
            offs = {l: t for (l, t) in zip(labels, ticks)}

            params['offsets'] = np.zeros(len(params['types']))
            for ind in inds:
                params['offsets'][ind] = offs.get(params['types'][ind],
                                                  5 * (len(labels)))
            ax.set_yticks(ticks)
            params['ax'].set_ylim(5 * (len(labels) + 1), 0)
            ax.set_yticklabels(labels)
        else:
            if 'selections' not in params:
                params['selections'] = _setup_browser_selection(
                    params['raw'], 'position', selector=False)
            sels = params['selections']
            selections = _SELECTIONS[1:]  # Vertex not used
            if ('Misc' in sels and len(sels['Misc']) > 0):
                selections += ['Misc']
            if params['group_by'] == 'selection' and 'eeg' in types:
                for sel in _EEG_SELECTIONS:
                    if sel in sels:
                        selections += [sel]
            picks = list()
            for selection in selections:
                picks.append(sels.get(selection, list()))
            labels = ax.yaxis.get_ticklabels()
            for label in labels:
                label.set_visible(True)
            ylim = (5. * len(picks), 0.)
            ax.set_ylim(ylim)
            offset = ylim[0] / (len(picks) + 1)
            ticks = np.arange(0, ylim[0], offset)
            ticks = [ticks[x] if x < len(ticks) else 0 for x in range(20)]
            ax.set_yticks(ticks)
            offsets = np.zeros(len(params['types']))

            for group_idx, group in enumerate(picks):
                for idx, pick in enumerate(group):
                    offsets[pick] = offset * (group_idx + 1)
            params['inds'] = params['orig_inds'].copy()
            params['offsets'] = offsets
            ax.set_yticklabels([''] + selections, color='black', rotation=45,
                               va='top')
    else:
        params['inds'] = params['orig_inds'].copy()
        if 'fig_selection' not in params:
            for idx in np.arange(params['n_channels'], len(params['lines'])):
                params['lines'][idx].set_xdata([])
                params['lines'][idx].set_ydata([])
        _setup_browser_offsets(params, max([params['n_channels'], 1]))
        if 'fig_selection' in params:
            radio = params['fig_selection'].radio
            active_idx = _get_active_radiobutton(radio)
            _radio_clicked(radio.labels[active_idx]._text, params)
    # For now, italics only work in non-grouped mode
    _set_ax_label_style(ax, params, italicize=not butterfly)
    params['ax_vscroll'].set_visible(not butterfly)
    params['plot_fun']()


def _connection_line(x, fig, sourceax, targetax, y=1.,
                     y_source_transform="transAxes"):
    """Connect source and target plots with a line.

    Connect source and target plots with a line, such as time series
    (source) and topolots (target). Primarily used for plot_joint
    functions.
    """
    from matplotlib.lines import Line2D
    trans_fig = fig.transFigure
    trans_fig_inv = fig.transFigure.inverted()

    xt, yt = trans_fig_inv.transform(targetax.transAxes.transform([.5, 0.]))
    xs, _ = trans_fig_inv.transform(sourceax.transData.transform([x, 0.]))
    _, ys = trans_fig_inv.transform(getattr(sourceax, y_source_transform
                                            ).transform([0., y]))

    return Line2D((xt, xs), (yt, ys), transform=trans_fig, color='grey',
                  linestyle='-', linewidth=1.5, alpha=.66, zorder=1,
                  clip_on=False)


class DraggableLine(object):
    """Custom matplotlib line for moving around by drag and drop.

    Parameters
    ----------
    line : instance of matplotlib Line2D
        Line to add interactivity to.
    callback : function
        Callback to call when line is released.
    """

    def __init__(self, line, modify_callback, drag_callback):  # noqa: D102
        self.line = line
        self.press = None
        self.x0 = line.get_xdata()[0]
        self.modify_callback = modify_callback
        self.drag_callback = drag_callback
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def set_x(self, x):
        """Repoisition the line."""
        self.line.set_xdata([x, x])
        self.x0 = x

    def on_press(self, event):
        """Store button press if on top of the line."""
        if event.inaxes != self.line.axes or not self.line.contains(event)[0]:
            return
        x0 = self.line.get_xdata()
        y0 = self.line.get_ydata()
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        """Move the line on drag."""
        if self.press is None:
            return
        if event.inaxes != self.line.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        self.line.set_xdata(x0 + dx)
        self.drag_callback((x0 + dx)[0])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        """Handle release."""
        if event.inaxes != self.line.axes or self.press is None:
            return
        self.press = None
        self.line.figure.canvas.draw()
        self.modify_callback(self.x0, event.xdata)
        self.x0 = event.xdata

    def remove(self):
        """Remove the line."""
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)
        self.line.figure.axes[0].lines.remove(self.line)


def _set_ax_facecolor(ax, face_color):
    """Fix call for old MPL."""
    try:
        ax.set_facecolor(face_color)
    except AttributeError:
        ax.set_axis_bgcolor(face_color)


def _setup_ax_spines(axes, vlines, tmin, tmax, invert_y=False,
                     ymax_bound=None, unit=None, truncate_xaxis=True):
    ymin, ymax = axes.get_ylim()
    y_range = -np.subtract(ymin, ymax)

    # style the spines/axes
    axes.spines["top"].set_position('zero')
    if truncate_xaxis is True:
        axes.spines["top"].set_smart_bounds(True)
    else:
        axes.spines['top'].set_bounds(tmin, tmax)

    axes.tick_params(direction='out')
    axes.tick_params(right=False)

    current_ymin = axes.get_ylim()[0]

    # set x label
    axes.set_xlabel('Time (s)')
    axes.xaxis.get_label().set_verticalalignment('center')

    # set y label and ylabel position
    if unit is not None:
        axes.set_ylabel(unit + "\n", rotation=90)
        ylabel_height = (-(current_ymin / y_range)
                         if 0 > current_ymin  # ... if we have negative values
                         else (axes.get_yticks()[-1] / 2 / y_range))
        axes.yaxis.set_label_coords(-0.05, 1 - ylabel_height
                                    if invert_y else ylabel_height)

    xticks = sorted(list(set([x for x in axes.get_xticks()] + vlines)))
    axes.set_xticks(xticks)
    x_extrema = [t for t in xticks if tmax >= t >= tmin]
    if truncate_xaxis is True:
        axes.spines['bottom'].set_bounds(x_extrema[0], x_extrema[-1])
    else:
        axes.spines['bottom'].set_bounds(tmin, tmax)
    if ymin >= 0:
        axes.spines["top"].set_color('none')
    axes.spines["left"].set_zorder(0)

    # finishing touches
    if invert_y:
        axes.invert_yaxis()
    axes.spines['right'].set_color('none')
    axes.set_xlim(tmin, tmax)
    if truncate_xaxis is False:
        axes.axis("tight")
        axes.set_autoscale_on(False)


def _handle_decim(info, decim, lowpass):
    """Handle decim parameter for plotters."""
    from ..evoked import _check_decim
    from ..utils import _ensure_int
    if isinstance(decim, string_types) and decim == 'auto':
        lp = info['sfreq'] if info['lowpass'] is None else info['lowpass']
        lp = min(lp, info['sfreq'] if lowpass is None else lowpass)
        info['lowpass'] = lp
        decim = max(int(info['sfreq'] / (lp * 3) + 1e-6), 1)
    decim = _ensure_int(decim, 'decim', must_be='an int or "auto"')
    if decim <= 0:
        raise ValueError('decim must be "auto" or a positive integer, got %s'
                         % (decim,))
    decim = _check_decim(info, decim, 0)[0]
    data_picks = _pick_data_channels(info, exclude=())
    return decim, data_picks


def _grad_pair_pick_and_name(info, picks):
    """Deal with grads. (Helper for a few viz functions)."""
    from ..channels.layout import _pair_grad_sensors
    picked_chans = list()
    pairpicks = _pair_grad_sensors(info, topomap_coords=False)
    for ii in np.arange(0, len(pairpicks), 2):
        first, second = pairpicks[ii], pairpicks[ii + 1]
        if first in picks or second in picks:
            picked_chans.append(first)
            picked_chans.append(second)
    picks = list(sorted(set(picked_chans)))
    ch_names = [info["ch_names"][pick] for pick in picks]
    return picks, ch_names


def _setup_plot_projector(info, noise_cov, proj=True, use_noise_cov=True,
                          nave=1):
    from ..cov import compute_whitener
    projector = np.eye(len(info['ch_names']))
    whitened_ch_names = []
    if noise_cov is not None and use_noise_cov:
        # any channels in noise_cov['bads'] but not in info['bads'] get
        # set to nan, which means that they are not plotted.
        data_picks = _pick_data_channels(info, with_ref_meg=False, exclude=())
        data_names = set(info['ch_names'][pick] for pick in data_picks)
        # these can be toggled by the user
        bad_names = set(info['bads'])
        # these can't in standard pipelines be enabled (we always take the
        # union), so pretend they're not in cov at all
        cov_names = ((set(noise_cov['names']) & set(info['ch_names'])) -
                     set(noise_cov['bads']))
        # Actually compute the whitener only using the difference
        whiten_names = cov_names - bad_names
        whiten_picks = pick_channels(info['ch_names'], whiten_names)
        whiten_info = pick_info(info, whiten_picks)
        rank = _triage_rank_sss(whiten_info, [noise_cov])[1][0]
        whitener, whitened_ch_names = compute_whitener(
            noise_cov, whiten_info, rank=rank, verbose=False)
        whitener *= np.sqrt(nave)  # proper scaling for Evoked data
        assert set(whitened_ch_names) == whiten_names
        projector[whiten_picks, whiten_picks[:, np.newaxis]] = whitener
        # Now we need to change the set of "whitened" channels to include
        # all data channel names so that they are properly italicized.
        whitened_ch_names = data_names
        # We would need to set "bad_picks" to identity to show the traces
        # (but in gray), but here we don't need to because "projector"
        # starts out as identity. So all that is left to do is take any
        # *good* data channels that are not in the noise cov to be NaN
        nan_names = data_names - (bad_names | cov_names)
        # XXX conditional necessary because of annoying behavior of
        # pick_channels where an empty list means "all"!
        if len(nan_names) > 0:
            nan_picks = pick_channels(info['ch_names'], nan_names)
            projector[nan_picks] = np.nan
    elif proj:
        projector, _ = setup_proj(info, add_eeg_ref=False, verbose=False)
    return projector, whitened_ch_names


def _set_ax_label_style(ax, params, italicize=True):
    import matplotlib.text
    for tick in params['ax'].get_yaxis().get_major_ticks():
        for text in tick.get_children():
            if isinstance(text, matplotlib.text.Text):
                whitened = text.get_text() in params['whitened_ch_names']
                whitened = whitened and italicize
                text.set_style('italic' if whitened else 'normal')


def _check_sss(info):
    """Check SSS history in info."""
    ch_used = [ch for ch in _DATA_CH_TYPES_SPLIT
               if _contains_ch_type(info, ch)]
    has_meg = 'mag' in ch_used and 'grad' in ch_used
    has_sss = (has_meg and len(info['proc_history']) > 0 and
               info['proc_history'][0].get('max_info') is not None)
    return ch_used, has_meg, has_sss


def _triage_rank_sss(info, covs, rank=None, scalings=None):
    from ..cov import _estimate_rank_meeg_cov
    rank = dict() if rank is None else rank
    scalings = _handle_default('scalings_cov_rank', scalings)

    # Only look at good channels
    picks = _pick_data_channels(info, with_ref_meg=False, exclude='bads')
    info = pick_info(info, picks)
    ch_used, has_meg, has_sss = _check_sss(info)
    if has_sss:
        if 'mag' in rank or 'grad' in rank:
            raise ValueError('When using SSS, pass "meg" to set the rank '
                             '(separate rank values for "mag" or "grad" are '
                             'meaningless).')
    elif 'meg' in rank:
        raise ValueError('When not using SSS, pass separate rank values '
                         'for "mag" and "grad" (do not use "meg").')

    picks_list = _picks_by_type(info, meg_combined=has_sss)
    if has_sss:
        # reduce ch_used to combined mag grad
        ch_used = list(zip(*picks_list))[0]
    # order pick list by ch_used (required for compat with plot_evoked)
    picks_list = [x for x, y in sorted(zip(picks_list, ch_used))]
    n_ch_used = len(ch_used)

    # make sure we use the same rank estimates for GFP and whitening

    picks_list2 = [k for k in picks_list]
    # add meg picks if needed.
    if has_meg:
        # append ("meg", picks_meg)
        picks_list2 += _picks_by_type(info, meg_combined=True)

    rank_list = []  # rank dict for each cov
    for cov in covs:
        # We need to add the covariance projectors, compute the projector,
        # and apply it, just like we will do in prepare_noise_cov, otherwise
        # we risk the rank estimates being incorrect (i.e., if the projectors
        # do not match).
        info_proj = info.copy()
        info_proj['projs'] += cov['projs']
        this_rank = {}
        C = cov['data'].copy()
        # assemble rank dict for this cov, such that we have meg
        for ch_type, this_picks in picks_list2:
            # if we have already estimates / values for mag/grad but not
            # a value for meg, combine grad and mag.
            if ('mag' in this_rank and 'grad' in this_rank and
                    'meg' not in rank):
                this_rank['meg'] = this_rank['mag'] + this_rank['grad']
                # and we're done here
                break

            if rank.get(ch_type) is None:
                this_info = pick_info(info_proj, this_picks)
                idx = np.ix_(this_picks, this_picks)
                projector = setup_proj(this_info, add_eeg_ref=False)[0]
                this_C = C[idx]
                if projector is not None:
                    this_C = np.dot(np.dot(projector, this_C), projector.T)
                this_estimated_rank = _estimate_rank_meeg_cov(
                    this_C, this_info, scalings)
                _check_estimated_rank(
                    this_estimated_rank, this_picks, this_info, info,
                    cov, ch_type, has_meg, has_sss)
                this_rank[ch_type] = this_estimated_rank
            elif rank.get(ch_type) is not None:
                this_rank[ch_type] = rank[ch_type]

        rank_list.append(this_rank)
    return n_ch_used, rank_list, picks_list, has_sss


def _check_estimated_rank(this_estimated_rank, this_picks, this_info, info,
                          cov, ch_type, has_meg, has_sss):
    """Compare estimated against expected rank."""
    expected_rank = len(this_picks)
    expected_rank_reduction = 0
    if has_meg and has_sss and ch_type == 'meg':
        sss_rank = _get_rank_sss(info)
        expected_rank_reduction += (expected_rank - sss_rank)
    n_ssp = sum(_match_proj_type(pp, this_info['ch_names'])
                for pp in cov['projs'])
    expected_rank_reduction += n_ssp
    expected_rank -= expected_rank_reduction
    if this_estimated_rank != expected_rank:
        logger.debug(
            'For (%s) the expected and estimated rank diverge '
            '(%i VS %i). \nThis may lead to surprising reults. '
            '\nPlease consider using the `rank` parameter to '
            'manually specify the spatial degrees of freedom.' % (
                ch_type, expected_rank, this_estimated_rank
            ))


def _match_proj_type(proj, ch_names):
    """See if proj should be counted."""
    proj_ch_names = proj['data']['col_names']
    select = any(kk in ch_names for kk in proj_ch_names)
    return select


def _check_cov(noise_cov, info):
    """Check the noise_cov for whitening and issue an SSS warning."""
    from ..cov import read_cov, Covariance
    if noise_cov is None:
        return None
    if isinstance(noise_cov, string_types):
        noise_cov = read_cov(noise_cov)
    if not isinstance(noise_cov, Covariance):
        raise TypeError('noise_cov must be a str or Covariance, got %s'
                        % (type(noise_cov),))
    if _check_sss(info)[2]:  # has_sss
        warn('Data have been processed with SSS, which changes the relative '
             'scaling of magnetometers and gradiometers when viewing data '
             'whitened by a noise covariance')
    return noise_cov


def _set_title_multiple_electrodes(title, combine, ch_names, max_chans=6,
                                   all=False, ch_type=None):
    """Prepare a title string for multiple electrodes."""
    if title is None:
        title = ", ".join(ch_names[:max_chans])
        ch_type = _channel_type_prettyprint.get(ch_type, ch_type)
        if ch_type is None:
            ch_type = "sensor"
        if len(ch_names) > 1:
            ch_type += "s"
        if all is True and isinstance(combine, string_types):
            combine = combine[0].upper() + combine[1:]
            title = "{} of {} {}".format(
                combine, len(ch_names), ch_type)
        elif len(ch_names) > max_chans and combine is not "gfp":
            warn("More than {} channels, truncating title ...".format(
                max_chans))
            title += ", ...\n({} of {} {})".format(
                combine, len(ch_names), ch_type,)
    return title


def _check_time_unit(time_unit, times, allow_none=False):
    if not (isinstance(time_unit, string_types) or
            (time_unit is None and allow_none)):
        raise TypeError('time_unit must be str, got %s' % (type(time_unit),))
    if time_unit is None:
        warn('time_unit defaults to "ms" in 0.16 but will change to "s" in '
             '0.17, set it explicitly to avoid this warning',
             DeprecationWarning)
        time_unit = 'ms'
    if time_unit == 's':
        times = times
    elif time_unit == 'ms':
        times = 1e3 * times
    else:
        raise ValueError("time_unit must be 's' or 'ms', got %r" % time_unit)
    return time_unit, times
