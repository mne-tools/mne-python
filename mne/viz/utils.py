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
from copy import deepcopy
from functools import partial
import difflib
import webbrowser
from warnings import warn
import tempfile

import numpy as np

from ..io import show_fiff
from ..utils import verbose


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
                            mag='Magnetometers', misc='misc'),
                mask_params=dict(marker='o',
                                 markerfacecolor='w',
                                 markeredgecolor='k',
                                 linewidth=0,
                                 markeredgewidth=1,
                                 markersize=4))


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


def _setup_vmin_vmax(data, vmin, vmax):
    """Aux function to handle vmin and vamx parameters"""
    if vmax is None and vmin is None:
        vmax = np.abs(data).max()
        vmin = -vmax
    else:
        if callable(vmin):
            vmin = vmin(data)
        elif vmin is None:
            vmin = np.min(data)
        if callable(vmax):
            vmax = vmax(data)
        elif vmin is None:
            vmax = np.max(data)
    return vmin, vmax


def tight_layout(pad=1.2, h_pad=None, w_pad=None, fig=None):
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
    if fig is None:
        fig = plt.gcf()

    try:  # see https://github.com/matplotlib/matplotlib/issues/2654
        fig.canvas.draw()
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    except:
        msg = ('Matplotlib function \'tight_layout\'%s.'
               ' Skipping subpplot adjusment.')
        if not hasattr(plt, 'tight_layout'):
            case = ' is not available'
        else:
            case = (' is not supported by your backend: `%s`'
                    % plt.get_backend())
        warn(msg % case)


def _check_delayed_ssp(container):
    """ Aux function to be used for interactive SSP selection
    """
    if container.proj is True or\
       all([p['active'] for p in container.info['projs']]):
        raise RuntimeError('Projs are already applied. Please initialize'
                           ' the data with proj set to False.')
    elif len(container.info['projs']) < 1:
        raise RuntimeError('No projs found in evoked.')


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
        ax.set_visible(False)
    return fig, axes


def _draw_proj_checkbox(event, params, draw_current_state=True):
    """Toggle options (projectors) dialog"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    projs = params['projs']
    # turn on options dialog

    labels = [p['desc'] for p in projs]
    actives = ([p['active'] for p in projs] if draw_current_state else
               [True] * len(params['projs']))

    width = max([len(p['desc']) for p in projs]) / 6.0 + 0.5
    height = len(projs) / 6.0 + 0.5
    fig_proj = figure_nobar(figsize=(width, height))
    fig_proj.canvas.set_window_title('SSP projection vectors')
    ax_temp = plt.axes((0, 0, 1, 1))
    ax_temp.get_yaxis().set_visible(False)
    ax_temp.get_xaxis().set_visible(False)
    fig_proj.add_axes(ax_temp)

    proj_checks = mpl.widgets.CheckButtons(ax_temp, labels=labels,
                                           actives=actives)
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
        fig_proj.show()
    except Exception:
        pass


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


def figure_nobar(*args, **kwargs):
    """Make matplotlib figure with no toolbar"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    old_val = mpl.rcParams['toolbar']
    try:
        mpl.rcParams['toolbar'] = 'none'
        fig = plt.figure(*args, **kwargs)
        # remove button press catchers (for toolbar)
        cbs = list(fig.canvas.callbacks.callbacks['key_press_event'].keys())
        for key in cbs:
            fig.canvas.callbacks.disconnect(key)
    except Exception as ex:
        raise ex
    finally:
        mpl.rcParams['toolbar'] = old_val
    return fig
