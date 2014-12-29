"""Functions to make simple plot on evoked M/EEG data (besides topographies)
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

from copy import deepcopy
from itertools import cycle

import numpy as np

from ..io.pick import channel_type, pick_types, _picks_by_type
from ..externals.six import string_types
from .utils import _mutable_defaults, _check_delayed_ssp
from .utils import _draw_proj_checkbox, tight_layout
from ..utils import logger
from ..io.pick import pick_info


def _plot_evoked(evoked, picks, exclude, unit, show,
                 ylim, proj, xlim, hline, units,
                 scalings, titles, axes, plot_type,
                 cmap=None):
    """Aux function for plot_evoked and plot_evoked_image (cf. docstrings)

    Extra param is:

    plot_type : str, value ('butterfly' | 'image')
        The type of graph to plot: 'butterfly' plots each channel as a line
        (x axis: time, y axis: amplitude). 'image' plots a 2D image where
        color depicts the amplitude of each channel at a given time point
        (x axis: time, y axis: channel). In 'image' mode, the plot is not
        interactive.
    """
    import matplotlib.pyplot as plt
    if axes is not None and proj == 'interactive':
        raise RuntimeError('Currently only single axis figures are supported'
                           ' for interactive SSP selection.')

    scalings, titles, units = _mutable_defaults(('scalings', scalings),
                                                ('titles', titles),
                                                ('units', units))

    channel_types = set(key for d in [scalings, titles, units] for key in d)
    channel_types = sorted(channel_types)  # to guarantee consistent order

    if picks is None:
        picks = list(range(evoked.info['nchan']))

    bad_ch_idx = [evoked.ch_names.index(ch) for ch in evoked.info['bads']
                  if ch in evoked.ch_names]
    if len(exclude) > 0:
        if isinstance(exclude, string_types) and exclude == 'bads':
            exclude = bad_ch_idx
        elif (isinstance(exclude, list)
              and all([isinstance(ch, string_types) for ch in exclude])):
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
            # Parameters for butterfly interactive plots
            if plot_type == 'butterfly':
                if any([i in bad_ch_idx for i in idx]):
                    colors = ['k'] * len(idx)
                    for i in bad_ch_idx:
                        if i in idx:
                            colors[idx.index(i)] = 'r'

                    ax._get_lines.color_cycle = iter(colors)
                else:
                    ax._get_lines.color_cycle = cycle(['k'])
            # Set amplitude scaling
            D = this_scaling * evoked.data[idx, :]
            # plt.axes(ax)
            if plot_type == 'butterfly':
                ax.plot(times, D.T)
            elif plot_type == 'image':
                im = ax.imshow(D, interpolation='nearest', origin='lower',
                               extent=[times[0], times[-1], 0, D.shape[0]],
                               aspect='auto', cmap=cmap)
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.set_title(ch_unit)
            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                ax.set_xlim(xlim)
            if ylim is not None and t in ylim:
                if plot_type == 'butterfly':
                    ax.set_ylim(ylim[t])
                elif plot_type == 'image':
                    im.set_clim(ylim[t])
            ax.set_title(titles[t] + ' (%d channel%s)' % (
                         len(D), 's' if len(D) > 1 else ''))
            ax.set_xlabel('time (ms)')
            if plot_type == 'butterfly':
                ax.set_ylabel('data (%s)' % ch_unit)
            elif plot_type == 'image':
                ax.set_ylabel('channels (%s)' % 'index')
            else:
                raise ValueError("plot_type has to be 'butterfly' or 'image'."
                                 "Got %s." % plot_type)

            if (plot_type == 'butterfly') and (hline is not None):
                for h in hline:
                    ax.axhline(h, color='r', linestyle='--', linewidth=2)

    if axes_init is None:
        plt.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
                      axes=axes, types=types, units=units, scalings=scalings,
                      unit=unit, ch_types_used=ch_types_used, picks=picks,
                      plot_update_proj_callback=_plot_update_evoked,
                      plot_type=plot_type)
        _draw_proj_checkbox(None, params)

    if show and plt.get_backend() != 'agg':
        plt.show()
        fig.canvas.draw()  # for axes plots update axes.
    tight_layout(fig=fig)

    return fig


def plot_evoked(evoked, picks=None, exclude='bads', unit=True, show=True,
                ylim=None, proj=False, xlim='tight', hline=None, units=None,
                scalings=None, titles=None, axes=None, plot_type="butterfly"):
    """Plot evoked data

    Note: If bad channels are not excluded they are shown in red.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    picks : array-like of int | None
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
    axes : instance of Axis | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
    """
    return _plot_evoked(evoked=evoked, picks=picks, exclude=exclude, unit=unit,
                        show=show, ylim=ylim, proj=proj, xlim=xlim,
                        hline=hline, units=units, scalings=scalings,
                        titles=titles, axes=axes, plot_type="butterfly")


def plot_evoked_image(evoked, picks=None, exclude='bads', unit=True, show=True,
                      clim=None, proj=False, xlim='tight', units=None,
                      scalings=None, titles=None, axes=None, cmap='RdBu_r'):
    """Plot evoked data as images

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    picks : array-like of int | None
        The indices of channels to plot. If None show all.
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded.
    unit : bool
        Scale plot with channel (SI) unit.
    show : bool
        Call pyplot.show() as the end or not.
    clim : dict | None
        clim for plots. e.g. clim = dict(eeg=[-200e-6, 200e6])
        Valid keys are eeg, mag, grad, misc. If None, the clim parameter
        for each channel equals the pyplot default.
    xlim : 'tight' | tuple | None
        xlim for plots.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    titles : dict | None
        The titles associated with the channels. If None, defaults to
        `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
    axes : instance of Axis | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
    cmap : matplotlib colormap
        Colormap.
    """
    return _plot_evoked(evoked=evoked, picks=picks, exclude=exclude, unit=unit,
                        show=show, ylim=clim, proj=proj, xlim=xlim,
                        hline=None, units=units, scalings=scalings,
                        titles=titles, axes=axes, plot_type="image",
                        cmap=cmap)


def _plot_update_evoked(params, bools):
    """ update the plot evoked lines
    """
    picks, evoked = [params[k] for k in ('picks', 'evoked')]
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
        if params['plot_type'] == 'butterfly':
            [line.set_data(times, di) for line, di in zip(ax.lines, D)]
        else:
            ax.images[0].set_data(D)
    params['fig'].canvas.draw()


def plot_evoked_white(evoked, noise_cov, scalings=None, rank=None, show=True):
    """Plot whitened evoked response

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked response.
    rank : dict of int | None
        Dict of ints where keys are 'eeg', 'mag' or 'grad'. If None,
        the rank is detected automatically. Defaults to None.
    noise_cov : list or tuple or single instance of mne.cov.Covariance
        The noise covs.
    show : bool
        Whether to show the figure or not. Defaults to True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object containing the plot.
    """
    from ..cov import whiten_evoked  # recursive import
    from ..cov import _estimate_rank_meeg_cov
    import matplotlib.pyplot as plt
    if scalings is None:
        scalings = dict(mag=1e12, grad=1e11, eeg=1e5)

    ch_used = [ch for ch in ['eeg', 'grad', 'mag'] if ch in evoked]
    has_meg = 'mag' in ch_used and 'grad' in ch_used

    if not isinstance(noise_cov, (list, tuple)):
        noise_cov = [noise_cov]

    proc_history = evoked.info.get('proc_history', [])
    has_sss = False
    if len(proc_history) > 0:
        # if SSSed, mags and grad are not longer independent
        # for correct display of the whitening we will drop the cross-terms
        # (the gradiometer * magnetometer covariance)
        has_sss = 'max_info' in proc_history[0] and has_meg
    if has_sss:
        logger.info('SSS has been applied to data. Showing mag and grad '
                    'whitening jointly.')

    evoked = evoked.copy()  # handle ref meg
    picks = pick_types(evoked.info, meg=True, eeg=True, ref_meg=False,
                       exclude='bads')
    evoked.pick_channels([evoked.ch_names[k] for k in picks], copy=False)
    picks_list = _picks_by_type(evoked.info, meg_combined=has_sss)
    if has_meg and has_sss:
        ch_used = [c for c, _ in picks_list]
    n_ch_used = len(ch_used)

    # make sure we use the same rank estimates for GFP and whitening
    rank_list = []
    for cov in noise_cov:
        rank_ = {}
        C = cov['data'].copy()
        picks_list2 = [k for k in picks_list]
        if rank is None:
            if has_meg and not has_sss:
                picks_list2 += _picks_by_type(evoked.info,
                                              meg_combined=True)
            for ch_type, this_picks in picks_list2:
                this_info = pick_info(evoked.info, this_picks)
                idx = np.ix_(this_picks, this_picks)
                this_rank = _estimate_rank_meeg_cov(C[idx], this_info, scalings)
                rank_[ch_type] = this_rank
        if rank is not None:
            rank_.update(rank)
        rank_list.append(rank_)

    evokeds_white = [whiten_evoked(evoked, n, picks, rank=r, scalings=scalings)
                     for n, r in zip(noise_cov, rank_list)]

    axes_evoked = None

    def whitened_gfp(x, rank=None):
        """Whitened Global Field Power

        The MNE inverse solver assumes zero mean whitened data as input.
        Therefore, a chi^2 statistic will be best to detect model violations.
        """
        return np.sum(x ** 2, axis=0) / (len(x) if rank is None else rank)

    # prepare plot
    if len(noise_cov) > 1:
        n_columns = 2
        n_extra_row = 0
    else:
        n_columns = 1
        n_extra_row = 1

    n_rows = n_ch_used + n_extra_row
    fig, axes = plt.subplots(n_rows,
                             n_columns, sharex=True, sharey=False,
                             figsize=(8.8, 2.2 * n_rows))
    if n_columns > 1:
        suptitle = ('Whitened evoked (left, best estimator = "%s")\n'
                    'and global field power '
                    '(right, comparison of estimators)' %
                    noise_cov[0]['method'])
        fig.suptitle(suptitle)

    ax_gfp = None
    if any(((n_columns == 1 and n_ch_used == 1),
            (n_columns == 1 and n_ch_used > 1),
            (n_columns == 2 and n_ch_used == 1))):
        axes_evoked = axes[:n_ch_used]
        ax_gfp = axes[-1:]
    elif n_columns == 2 and n_ch_used > 1:
        axes_evoked = axes[:n_ch_used, 0]
        ax_gfp = axes[:, 1]
    else:
        raise RuntimeError('Wrong axes inputs')

    times = evoked.times * 1e3
    titles_ = _mutable_defaults(('titles', None))[0]
    if has_sss:
        titles_['meg'] = 'MEG (combined)'

    colors = [plt.cm.Set1(i) for i in np.linspace(0, 0.5, len(noise_cov))]
    ch_colors = {'eeg': 'black', 'mag': 'blue', 'grad': 'cyan',
                 'meg': 'steelblue'}
    iter_gfp = zip(evokeds_white, noise_cov, rank_list, colors)

    if not has_sss:
        evokeds_white[0].plot(unit=False, axes=axes_evoked, hline=[-1.96, 1.96])
    else:
        for ((ch_type, picks), ax) in zip(picks_list, axes_evoked):
            ax.plot(times, evokeds_white[0].data[picks].T, color='k')
            for hline in [-1.96, 1.96]:
                ax.axhline(hline, color='red', linestyle='--')

    # Now plot the GFP
    for evoked_white, noise_cov, rank_, color in iter_gfp:
        i = 0
        for ch, sub_picks in picks_list:
            this_rank = rank_[ch]
            title = '{0} ({2}{1})'.format(
                    titles_[ch] if n_columns > 1 else ch,
                    this_rank, 'rank ' if n_columns > 1 else '')
            label = noise_cov['method']

            ax_gfp[i].set_title(title if n_columns > 1 else
                                'whitened global field power (GFP),'
                                ' method = "%s"' % label)

            data = evoked_white.data[sub_picks]
            gfp = whitened_gfp(data, rank=this_rank)
            ax_gfp[i].plot(times, gfp,
                           label=(label if n_columns > 1 else title),
                           color=color if n_columns > 1 else ch_colors[ch])
            ax_gfp[i].set_xlabel('times [ms]')
            ax_gfp[i].set_ylabel('GFP [chi^2]')
            ax_gfp[i].set_xlim(times[0], times[-1])
            ax_gfp[i].set_ylim(0, 10)
            ax_gfp[i].axhline(1, color='red', linestyle='--')
            if n_columns > 1:
                i += 1

    ax = ax_gfp[0]
    if n_columns == 1:
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.9), fontsize=12)
    else:
        ax.legend(loc='upper right', fontsize=10)
        params = dict(top=[0.69, 0.82, 0.87][n_rows - 1],
                      bottom=[0.22, 0.13, 0.09][n_rows - 1])
        if has_sss:
            params['hspace'] = 0.49
        fig.subplots_adjust(**params)
    fig.canvas.draw()

    if show is True:
        fig.show()
    return fig
