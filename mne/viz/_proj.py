"""Functions for plotting projectors."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy

import numpy as np

from .evoked import _plot_evoked
from .topomap import _plot_projs_topomap
from .utils import plt_show, _check_type_projs
from ..defaults import DEFAULTS
from ..io.pick import _picks_to_idx
from ..utils import _validate_type, warn, _pl, verbose


@verbose
def plot_projs_joint(projs, evoked, picks_trace=None, *, topomap_kwargs=None,
                     show=True, verbose=None):
    """Plot projectors and evoked jointly.

    Parameters
    ----------
    projs : list of Projection
        The projectors to plot.
    evoked : instance of Evoked
        The data to plot. Typically this is the evoked instance created from
        averaging the epochs used to create the projection.
    %(picks_plot_projs_joint_trace)s
    topomap_kwargs : dict | None
        Keyword arguments to pass to :func:`mne.viz.plot_projs_topomap`.
    %(show)s
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib Figure
        The figure.

    Notes
    -----
    This function creates a figure with three columns:

    1. The left shows the evoked data traces before (black) and after (green)
       projection.
    2. The center shows the topomaps associated with each of the projectors.
    3. The right again shows the data traces (black), but this time with:

       1. The data projected onto each projector with a single normalization
          factor (solid lines). This is useful for seeing the relative power
          in each projection vector.
       2. The data projected onto each projector with individual normalization
          factors (dashed lines). This is useful for visualizing each time
          course regardless of its power.
       3. Additional data traces from ``picks_trace`` (solid yellow lines).
          This is useful for visualizing the "ground truth" of the time
          course, e.g. the measured EOG or ECG channel time courses.

    .. versionadded:: 1.1
    """
    import matplotlib.pyplot as plt
    from ..evoked import Evoked
    _validate_type(evoked, Evoked, 'evoked')
    _validate_type(topomap_kwargs, (None, dict), 'topomap_kwargs')
    projs = _check_type_projs(projs)
    topomap_kwargs = dict() if topomap_kwargs is None else topomap_kwargs
    if picks_trace is not None:
        picks_trace = _picks_to_idx(
            evoked.info, picks_trace, allow_empty=False)
    info = evoked.info
    ch_types = evoked.get_channel_types(unique=True, only_data_chs=True)
    proj_by_type = dict()  # will be set up like an enumerate key->[pi, proj]
    ch_names_by_type = dict()
    used = np.zeros(len(projs), int)
    for ch_type in ch_types:
        these_picks = _picks_to_idx(info, ch_type, allow_empty=True)
        these_chs = [evoked.ch_names[pick] for pick in these_picks]
        ch_names_by_type[ch_type] = these_chs
        for pi, proj in enumerate(projs):
            if not set(these_chs).intersection(proj['data']['col_names']):
                continue
            if ch_type not in proj_by_type:
                proj_by_type[ch_type] = list()
            proj_by_type[ch_type].append([pi, deepcopy(proj)])
            used[pi] += 1
    missing = (~used.astype(bool)).sum()
    if missing:
        warn(f'{missing} projector{_pl(missing)} had no channel names '
             'present in epochs')
    del projs
    ch_types = list(proj_by_type)  # reduce to number we actually need
    # room for legend
    max_proj_per_type = max(len(x) for x in proj_by_type.values())
    cs_trace = 3
    cs_topo = 2
    n_col = max_proj_per_type * cs_topo + 2 * cs_trace
    n_row = len(ch_types)
    shape = (n_row, n_col)
    fig = plt.figure(figsize=(n_col * 1.1 + 0.5, n_row * 1.8 + 0.5),
                     constrained_layout=True)
    ri = 0
    # pick some sufficiently distinct colors (6 per proj type, e.g., ECG,
    # should be enough hopefully!)
    # https://personal.sron.nl/~pault/data/colourschemes.pdf
    # "Vibrant" color scheme
    proj_colors = [
        '#CC3311',  # red
        '#009988',  # teal
        '#0077BB',  # blue
        '#EE3377',  # magenta
        '#EE7733',  # orange
        '#33BBEE',  # cyan
    ]
    trace_color = '#CCBB44'  # yellow
    after_color, after_name = '#228833', 'green'
    type_titles = DEFAULTS['titles']
    last_ax = [None] * 2
    first_ax = dict()
    pe_kwargs = dict(show=False, draw=False)
    for ch_type, these_projs in proj_by_type.items():
        these_idxs, these_projs = zip(*these_projs)
        ch_names = ch_names_by_type[ch_type]
        idx = np.where([np.in1d(ch_names, proj['data']['col_names']).all()
                        for proj in these_projs])[0]
        used[idx] += 1
        count = len(these_projs)
        for proj in these_projs:
            sub_idx = [proj['data']['col_names'].index(name)
                       for name in ch_names]
            proj['data']['data'] = proj['data']['data'][:, sub_idx]
            proj['data']['col_names'] = ch_names
        ba_ax = plt.subplot2grid(
            shape, (ri, 0), colspan=cs_trace, fig=fig)
        topo_axes = [
            plt.subplot2grid(
                shape, (ri, ci * cs_topo + cs_trace), colspan=cs_topo, fig=fig)
            for ci in range(count)]
        tr_ax = plt.subplot2grid(
            shape, (ri, n_col - cs_trace), colspan=cs_trace, fig=fig)
        # topomaps
        _plot_projs_topomap(these_projs, info=info,
                            axes=topo_axes, **topomap_kwargs)
        for idx, proj, ax_ in zip(these_idxs, these_projs, topo_axes):
            ax_.set_title('')  # could use proj['desc'] but it's long
            ax_.set_xlabel(f'projs[{idx}]', fontsize='small')
        unit = DEFAULTS['units'][ch_type]
        # traces
        this_evoked = evoked.copy().pick_channels(ch_names)
        p = np.concatenate([p['data']['data'] for p in these_projs])
        assert p.shape == (len(these_projs), len(this_evoked.data))
        traces = np.dot(p, this_evoked.data)
        traces *= np.sign(np.mean(
            np.dot(this_evoked.data, traces.T), 0))[:, np.newaxis]
        if picks_trace is not None:
            ch_traces = evoked.data[picks_trace]
            ch_traces -= np.mean(ch_traces, axis=1, keepdims=True)
            ch_traces /= np.abs(ch_traces).max()
        _plot_evoked(this_evoked, picks='all', axes=[tr_ax], **pe_kwargs,
                     spatial_colors=False)
        for line in tr_ax.lines:
            line.set(lw=0.5, zorder=3)
        for t in list(tr_ax.texts):
            t.remove()
        scale = 0.8 * np.abs(tr_ax.get_ylim()).max()
        hs, labels = list(), list()
        traces /= np.abs(traces).max()  # uniformly scaled
        for ti, trace in enumerate(traces):
            hs.append(tr_ax.plot(
                this_evoked.times, trace * scale,
                color=proj_colors[ti % len(proj_colors)], zorder=5)[0])
            labels.append(f'projs[{these_idxs[ti]}]')
        traces /= np.abs(traces).max(1, keepdims=True)  # independently
        for ti, trace in enumerate(traces):
            tr_ax.plot(
                this_evoked.times, trace * scale,
                color=proj_colors[ti % len(proj_colors)], zorder=3.5,
                ls='--', lw=1., alpha=0.75)
        if picks_trace is not None:
            trace_ch = [evoked.ch_names[pick] for pick in picks_trace]
            if len(picks_trace) == 1:
                trace_ch = trace_ch[0]
            hs.append(tr_ax.plot(
                this_evoked.times, ch_traces.T * scale, color=trace_color,
                lw=3, zorder=4, alpha=0.75)[0])
            labels.append(str(trace_ch))
        tr_ax.set(title='', xlabel='', ylabel='')
        # This will steal space from the subplots in a constrained layout
        # https://matplotlib.org/3.5.0/tutorials/intermediate/constrainedlayout_guide.html#legends  # noqa: E501
        tr_ax.legend(
            hs, labels, loc='center left', borderaxespad=0.05,
            bbox_to_anchor=[1.05, 0.5])
        last_ax[1] = tr_ax
        key = 'Projected time course'
        if key not in first_ax:
            first_ax[key] = tr_ax
        # Before and after traces
        _plot_evoked(this_evoked, picks='all', axes=[ba_ax], **pe_kwargs)
        for line in ba_ax.lines:
            line.set(lw=0.5, zorder=3)
        loff = len(ba_ax.lines)
        this_proj_evoked = this_evoked.copy().add_proj(these_projs)
        # with meg='combined' any existing mag projectors (those already part
        # of evoked before we add_proj above) will have greatly
        # reduced power, so we ignore the warning about this issue
        this_proj_evoked.apply_proj(verbose='error')
        _plot_evoked(this_proj_evoked, picks='all', axes=[ba_ax], **pe_kwargs)
        for line in ba_ax.lines[loff:]:
            line.set(lw=0.5, zorder=4, color=after_color)
        for t in list(ba_ax.texts):
            t.remove()
        ba_ax.set(title='', xlabel='')
        ba_ax.set(ylabel=f'{type_titles[ch_type]}\n{unit}')
        last_ax[0] = ba_ax
        key = f'Before (black) and after ({after_name})'
        if key not in first_ax:
            first_ax[key] = ba_ax
        ri += 1
    for ax in last_ax:
        ax.set(xlabel='Time (sec)')
    for title, ax in first_ax.items():
        ax.set_title(title, fontsize='medium')
    plt_show(show)
    return fig
