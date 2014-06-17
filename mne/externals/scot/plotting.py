# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Graphical output with matplotlib

This module attempts to import matplotlib for plotting functionality.
If matplotlib is not available no error is raised, but plotting functions will not be available.

"""

import numpy as np

try:
    #noinspection PyPep8Naming
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.cm as cm

    _have_pyplot = True
except ImportError:
    plt, MaxNLocator = None, None
    _have_pyplot = False


def show_plots():
    plt.show()


def prepare_topoplots(topo, values):
    """ Prepare multiple topo maps for cached plotting.

    .. note:: Parameter `topo` is modified by the function by calling :func:`~eegtopo.topoplot.Topoplot.set_values`.

    Parameters
    ----------
    topo : :class:`~eegtopo.topoplot.Topoplot`
        Scalp maps are created with this class
    values : array, shape = [n_topos, n_channels]
        Channel values for each topo plot

    Returns
    -------
    topomaps : list of array
        The map for each topo plot
    """
    values = np.atleast_2d(values)

    topomaps = []

    for i in range(values.shape[0]):
        topo.set_values(values[i, :])
        topo.create_map()
        topomaps.append(topo.get_map())

    return topomaps


def plot_topo(axis, topo, topomap, crange=None, offset=(0,0)):
    """ Draw a topoplot in given axis.

    .. note:: Parameter `topo` is modified by the function by calling :func:`~eegtopo.topoplot.Topoplot.set_map`.

    Parameters
    ----------
    axis : axis
        Axis to draw into.
    topo : :class:`~eegtopo.topoplot.Topoplot`
        This object draws the topo plot
    topomap : array, shape = [w_pixels, h_pixels]
        Scalp-projected data
    crange : [int, int], optional
        Range of values covered by the colormap.
        If set to None, [-max(abs(topomap)), max(abs(topomap))] is substituted.
    offset : [float, float], optional
        Shift the topo plot by [x,y] in axis units.

    Returns
    -------
    h : image
        Image object the map was plotted into
    """
    topo.set_map(topomap)
    h = topo.plot_map(axis, crange=crange, offset=offset)
    topo.plot_locations(axis, offset=offset)
    topo.plot_head(axis, offset=offset)
    return h


def plot_sources(topo, mixmaps, unmixmaps, global_scale=None, fig=None):
    """ Plot all scalp projections of mixing- and unmixing-maps.

    .. note:: Parameter `topo` is modified by the function by calling :func:`~eegtopo.topoplot.Topoplot.set_map`.

    Parameters
    ----------
    topo : :class:`~eegtopo.topoplot.Topoplot`
        This object draws the topo plot
    mixmaps : array, shape = [w_pixels, h_pixels]
        Scalp-projected mixing matrix
    unmixmaps : array, shape = [w_pixels, h_pixels]
        Scalp-projected unmixing matrix
    global_scale : float, optional
        Set common color scale as given percentile of all map values to use as the maximum.
        `None` scales each plot individually (default).
    fig : Figure object, optional
        Figure to plot into. If set to `None`, a new figure is created.

    Returns
    -------
    fig : Figure object
        The figure into which was plotted.
    """
    if not _have_pyplot:
        raise ImportError("matplotlib.pyplot is required for plotting")

    urange, mrange = None, None

    m = len(mixmaps)

    if global_scale:
        tmp = np.asarray(unmixmaps)
        tmp = tmp[np.logical_not(np.isnan(tmp))]
        umax = np.percentile(np.abs(tmp), global_scale)
        umin = -umax
        urange = [umin, umax]

        tmp = np.asarray(mixmaps)
        tmp = tmp[np.logical_not(np.isnan(tmp))]
        mmax = np.percentile(np.abs(tmp), global_scale)
        mmin = -mmax
        mrange = [mmin, mmax]

    y = np.floor(np.sqrt(m * 3 / 4))
    x = np.ceil(m / y)

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(m):
        axes.append(fig.add_subplot(2 * y, x, i + 1))
        plot_topo(axes[-1], topo, unmixmaps[i], crange=urange)
        axes[-1].set_title(str(i))

        axes.append(fig.add_subplot(2 * y, x, m + i + 1))
        plot_topo(axes[-1], topo, mixmaps[i], crange=mrange)
        axes[-1].set_title(str(i))

    for a in axes:
        a.set_yticks([])
        a.set_xticks([])
        a.set_frame_on(False)

    axes[0].set_ylabel('Unmixing weights')
    axes[1].set_ylabel('Scalp projections')

    return fig


def plot_connectivity_topos(layout='diagonal', topo=None, topomaps=None, fig=None):
    """ Place topo plots in a figure suitable for connectivity visualization.

    .. note:: Parameter `topo` is modified by the function by calling :func:`~eegtopo.topoplot.Topoplot.set_map`.

    Parameters
    ----------
    layout : str
        'diagonal' -> place topo plots on diagonal.
        otherwise -> place topo plots in left column and top row.
    topo : :class:`~eegtopo.topoplot.Topoplot`
        This object draws the topo plot
    topomaps : array, shape = [w_pixels, h_pixels]
        Scalp-projected map
    fig : Figure object, optional
        Figure to plot into. If set to `None`, a new figure is created.

    Returns
    -------
    fig : Figure object
        The figure into which was plotted.
    """

    m = len(topomaps)

    if fig is None:
        fig = plt.figure()

    if layout == 'diagonal':
        for i in range(m):
            ax = fig.add_subplot(m, m, i*(1+m) + 1)
            plot_topo(ax, topo, topomaps[i])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_frame_on(False)
    else:
        for i in range(m):
            for j in [i+2, (i+1)*(m+1)+1]:
                ax = fig.add_subplot(m+1, m+1, j)
                plot_topo(ax, topo, topomaps[i])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_frame_on(False)

    return fig


def plot_connectivity_spectrum(a, fs=2, freq_range=(-np.inf, np.inf), diagonal=0, border=False, fig=None):
    """ Draw connectivity plots.

    Parameters
    ----------
    a : array, shape = [n_channels, n_channels, n_fft] or [1 or 3, n_channels, n_channels, n_fft]
        If a.ndim == 3, normal plots are created,
        If a.ndim == 4 and a.shape[0] == 1, the area between the curve and y=0 is filled transparently,
        If a.ndim == 4 and a.shape[0] == 3, a[0,:,:,:] is plotted normally and the area between a[1,:,:,:] and
        a[2,:,:,:] is filled transparently.
    fs : float
        Sampling frequency
    freq_range : (float, float)
        Frequency range to plot
    diagonal : {-1, 0, 1}
        If diagonal == -1 nothing is plotted on the diagonal (a[i,i,:] are not plotted),
        if diagonal == 0, a is plotted on the diagonal too (all a[i,i,:] are plotted),
        if diagonal == 1, a is plotted on the diagonal only (only a[i,i,:] are plotted)
    border : bool
        If border == true the leftmost column and the topmost row are left blank
    fig : Figure object, optional
        Figure to plot into. If set to `None`, a new figure is created.

    Returns
    -------
    fig : Figure object
        The figure into which was plotted.
    """

    a = np.atleast_3d(a)
    if a.ndim == 3:
        [_, m, f] = a.shape
        l = 0
    else:
        [l, _, m, f] = a.shape
    freq = np.linspace(0, fs / 2, f)

    lowest, highest = np.inf, -np.inf
    left = max(freq_range[0], freq[0])
    right = min(freq_range[1], freq[-1])

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(m):
        if diagonal == 1:
            jrange = [i]
        elif diagonal == 0:
            jrange = range(m)
        else:
            jrange = [j for j in range(m) if j != i]
        for j in jrange:
            if border:
                ax = fig.add_subplot(m+1, m+1, j + (i+1) * (m+1) + 2)
            else:
                ax = fig.add_subplot(m, m, j + i * m + 1)
            axes.append((i, j, ax))
            if l == 0:
                ax.plot(freq, a[i, j, :])
                lowest = min(lowest, np.min(a[i, j, :]))
                highest = max(highest, np.max(a[i, j, :]))
            elif l == 1:
                ax.fill_between(freq, 0, a[0, i, j, :], facecolor=[0.25, 0.25, 0.25], alpha=0.25)
                lowest = min(lowest, np.min(a[0, i, j, :]))
                highest = max(highest, np.max(a[0, i, j, :]))
            else:
                baseline,  = ax.plot(freq, a[0, i, j, :])
                ax.fill_between(freq, a[1, i, j, :], a[2, i, j, :], facecolor=baseline.get_color(), alpha=0.25)
                lowest = min(lowest, np.min(a[:, i, j, :]))
                highest = max(highest, np.max(a[:, i, j, :]))

    for i, j, ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
        ax.yaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
        al = ax.get_ylim()
        ax.set_ylim(min(al[0], lowest), max(al[1], highest))
        ax.set_xlim(left, right)

        if 0 < i < m - 1:
            ax.set_xticks([])
        if 0 < j < m - 1:
            ax.set_yticks([])

        if i == 0:
            ax.xaxis.tick_top()
        if i == m-1:
            ax.xaxis.tick_bottom()

        if j == 0:
            ax.yaxis.tick_left()
        if j == m-1:
            ax.yaxis.tick_right()

    _plot_labels(fig,
                 {'x': 0.5, 'y': 0.025, 's': 'frequency (Hz)', 'horizontalalignment': 'center'},
                 {'x': 0.05, 'y': 0.5, 's': 'magnitude', 'horizontalalignment': 'center', 'rotation': 'vertical'})

    return fig


def plot_connectivity_significance(s, fs=2, freq_range=(-np.inf, np.inf), diagonal=0, border=False, fig=None):
    """ Plot significance.

    Significance is drawn as a background image where dark vertical stripes indicate freuquencies where a evaluates to
    True.

    Parameters
    ----------
    a : array, dtype=bool, shape = [n_channels, n_channels, n_fft]
        Significance
    fs : float
        Sampling frequency
    freq_range : (float, float)
        Frequency range to plot
    diagonal : {-1, 0, 1}
        If diagonal == -1 nothing is plotted on the diagonal (a[i,i,:] are not plotted),
        if diagonal == 0, a is plotted on the diagonal too (all a[i,i,:] are plotted),
        if diagonal == 1, a is plotted on the diagonal only (only a[i,i,:] are plotted)
    border : bool
        If border == true the leftmost column and the topmost row are left blank
    fig : Figure object, optional
        Figure to plot into. If set to `None`, a new figure is created.

    Returns
    -------
    fig : Figure object
        The figure into which was plotted.
    """

    a = np.atleast_3d(s)
    [_, m, f] = a.shape
    freq = np.linspace(0, fs / 2, f)

    left = max(freq_range[0], freq[0])
    right = min(freq_range[1], freq[-1])

    imext = (freq[0], freq[-1], -1e25, 1e25)

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(m):
        if diagonal == 1:
            jrange = [i]
        elif diagonal == 0:
            jrange = range(m)
        else:
            jrange = [j for j in range(m) if j != i]
        for j in jrange:
            if border:
                ax = fig.add_subplot(m+1, m+1, j + (i+1) * (m+1) + 2)
            else:
                ax = fig.add_subplot(m, m, j + i * m + 1)
            axes.append((i, j, ax))
            ax.imshow(s[i, j, np.newaxis], vmin=0, vmax=2, cmap=cm.binary, aspect='auto', extent=imext, zorder=-999)

            ax.xaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
            ax.yaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
            ax.set_xlim(left, right)

            if 0 < i < m - 1:
                ax.set_xticks([])
            if 0 < j < m - 1:
                ax.set_yticks([])

            if j == 0:
                ax.yaxis.tick_left()
            if j == m-1:
                ax.yaxis.tick_right()

    _plot_labels(fig,
                 {'x': 0.5, 'y': 0.025, 's': 'frequency (Hz)', 'horizontalalignment': 'center'},
                 {'x': 0.05, 'y': 0.5, 's': 'magnitude', 'horizontalalignment': 'center', 'rotation': 'vertical'})

    return fig


def plot_connectivity_timespectrum(a, fs=2, crange=None, freq_range=(-np.inf, np.inf), time_range=None, diagonal=0, border=False, fig=None):
    """ Draw time/frequency connectivity plots.

    Parameters
    ----------
    a : array, shape = [n_channels, n_channels, n_fft, n_timesteps]
        Values to draw
    fs : float
        Sampling frequency
    crange : [int, int], optional
        Range of values covered by the colormap.
        If set to None, [min(a), max(a)] is substituted.
    freq_range : (float, float)
        Frequency range to plot
    time_range : (float, float)
        Time range covered by `a`
    diagonal : {-1, 0, 1}
        If diagonal == -1 nothing is plotted on the diagonal (a[i,i,:] are not plotted),
        if diagonal == 0, a is plotted on the diagonal too (all a[i,i,:] are plotted),
        if diagonal == 1, a is plotted on the diagonal only (only a[i,i,:] are plotted)
    border : bool
        If border == true the leftmost column and the topmost row are left blank
    fig : Figure object, optional
        Figure to plot into. If set to `None`, a new figure is created.

    Returns
    -------
    fig : Figure object
        The figure into which was plotted.
    """
    a = np.asarray(a)
    [_, m, _, t] = a.shape

    if crange is None:
        crange = [np.min(a), np.max(a)]

    if time_range is None:
        t0 = 0
        t1 = t
    else:
        t0, t1 = time_range

    f0, f1 = fs / 2, 0
    extent = [t0, t1, f0, f1]

    ymin = max(freq_range[0], f1)
    ymax = min(freq_range[1], f0)

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(m):
        if diagonal == 1:
            jrange = [i]
        elif diagonal == 0:
            jrange = range(m)
        else:
            jrange = [j for j in range(m) if j != i]
        for j in jrange:
            if border:
                ax = fig.add_subplot(m+1, m+1, j + (i+1) * (m+1) + 2)
            else:
                ax = fig.add_subplot(m, m, j + i * m + 1)
            axes.append(ax)
            ax.imshow(a[i, j, :, :], vmin=crange[0], vmax=crange[1], aspect='auto', extent=extent)
            ax.invert_yaxis()

            ax.xaxis.set_major_locator(MaxNLocator(max(1, 9 - m)))
            ax.yaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
            ax.set_ylim(ymin, ymax)

            if 0 < i < m - 1:
                ax.set_xticks([])
            if 0 < j < m - 1:
                ax.set_yticks([])

            if i == 0:
                ax.xaxis.tick_top()
            if i == m-1:
                ax.xaxis.tick_bottom()

            if j == 0:
                ax.yaxis.tick_left()
            if j == m-1:
                ax.yaxis.tick_right()

    _plot_labels(fig,
                 {'x': 0.5, 'y': 0.025, 's': 'time (s)', 'horizontalalignment': 'center'},
                 {'x': 0.05, 'y': 0.5, 's': 'frequency (Hz)', 'horizontalalignment': 'center', 'rotation': 'vertical'})

    return fig


def plot_circular(widths, colors, curviness=0.2, mask=True, topo=None, topomaps=None, axes=None, order=None):
    """ Circluar connectivity plot

    Topos are arranged in a circle, with arrows indicating connectivity

    Parameters
    ----------
    widths : {float or array, shape = [n_channels, n_channels]}
        Width of each arrow. Can be a scalar to assign the same width to all arrows.
    colors : array, shape = [n_channels, n_channels, 3] or [3]
        RGB color values for each arrow or one RGB color value for all arrows.
    curviness : float, optional
        Factor that determines how much arrows tend to deviate from a straight line.
    mask : array, dtype = bool, shape = [n_channels, n_channels]
        Enable or disable individual arrows
    topo : :class:`~eegtopo.topoplot.Topoplot`
        This object draws the topo plot
    topomaps : array, shape = [w_pixels, h_pixels]
        Scalp-projected map
    axes : axis, optional
        Axis to draw into. A new figure is created by default.
    order : list of int
        Rearrange channels.

    Returns
    -------
    axes : Axes object
        The axes into which was plotted.
    """
    colors = np.asarray(colors)
    widths = np.asarray(widths)
    mask = np.asarray(mask)

    colors = np.maximum(colors, 0)
    colors = np.minimum(colors, 1)

    if len(widths.shape) > 2:
        [n, m] = widths.shape
    elif len(colors.shape) > 3:
        [n, m, c] = widths.shape
    elif len(mask.shape) > 2:
        [n, m] = mask.shape
    else:
        n = len(topomaps)
        m = n

    if not order:
        order = list(range(n))

    #a = np.asarray(a)
    #[n, m] = a.shape

    assert(n == m)

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.set_frame_on(False)

    if len(colors.shape) < 3:
        colors = np.tile(colors, (n,n,1))

    if len(widths.shape) < 2:
        widths = np.tile(widths, (n,n))

    if len(mask.shape) < 2:
        mask = np.tile(mask, (n,n))
    np.fill_diagonal(mask, False)

    if topo:
        alpha = 1.5 if n < 10 else 1.25
        r = alpha * topo.head_radius / (np.sin(np.pi/n))
    else:
        r = 1

    for i in range(n):
        if topo:
            o = (r*np.sin(i*2*np.pi/n), r*np.cos(i*2*np.pi/n))
            plot_topo(axes, topo, topomaps[order[i]], offset=o)

    for i in range(n):
        for j in range(n):
            if not mask[order[i], order[j]]:
                continue
            a0 = j*2*np.pi/n
            a1 = i*2*np.pi/n

            x0, y0 = r*np.sin(a0), r*np.cos(a0)
            x1, y1 = r*np.sin(a1), r*np.cos(a1)

            ex = (x0 + x1) / 2
            ey = (y0 + y1) / 2
            en = np.sqrt(ex**2 + ey**2)

            if en < 1e-10:
                en = 0
                ex = y0 / r
                ey = -x0 / r
                w = -r
            else:
                ex /= en
                ey /= en
                w = np.sqrt((x1-x0)**2 + (y1-y0)**2) / 2

                if x0*y1-y0*x1 < 0:
                    w = -w

            d = en*(1-curviness)
            h = en-d

            t = np.linspace(-1, 1, 100)

            dist = (t**2+2*t+1)*w**2 + (t**4-2*t**2+1)*h**2

            tmask1 = dist >= (1.4*topo.head_radius)**2
            tmask2 = dist >= (1.2*topo.head_radius)**2
            tmask = np.logical_and(tmask1, tmask2[::-1])
            t = t[tmask]

            x = (h*t*t+d)*ex - w*t*ey
            y = (h*t*t+d)*ey + w*t*ex

            # Arrow Head
            s = np.sqrt((x[-2] - x[-1])**2 + (y[-2] - y[-1])**2)

            width = widths[order[i], order[j]]

            x1 = 0.1*width*(x[-2] - x[-1] + y[-2] - y[-1])/s + x[-1]
            y1 = 0.1*width*(y[-2] - y[-1] - x[-2] + x[-1])/s + y[-1]

            x2 = 0.1*width*(x[-2] - x[-1] - y[-2] + y[-1])/s + x[-1]
            y2 = 0.1*width*(y[-2] - y[-1] + x[-2] - x[-1])/s + y[-1]

            x = np.concatenate([x, [x1, x[-1], x2]])
            y = np.concatenate([y, [y1, y[-1], y2]])
            axes.plot(x, y, lw=width, color=colors[order[i], order[j]], solid_capstyle='round', solid_joinstyle='round')

    return axes


def plot_whiteness(var, h, repeats=1000, axis=None):
    """ Draw distribution of the Portmanteu whiteness test.

    Parameters
    ----------
    var : :class:`~scot.var.VARBase`-like object
        Vector autoregressive model (VAR) object whose residuals are tested for whiteness.
    h : int
        Maximum lag to include in the test.
    repeats : int, optional
        Number of surrogate estimates to draw under the null hypothesis.
    axis : axis, optional
        Axis to draw into. By default draws into :func:`matplotlib.pyplot.gca()`.

    Returns
    -------
    pr : float
        *p*-value of whiteness under the null hypothesis
    """
    pr, q0, q = var.test_whiteness(h, repeats, True)

    if axis is None:
        axis = plt.gca()

    pdf, _, _ = axis.hist(q0, 30, normed=True, label='surrogate distribution')
    axis.plot([q,q], [0,np.max(pdf)], 'r-', label='fitted model')

    #df = m*m*(h-p)
    #x = np.linspace(np.min(q0)*0.0, np.max(q0)*2.0, 100)
    #y = sp.stats.chi2.pdf(x, df)
    #hc = axis.plot(x, y, label='chi-squared distribution (df=%i)' % df)

    axis.set_title('significance: p = %f'%pr)
    axis.set_xlabel('Li-McLeod statistic (Q)')
    axis.set_ylabel('probability')

    axis.legend()

    return pr


def _plot_labels(target, *labels):
    for l in labels:
        have_label = False
        for child in target.get_children():
            try:
                if child.get_text() == l['s'] and child.get_position() == (l['x'], l['y']):
                    have_label = True
                    break
            except AttributeError:
                pass
        if not have_label:
            target.text(**l)
