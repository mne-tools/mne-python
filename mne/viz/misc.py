# -*- coding: utf-8 -*-
"""Functions to make simple plots with M/EEG data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import copy
from glob import glob
from itertools import cycle
import os.path as op
import warnings
from distutils.version import LooseVersion
from collections import defaultdict

import numpy as np
from scipy import linalg

from ..defaults import DEFAULTS
from ..surface import read_surface
from ..io.proj import make_projector
from ..io.pick import _DATA_CH_TYPES_SPLIT, pick_types
from ..source_space import read_source_spaces, SourceSpaces
from ..utils import (logger, verbose, get_subjects_dir, warn, _check_option,
                     _mask_to_onsets_offsets)
from ..io.pick import _picks_by_type
from ..filter import estimate_ringing_samples
from ..fixes import get_sosfiltfilt
from .utils import tight_layout, _get_color_list, _prepare_trellis, plt_show


@verbose
def plot_cov(cov, info, exclude=[], colorbar=True, proj=False, show_svd=True,
             show=True, verbose=None):
    """Plot Covariance data.

    Parameters
    ----------
    cov : instance of Covariance
        The covariance matrix.
    info: dict
        Measurement info.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any channel.
        If 'bads', exclude info['bads'].
    colorbar : bool
        Show colorbar or not.
    proj : bool
        Apply projections or not.
    show_svd : bool
        Plot also singular values of the noise covariance for each sensor
        type. We show square roots ie. standard deviations.
    show : bool
        Show figure if True.
    %(verbose)s

    Returns
    -------
    fig_cov : instance of matplotlib.figure.Figure
        The covariance plot.
    fig_svd : instance of matplotlib.figure.Figure | None
        The SVD spectra plot of the covariance.
    """
    if exclude == 'bads':
        exclude = info['bads']
    picks_list = \
        _picks_by_type(info, meg_combined=False, ref_meg=False,
                       exclude=exclude)
    picks_by_type = dict(picks_list)

    ch_names = [n for n in cov.ch_names if n not in exclude]
    ch_idx = [cov.ch_names.index(n) for n in ch_names]

    info_ch_names = info['ch_names']
    idx_by_type = defaultdict(list)
    for ch_type, sel in picks_by_type.items():
        idx_by_type[ch_type] = [ch_names.index(info_ch_names[c])
                                for c in sel if info_ch_names[c] in ch_names]

    idx_names = [(idx_by_type[key],
                  '%s covariance' % DEFAULTS['titles'][key],
                  DEFAULTS['units'][key],
                  DEFAULTS['scalings'][key])
                 for key in _DATA_CH_TYPES_SPLIT
                 if len(idx_by_type[key]) > 0]
    C = cov.data[ch_idx][:, ch_idx]

    if proj:
        projs = copy.deepcopy(info['projs'])

        #   Activate the projection items
        for p in projs:
            p['active'] = True

        P, ncomp, _ = make_projector(projs, ch_names)
        if ncomp > 0:
            logger.info('    Created an SSP operator (subspace dimension'
                        ' = %d)' % ncomp)
            C = np.dot(P, np.dot(C, P.T))
        else:
            logger.info('    The projection vectors do not apply to these '
                        'channels.')

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig_cov, axes = plt.subplots(1, len(idx_names), squeeze=False,
                                 figsize=(3.8 * len(idx_names), 3.7))
    for k, (idx, name, _, _) in enumerate(idx_names):
        vlim = np.max(np.abs(C[idx][:, idx]))
        im = axes[0, k].imshow(C[idx][:, idx], interpolation="nearest",
                               norm=Normalize(vmin=-vlim, vmax=vlim),
                               cmap='RdBu_r')
        axes[0, k].set(title=name)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axes[0, k])
            cax = divider.append_axes("right", size="5.5%", pad=0.05)
            plt.colorbar(im, cax=cax, format='%.0e')

    fig_cov.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
    tight_layout(fig=fig_cov)

    fig_svd = None
    if show_svd:
        fig_svd, axes = plt.subplots(1, len(idx_names), squeeze=False,
                                     figsize=(3.8 * len(idx_names), 3.7))
        for k, (idx, name, unit, scaling) in enumerate(idx_names):
            s = linalg.svd(C[idx][:, idx], compute_uv=False)
            # Protect against true zero singular values
            s[s <= 0] = 1e-10 * s[s > 0].min()
            s = np.sqrt(s) * scaling
            axes[0, k].plot(s)
            axes[0, k].set(ylabel=u'Noise σ (%s)' % unit, yscale='log',
                           xlabel='Eigenvalue index', title=name)
        tight_layout(fig=fig_svd)

    plt_show(show)

    return fig_cov, fig_svd


def plot_source_spectrogram(stcs, freq_bins, tmin=None, tmax=None,
                            source_index=None, colorbar=False, show=True):
    """Plot source power in time-freqency grid.

    Parameters
    ----------
    stcs : list of SourceEstimate
        Source power for consecutive time windows, one SourceEstimate object
        should be provided for each frequency bin.
    freq_bins : list of tuples of float
        Start and end points of frequency bins of interest.
    tmin : float
        Minimum time instant to show.
    tmax : float
        Maximum time instant to show.
    source_index : int | None
        Index of source for which the spectrogram will be plotted. If None,
        the source with the largest activation will be selected.
    colorbar : bool
        If true, a colorbar will be added to the plot.
    show : bool
        Show figure if True.
    """
    import matplotlib.pyplot as plt

    # Input checks
    if len(stcs) == 0:
        raise ValueError('cannot plot spectrogram if len(stcs) == 0')

    stc = stcs[0]
    if tmin is not None and tmin < stc.times[0]:
        raise ValueError('tmin cannot be smaller than the first time point '
                         'provided in stcs')
    if tmax is not None and tmax > stc.times[-1] + stc.tstep:
        raise ValueError('tmax cannot be larger than the sum of the last time '
                         'point and the time step, which are provided in stcs')

    # Preparing time-frequency cell boundaries for plotting
    if tmin is None:
        tmin = stc.times[0]
    if tmax is None:
        tmax = stc.times[-1] + stc.tstep
    time_bounds = np.arange(tmin, tmax + stc.tstep, stc.tstep)
    freq_bounds = sorted(set(np.ravel(freq_bins)))
    freq_ticks = copy.deepcopy(freq_bounds)

    # Reject time points that will not be plotted and gather results
    source_power = []
    for stc in stcs:
        stc = stc.copy()  # copy since crop modifies inplace
        stc.crop(tmin, tmax - stc.tstep)
        source_power.append(stc.data)
    source_power = np.array(source_power)

    # Finding the source with maximum source power
    if source_index is None:
        source_index = np.unravel_index(source_power.argmax(),
                                        source_power.shape)[1]

    # If there is a gap in the frequency bins record its locations so that it
    # can be covered with a gray horizontal bar
    gap_bounds = []
    for i in range(len(freq_bins) - 1):
        lower_bound = freq_bins[i][1]
        upper_bound = freq_bins[i + 1][0]
        if lower_bound != upper_bound:
            freq_bounds.remove(lower_bound)
            gap_bounds.append((lower_bound, upper_bound))

    # Preparing time-frequency grid for plotting
    time_grid, freq_grid = np.meshgrid(time_bounds, freq_bounds)

    # Plotting the results
    fig = plt.figure(figsize=(9, 6))
    plt.pcolor(time_grid, freq_grid, source_power[:, source_index, :],
               cmap='Reds')
    ax = plt.gca()

    ax.set(title='Source power', xlabel='Time (s)', ylabel='Frequency (Hz)')

    time_tick_labels = [str(np.round(t, 2)) for t in time_bounds]
    n_skip = 1 + len(time_bounds) // 10
    for i in range(len(time_bounds)):
        if i % n_skip != 0:
            time_tick_labels[i] = ''

    ax.set_xticks(time_bounds)
    ax.set_xticklabels(time_tick_labels)
    plt.xlim(time_bounds[0], time_bounds[-1])
    plt.yscale('log')
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels([np.round(freq, 2) for freq in freq_ticks])
    plt.ylim(freq_bounds[0], freq_bounds[-1])

    plt.grid(True, ls='-')
    if colorbar:
        plt.colorbar()
    tight_layout(fig=fig)

    # Covering frequency gaps with horizontal bars
    for lower_bound, upper_bound in gap_bounds:
        plt.barh(lower_bound, time_bounds[-1] - time_bounds[0], upper_bound -
                 lower_bound, time_bounds[0], color='#666666')

    plt_show(show)
    return fig


def _plot_mri_contours(mri_fname, surfaces, src, orientation='coronal',
                       slices=None, show=True):
    """Plot BEM contours on anatomical slices."""
    import matplotlib.pyplot as plt
    import nibabel as nib

    # plot axes (x, y, z) as data axes (0, 1, 2)
    if orientation == 'coronal':
        x, y, z = 0, 1, 2
    elif orientation == 'axial':
        x, y, z = 2, 0, 1
    elif orientation == 'sagittal':
        x, y, z = 2, 1, 0
    else:
        raise ValueError("Orientation must be 'coronal', 'axial' or "
                         "'sagittal'. Got %s." % orientation)

    # Load the T1 data
    nim = nib.load(mri_fname)
    data = nim.get_data()
    try:
        affine = nim.affine
    except AttributeError:  # older nibabel
        affine = nim.get_affine()

    n_sag, n_axi, n_cor = data.shape
    orientation_name2axis = dict(sagittal=0, axial=1, coronal=2)
    orientation_axis = orientation_name2axis[orientation]

    if slices is None:
        n_slices = data.shape[orientation_axis]
        slices = np.linspace(0, n_slices, 12, endpoint=False).astype(np.int)

    # create of list of surfaces
    surfs = list()

    trans = linalg.inv(affine)
    # XXX : next line is a hack don't ask why
    trans[:3, -1] = [n_sag // 2, n_axi // 2, n_cor // 2]

    for file_name, color in surfaces:
        surf = dict()
        surf['rr'], surf['tris'] = read_surface(file_name)
        # move back surface to MRI coordinate system
        surf['rr'] = nib.affines.apply_affine(trans, surf['rr'])
        surfs.append((surf, color))

    src_points = list()
    if isinstance(src, SourceSpaces):
        for src_ in src:
            points = src_['rr'][src_['inuse'].astype(bool)] * 1e3
            src_points.append(nib.affines.apply_affine(trans, points))
    elif src is not None:
        raise TypeError("src needs to be None or SourceSpaces instance, not "
                        "%s" % repr(src))

    fig, axs = _prepare_trellis(len(slices), 4)

    for ax, sl in zip(axs, slices):

        # adjust the orientations for good view
        if orientation == 'coronal':
            dat = data[:, :, sl].transpose()
        elif orientation == 'axial':
            dat = data[:, sl, :]
        elif orientation == 'sagittal':
            dat = data[sl, :, :]

        # First plot the anatomical data
        ax.imshow(dat, cmap=plt.cm.gray)
        ax.set_autoscale_on(False)
        ax.axis('off')

        # and then plot the contours on top
        for surf, color in surfs:
            with warnings.catch_warnings(record=True):  # ignore contour warn
                warnings.simplefilter('ignore')
                ax.tricontour(surf['rr'][:, x], surf['rr'][:, y],
                              surf['tris'], surf['rr'][:, z],
                              levels=[sl], colors=color, linewidths=1.0,
                              zorder=1)

        for sources in src_points:
            in_slice = np.logical_and(sources[:, z] > sl - 0.5,
                                      sources[:, z] < sl + 0.5)
            ax.scatter(sources[in_slice, x], sources[in_slice, y], marker='.',
                       color='#FF00FF', s=1, zorder=2)

    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.,
                        hspace=0.)
    plt_show(show)
    return fig


def plot_bem(subject=None, subjects_dir=None, orientation='coronal',
             slices=None, brain_surfaces=None, src=None, show=True):
    """Plot BEM contours on anatomical slices.

    Parameters
    ----------
    subject : str
        Subject name.
    subjects_dir : str | None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
    orientation : str
        'coronal' or 'axial' or 'sagittal'.
    slices : list of int
        Slice indices.
    brain_surfaces : None | str | list of str
        One or more brain surface to plot (optional). Entries should correspond
        to files in the subject's ``surf`` directory (e.g. ``"white"``).
    src : None | SourceSpaces | str
        SourceSpaces instance or path to a source space to plot individual
        sources as scatter-plot. Only sources lying in the shown slices will be
        visible, sources that lie between visible slices are not shown. Path
        can be absolute or relative to the subject's ``bem`` folder.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.

    See Also
    --------
    mne.viz.plot_alignment
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    # Get the MRI filename
    mri_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    if not op.isfile(mri_fname):
        raise IOError('MRI file "%s" does not exist' % mri_fname)

    # Get the BEM surface filenames
    bem_path = op.join(subjects_dir, subject, 'bem')

    if not op.isdir(bem_path):
        raise IOError('Subject bem directory "%s" does not exist' % bem_path)

    surfaces = []
    for surf_name, color in (('*inner_skull', '#FF0000'),
                             ('*outer_skull', '#FFFF00'),
                             ('*outer_skin', '#FFAA80')):
        surf_fname = glob(op.join(bem_path, surf_name + '.surf'))
        if len(surf_fname) > 0:
            surf_fname = surf_fname[0]
            logger.info("Using surface: %s" % surf_fname)
            surfaces.append((surf_fname, color))

    if brain_surfaces is not None:
        if isinstance(brain_surfaces, str):
            brain_surfaces = (brain_surfaces,)
        for surf_name in brain_surfaces:
            for hemi in ('lh', 'rh'):
                surf_fname = op.join(subjects_dir, subject, 'surf',
                                     hemi + '.' + surf_name)
                if op.exists(surf_fname):
                    surfaces.append((surf_fname, '#00DD00'))
                else:
                    raise IOError("Surface %s does not exist." % surf_fname)

    if isinstance(src, str):
        if not op.exists(src):
            src_ = op.join(subjects_dir, subject, 'bem', src)
            if op.exists(src_):
                src = src_
            else:
                raise IOError("%s does not exist" % src)
        src = read_source_spaces(src)
    elif src is not None and not isinstance(src, SourceSpaces):
        raise TypeError("src needs to be None, str or SourceSpaces instance, "
                        "not %s" % repr(src))

    if len(surfaces) == 0:
        raise IOError('No surface files found. Surface files must end with '
                      'inner_skull.surf, outer_skull.surf or outer_skin.surf')

    # Plot the contours
    return _plot_mri_contours(mri_fname, surfaces, src, orientation, slices,
                              show)


def plot_events(events, sfreq=None, first_samp=0, color=None, event_id=None,
                axes=None, equal_spacing=True, show=True):
    """Plot events to get a visual display of the paradigm.

    Parameters
    ----------
    events : array, shape (n_events, 3)
        The events.
    sfreq : float | None
        The sample frequency. If None, data will be displayed in samples (not
        seconds).
    first_samp : int
        The index of the first sample. Recordings made on Neuromag systems
        number samples relative to the system start (not relative to the
        beginning of the recording). In such cases the ``raw.first_samp``
        attribute can be passed here. Default is 0.
    color : dict | None
        Dictionary of event_id integers as keys and colors as values. If None,
        colors are automatically drawn from a default list (cycled through if
        number of events longer than list of default colors). Color can be any
        valid :doc:`matplotlib color <tutorials/colors/colors>`.
    event_id : dict | None
        Dictionary of event labels (e.g. 'aud_l') as keys and their associated
        event_id values. Labels are used to plot a legend. If None, no legend
        is drawn.
    axes : instance of Axes
       The subplot handle.
    equal_spacing : bool
        Use equal spacing between events in y-axis.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    if sfreq is None:
        sfreq = 1.0
        xlabel = 'Samples'
    else:
        xlabel = 'Time (s)'

    events = np.asarray(events)
    unique_events = np.unique(events[:, 2])

    if event_id is not None:
        # get labels and unique event ids from event_id dict,
        # sorted by value
        event_id_rev = {v: k for k, v in event_id.items()}
        conditions, unique_events_id = zip(*sorted(event_id.items(),
                                                   key=lambda x: x[1]))

        for this_event in unique_events_id:
            if this_event not in unique_events:
                raise ValueError('%s from event_id is not present in events.'
                                 % this_event)

        for this_event in unique_events:
            if this_event not in unique_events_id:
                warn('event %s missing from event_id will be ignored'
                     % this_event)
    else:
        unique_events_id = unique_events

    color = _handle_event_colors(unique_events, color, unique_events_id)
    import matplotlib.pyplot as plt

    fig = None
    if axes is None:
        fig = plt.figure()
    ax = axes if axes else plt.gca()

    unique_events_id = np.array(unique_events_id)
    min_event = np.min(unique_events_id)
    max_event = np.max(unique_events_id)

    for idx, ev in enumerate(unique_events_id):
        ev_mask = events[:, 2] == ev
        kwargs = {}
        if event_id is not None:
            event_label = '{} ({})'.format(event_id_rev[ev], np.sum(ev_mask))
            kwargs['label'] = event_label
        if ev in color:
            kwargs['color'] = color[ev]
        if equal_spacing:
            ax.plot((events[ev_mask, 0] - first_samp) / sfreq,
                    (idx + 1) * np.ones(ev_mask.sum()), '.', **kwargs)
        else:
            ax.plot((events[ev_mask, 0] - first_samp) / sfreq,
                    events[ev_mask, 2], '.', **kwargs)

    if equal_spacing:
        ax.set_ylim(0, unique_events_id.size + 1)
        ax.set_yticks(1 + np.arange(unique_events_id.size))
        ax.set_yticklabels(unique_events_id)
    else:
        ax.set_ylim([min_event - 1, max_event + 1])

    ax.set(xlabel=xlabel, ylabel='Events id')

    ax.grid(True)

    fig = fig if fig is not None else plt.gcf()
    if event_id is not None:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.canvas.draw()
    plt_show(show)
    return fig


def _get_presser(fig):
    """Get our press callback."""
    import matplotlib
    callbacks = fig.canvas.callbacks.callbacks['button_press_event']
    func = None
    for key, val in callbacks.items():
        if LooseVersion(matplotlib.__version__) >= '3':
            func = val()
        else:
            func = val.func
        if func.__class__.__name__ == 'partial':
            break
        else:
            func = None
    assert func is not None
    return func


def plot_dipole_amplitudes(dipoles, colors=None, show=True):
    """Plot the amplitude traces of a set of dipoles.

    Parameters
    ----------
    dipoles : list of instance of Dipole
        The dipoles whose amplitudes should be shown.
    colors: list of color | None
        Color to plot with each dipole. If None default colors are used.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    import matplotlib.pyplot as plt
    if colors is None:
        colors = cycle(_get_color_list())
    fig, ax = plt.subplots(1, 1)
    xlim = [np.inf, -np.inf]
    for dip, color in zip(dipoles, colors):
        ax.plot(dip.times, dip.amplitude * 1e9, color=color, linewidth=1.5)
        xlim[0] = min(xlim[0], dip.times[0])
        xlim[1] = max(xlim[1], dip.times[-1])
    ax.set(xlim=xlim, xlabel='Time (s)', ylabel='Amplitude (nAm)')
    if show:
        fig.show(warn=False)
    return fig


def adjust_axes(axes, remove_spines=('top', 'right'), grid=True):
    """Adjust some properties of axes.

    Parameters
    ----------
    axes : list
        List of axes to process.
    remove_spines : list of str
        Which axis spines to remove.
    grid : bool
        Turn grid on (True) or off (False).
    """
    axes = [axes] if not isinstance(axes, (list, tuple, np.ndarray)) else axes
    for ax in axes:
        if grid:
            ax.grid(zorder=0)
        for key in remove_spines:
            ax.spines[key].set_visible(False)


def _filter_ticks(lims, fscale):
    """Create approximately spaced ticks between lims."""
    if fscale == 'linear':
        return None, None  # let matplotlib handle it
    lims = np.array(lims)
    ticks = list()
    if lims[1] > 20 * lims[0]:
        base = np.array([1, 2, 4])
    else:
        base = np.arange(1, 11)
    for exp in range(int(np.floor(np.log10(lims[0]))),
                     int(np.floor(np.log10(lims[1]))) + 1):
        ticks += (base * (10 ** exp)).tolist()
    ticks = np.array(ticks)
    ticks = ticks[(ticks >= lims[0]) & (ticks <= lims[1])]
    ticklabels = [('%g' if t < 1 else '%d') % t for t in ticks]
    return ticks, ticklabels


def _get_flim(flim, fscale, freq, sfreq=None):
    """Get reasonable frequency limits."""
    if flim is None:
        if freq is None:
            flim = [0.1 if fscale == 'log' else 0., sfreq / 2.]
        else:
            if fscale == 'linear':
                flim = [freq[0]]
            else:
                flim = [freq[0] if freq[0] > 0 else 0.1 * freq[1]]
            flim += [freq[-1]]
    if fscale == 'log':
        if flim[0] <= 0:
            raise ValueError('flim[0] must be positive, got %s' % flim[0])
    elif flim[0] < 0:
        raise ValueError('flim[0] must be non-negative, got %s' % flim[0])
    return flim


def _check_fscale(fscale):
    """Check for valid fscale."""
    if not isinstance(fscale, str) or fscale not in ('log', 'linear'):
        raise ValueError('fscale must be "log" or "linear", got %s'
                         % (fscale,))


_DEFAULT_ALIM = (-80, 10)


def plot_filter(h, sfreq, freq=None, gain=None, title=None, color='#1f77b4',
                flim=None, fscale='log', alim=_DEFAULT_ALIM, show=True,
                compensate=False):
    """Plot properties of a filter.

    Parameters
    ----------
    h : dict or ndarray
        An IIR dict or 1D ndarray of coefficients (for FIR filter).
    sfreq : float
        Sample rate of the data (Hz).
    freq : array-like or None
        The ideal response frequencies to plot (must be in ascending order).
        If None (default), do not plot the ideal response.
    gain : array-like or None
        The ideal response gains to plot.
        If None (default), do not plot the ideal response.
    title : str | None
        The title to use. If None (default), deteremine the title based
        on the type of the system.
    color : color object
        The color to use (default '#1f77b4').
    flim : tuple or None
        If not None, the x-axis frequency limits (Hz) to use.
        If None, freq will be used. If None (default) and freq is None,
        ``(0.1, sfreq / 2.)`` will be used.
    fscale : str
        Frequency scaling to use, can be "log" (default) or "linear".
    alim : tuple
        The y-axis amplitude limits (dB) to use (default: (-60, 10)).
    show : bool
        Show figure if True (default).
    compensate : bool
        If True, compensate for the filter delay (phase will not be shown).

        - For linear-phase FIR filters, this visualizes the filter coefficients
          assuming that the output will be shifted by ``N // 2``.
        - For IIR filters, this changes the filter coefficient display
          by filtering backward and forward, and the frequency response
          by squaring it.

        .. versionadded:: 0.18

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.

    See Also
    --------
    mne.filter.create_filter
    plot_ideal_filter

    Notes
    -----
    .. versionadded:: 0.14
    """
    from scipy.signal import freqz, group_delay, lfilter, filtfilt, sosfilt
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, NullFormatter
    sosfiltfilt = get_sosfiltfilt()
    sfreq = float(sfreq)
    _check_option('fscale', fscale, ['log', 'linear'])
    flim = _get_flim(flim, fscale, freq, sfreq)
    if fscale == 'log':
        omega = np.logspace(np.log10(flim[0]), np.log10(flim[1]), 1000)
    else:
        omega = np.linspace(flim[0], flim[1], 1000)
    omega /= sfreq / (2 * np.pi)
    if isinstance(h, dict):  # IIR h.ndim == 2:  # second-order sections
        if 'sos' in h:
            H = np.ones(len(omega), np.complex128)
            gd = np.zeros(len(omega))
            for section in h['sos']:
                this_H = freqz(section[:3], section[3:], omega)[1]
                H *= this_H
                if compensate:
                    H *= this_H.conj()  # time reversal is freq conj
                else:
                    # Assume the forward-backward delay zeros out, which it
                    # mostly should
                    with warnings.catch_warnings(record=True):  # singular GD
                        warnings.simplefilter('ignore')
                        gd += group_delay((section[:3], section[3:]), omega)[1]
            n = estimate_ringing_samples(h['sos'])
            delta = np.zeros(n)
            delta[0] = 1
            if compensate:
                delta = np.pad(delta, [(n - 1, 0)], 'constant')
                func = sosfiltfilt
                gd += (len(delta) - 1) // 2
            else:
                func = sosfilt
            h = func(h['sos'], delta)
        else:
            H = freqz(h['b'], h['a'], omega)[1]
            if compensate:
                H *= H.conj()
            with warnings.catch_warnings(record=True):  # singular GD
                warnings.simplefilter('ignore')
                gd = group_delay((h['b'], h['a']), omega)[1]
                if compensate:
                    gd += group_delay(h['b'].conj(), h['a'].conj(), omega)[1]
            n = estimate_ringing_samples((h['b'], h['a']))
            delta = np.zeros(n)
            delta[0] = 1
            if compensate:
                delta = np.pad(delta, [(n - 1, 0)], 'constant')
                func = filtfilt
            else:
                func = lfilter
            h = func(h['b'], h['a'], delta)
        if title is None:
            title = 'SOS (IIR) filter'
        if compensate:
            title += ' (forward-backward)'
    else:
        H = freqz(h, worN=omega)[1]
        with warnings.catch_warnings(record=True):  # singular GD
            warnings.simplefilter('ignore')
            gd = group_delay((h, [1.]), omega)[1]
        title = 'FIR filter' if title is None else title
        if compensate:
            title += ' (delay-compensated)'
    # eventually axes could be a parameter
    fig, (ax_time, ax_freq, ax_delay) = plt.subplots(3)
    t = np.arange(len(h))
    if compensate:
        n_shift = (len(h) - 1) // 2
        t -= n_shift
        assert t[0] == -t[-1]
        gd -= n_shift
    t = t / sfreq
    gd = gd / sfreq
    f = omega * sfreq / (2 * np.pi)
    ax_time.plot(t, h, color=color)
    ax_time.set(xlim=t[[0, -1]], xlabel='Time (s)',
                ylabel='Amplitude', title=title)
    mag = 10 * np.log10(np.maximum((H * H.conj()).real, 1e-20))
    sl = slice(0 if fscale == 'linear' else 1, None, None)
    # Magnitude
    ax_freq.plot(f[sl], mag[sl], color=color, linewidth=2, zorder=4)
    if freq is not None and gain is not None:
        plot_ideal_filter(freq, gain, ax_freq, fscale=fscale, show=False)
    ax_freq.set(ylabel='Magnitude (dB)', xlabel='', xscale=fscale)
    # Delay
    ax_delay.plot(f[sl], gd[sl], color=color, linewidth=2, zorder=4)
    # shade nulled regions
    for start, stop in zip(*_mask_to_onsets_offsets(mag <= -39.9)):
        ax_delay.axvspan(f[start], f[stop - 1], facecolor='k', alpha=0.05,
                         zorder=5)
    ax_delay.set(xlim=flim, ylabel='Group delay (s)', xlabel='Frequency (Hz)',
                 xscale=fscale)
    xticks, xticklabels = _filter_ticks(flim, fscale)
    dlim = np.abs(t).max() / 2.
    dlim = [-dlim, dlim]
    for ax, ylim, ylabel in ((ax_freq, alim, 'Amplitude (dB)'),
                             (ax_delay, dlim, 'Delay (s)')):
        if xticks is not None:
            ax.set(xticks=xticks)
            ax.set(xticklabels=xticklabels)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(xlim=flim, ylim=ylim, xlabel='Frequency (Hz)', ylabel=ylabel)
    adjust_axes([ax_time, ax_freq, ax_delay])
    tight_layout()
    plt_show(show)
    return fig


def plot_ideal_filter(freq, gain, axes=None, title='', flim=None, fscale='log',
                      alim=_DEFAULT_ALIM, color='r', alpha=0.5, linestyle='--',
                      show=True):
    """Plot an ideal filter response.

    Parameters
    ----------
    freq : array-like
        The ideal response frequencies to plot (must be in ascending order).
    gain : array-like or None
        The ideal response gains to plot.
    axes : instance of Axes | None
        The subplot handle. With None (default), axes are created.
    title : str
        The title to use, (default: '').
    flim : tuple or None
        If not None, the x-axis frequency limits (Hz) to use.
        If None (default), freq used.
    fscale : str
        Frequency scaling to use, can be "log" (default) or "linear".
    alim : tuple
        If not None (default), the y-axis limits (dB) to use.
    color : color object
        The color to use (default: 'r').
    alpha : float
        The alpha to use (default: 0.5).
    linestyle : str
        The line style to use (default: '--').
    show : bool
        Show figure if True (default).

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.

    See Also
    --------
    plot_filter

    Notes
    -----
    .. versionadded:: 0.14

    Examples
    --------
    Plot a simple ideal band-pass filter::

        >>> from mne.viz import plot_ideal_filter
        >>> freq = [0, 1, 40, 50]
        >>> gain = [0, 1, 1, 0]
        >>> plot_ideal_filter(freq, gain, flim=(0.1, 100))  #doctest: +ELLIPSIS
        <...Figure...>

    """
    import matplotlib.pyplot as plt
    my_freq, my_gain = list(), list()
    if freq[0] != 0:
        raise ValueError('freq should start with DC (zero) and end with '
                         'Nyquist, but got %s for DC' % (freq[0],))
    freq = np.array(freq)
    # deal with semilogx problems @ x=0
    _check_option('fscale', fscale, ['log', 'linear'])
    if fscale == 'log':
        freq[0] = 0.1 * freq[1] if flim is None else min(flim[0], freq[1])
    flim = _get_flim(flim, fscale, freq)
    transitions = list()
    for ii in range(len(freq)):
        if ii < len(freq) - 1 and gain[ii] != gain[ii + 1]:
            transitions += [[freq[ii], freq[ii + 1]]]
            my_freq += np.linspace(freq[ii], freq[ii + 1], 20,
                                   endpoint=False).tolist()
            my_gain += np.linspace(gain[ii], gain[ii + 1], 20,
                                   endpoint=False).tolist()
        else:
            my_freq.append(freq[ii])
            my_gain.append(gain[ii])
    my_gain = 10 * np.log10(np.maximum(my_gain, 10 ** (alim[0] / 10.)))
    if axes is None:
        axes = plt.subplots(1)[1]
    for transition in transitions:
        axes.axvspan(*transition, color=color, alpha=0.1)
    axes.plot(my_freq, my_gain, color=color, linestyle=linestyle, alpha=0.5,
              linewidth=4, zorder=3)
    xticks, xticklabels = _filter_ticks(flim, fscale)
    axes.set(ylim=alim, xlabel='Frequency (Hz)', ylabel='Amplitude (dB)',
             xscale=fscale)
    if xticks is not None:
        axes.set(xticks=xticks)
        axes.set(xticklabels=xticklabels)
    axes.set(xlim=flim)
    if title:
        axes.set(title=title)
    adjust_axes(axes)
    tight_layout()
    plt_show(show)
    return axes.figure


def _handle_event_colors(unique_events, color, unique_events_id):
    """Handle event colors."""
    if color is None:
        if len(unique_events) > len(_get_color_list()):
            warn('More events than colors available. You should pass a list '
                 'of unique colors.')
        colors = cycle(_get_color_list())
        color = dict()
        for this_event, this_color in zip(sorted(unique_events_id), colors):
            color[this_event] = this_color
    else:
        for this_event in color:
            if this_event not in unique_events_id:
                raise ValueError('Event ID %s is in the color dict but is not '
                                 'present in events or event_id.' % this_event)

        for this_event in unique_events_id:
            if this_event not in color:
                warn('Color is not available for event %d. Default colors '
                     'will be used.' % this_event)
    return color


def plot_csd(csd, info=None, mode='csd', colorbar=True, cmap=None,
             n_cols=None, show=True):
    """Plot CSD matrices.

    A sub-plot is created for each frequency. If an info object is passed to
    the function, different channel types are plotted in different figures.

    Parameters
    ----------
    csd : instance of CrossSpectralDensity
        The CSD matrix to plot.
    info: instance of Info | None
        To split the figure by channel-type, provide the measurement info.
        By default, the CSD matrix is plotted as a whole.
    mode : 'csd' | 'coh'
        Whether to plot the cross-spectral density ('csd', the default), or
        the coherence ('coh') between the channels.
    colorbar : bool
        Whether to show a colorbar. Defaults to ``True``.
    cmap : str | None
        The matplotlib colormap to use. Defaults to None, which means the
        colormap will default to matplotlib's default.
    n_cols : int | None
        CSD matrices are plotted in a grid. This parameter controls how
        many matrix to plot side by side before starting a new row. By
        default, a number will be chosen to make the grid as square as
        possible.
    show : bool
        Whether to show the figure. Defaults to ``True``.

    Returns
    -------
    fig : list of Figure
        The figures created by this function.
    """
    import matplotlib.pyplot as plt

    if mode not in ['csd', 'coh']:
        raise ValueError('"mode" should be either "csd" or "coh".')

    if info is not None:
        info_ch_names = info['ch_names']
        sel_eeg = pick_types(info, meg=False, eeg=True, ref_meg=False,
                             exclude=[])
        sel_mag = pick_types(info, meg='mag', eeg=False, ref_meg=False,
                             exclude=[])
        sel_grad = pick_types(info, meg='grad', eeg=False, ref_meg=False,
                              exclude=[])
        idx_eeg = [csd.ch_names.index(info_ch_names[c])
                   for c in sel_eeg if info_ch_names[c] in csd.ch_names]
        idx_mag = [csd.ch_names.index(info_ch_names[c])
                   for c in sel_mag if info_ch_names[c] in csd.ch_names]
        idx_grad = [csd.ch_names.index(info_ch_names[c])
                    for c in sel_grad if info_ch_names[c] in csd.ch_names]
        indices = [idx_eeg, idx_mag, idx_grad]
        titles = ['EEG', 'Magnetometers', 'Gradiometers']

        if mode == 'csd':
            # The units in which to plot the CSD
            units = dict(eeg=u'μV²', grad=u'fT²/cm²', mag=u'fT²')
            scalings = dict(eeg=1e12, grad=1e26, mag=1e30)
    else:
        indices = [np.arange(len(csd.ch_names))]
        if mode == 'csd':
            titles = ['Cross-spectral density']
            # Units and scaling unknown
            units = dict()
            scalings = dict()
        elif mode == 'coh':
            titles = ['Coherence']

    n_freqs = len(csd.frequencies)

    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_freqs)))
    n_rows = int(np.ceil(n_freqs / float(n_cols)))

    figs = []
    for ind, title, ch_type in zip(indices, titles, ['eeg', 'mag', 'grad']):
        if len(ind) == 0:
            continue

        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                                 figsize=(2 * n_cols + 1, 2.2 * n_rows))

        csd_mats = []
        for i in range(len(csd.frequencies)):
            cm = csd.get_data(index=i)[ind][:, ind]
            if mode == 'csd':
                cm = np.abs(cm) * scalings.get(ch_type, 1)
            elif mode == 'coh':
                # Compute coherence from the CSD matrix
                psd = np.diag(cm).real
                cm = np.abs(cm) ** 2 / psd[np.newaxis, :] / psd[:, np.newaxis]
            csd_mats.append(cm)

        vmax = np.max(csd_mats)

        for i, (freq, mat) in enumerate(zip(csd.frequencies, csd_mats)):
            ax = axes[i // n_cols][i % n_cols]
            im = ax.imshow(mat, interpolation='nearest', cmap=cmap, vmin=0,
                           vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if csd._is_sum:
                ax.set_title('%.1f-%.1f Hz.' % (np.min(freq),
                                                np.max(freq)))
            else:
                ax.set_title('%.1f Hz.' % freq)

        plt.suptitle(title)
        plt.subplots_adjust(top=0.8)

        if colorbar:
            cb = plt.colorbar(im, ax=[a for ax_ in axes for a in ax_])
            if mode == 'csd':
                label = u'CSD'
                if ch_type in units:
                    label += u' (%s)' % units[ch_type]
                cb.set_label(label)
            elif mode == 'coh':
                cb.set_label('Coherence')

        figs.append(fig)

    plt_show(show)
    return figs
