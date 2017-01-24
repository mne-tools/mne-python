"""Functions to make simple plots with M/EEG data."""

from __future__ import print_function

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

import numpy as np
from scipy import linalg

from ..surface import read_surface
from ..externals.six import string_types
from ..io.proj import make_projector
from ..source_space import read_source_spaces, SourceSpaces
from ..utils import logger, verbose, get_subjects_dir, warn
from ..io.pick import pick_types
from ..filter import estimate_ringing_samples
from .utils import tight_layout, COLORS, _prepare_trellis, plt_show


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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig_cov : instance of matplotlib.pyplot.Figure
        The covariance plot.
    fig_svd : instance of matplotlib.pyplot.Figure | None
        The SVD spectra plot of the covariance.
    """
    if exclude == 'bads':
        exclude = info['bads']
    ch_names = [n for n in cov.ch_names if n not in exclude]
    ch_idx = [cov.ch_names.index(n) for n in ch_names]
    info_ch_names = info['ch_names']
    sel_eeg = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude=exclude)
    sel_mag = pick_types(info, meg='mag', eeg=False, ref_meg=False,
                         exclude=exclude)
    sel_grad = pick_types(info, meg='grad', eeg=False, ref_meg=False,
                          exclude=exclude)
    idx_eeg = [ch_names.index(info_ch_names[c])
               for c in sel_eeg if info_ch_names[c] in ch_names]
    idx_mag = [ch_names.index(info_ch_names[c])
               for c in sel_mag if info_ch_names[c] in ch_names]
    idx_grad = [ch_names.index(info_ch_names[c])
                for c in sel_grad if info_ch_names[c] in ch_names]

    idx_names = [(idx_eeg, 'EEG covariance', 'uV', 1e6),
                 (idx_grad, 'Gradiometers', 'fT/cm', 1e13),
                 (idx_mag, 'Magnetometers', 'fT', 1e15)]
    idx_names = [(idx, name, unit, scaling)
                 for idx, name, unit, scaling in idx_names if len(idx) > 0]

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

    fig_cov, axes = plt.subplots(1, len(idx_names), squeeze=False,
                                 figsize=(2.5 * len(idx_names), 2.7))
    for k, (idx, name, _, _) in enumerate(idx_names):
        axes[0, k].imshow(C[idx][:, idx], interpolation="nearest",
                          cmap='RdBu_r')
        axes[0, k].set(title=name)
    fig_cov.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
    tight_layout(fig=fig_cov)

    fig_svd = None
    if show_svd:
        fig_svd, axes = plt.subplots(1, len(idx_names), squeeze=False)
        for k, (idx, name, unit, scaling) in enumerate(idx_names):
            s = linalg.svd(C[idx][:, idx], compute_uv=False)
            # Protect against true zero singular values
            s[s <= 0] = 1e-10 * s[s > 0].min()
            s = np.sqrt(s) * scaling
            axes[0, k].plot(s)
            axes[0, k].set(ylabel='Noise std (%s)' % unit, yscale='log',
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

    plt.title('Time-frequency source power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

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
    """Plot BEM contours on anatomical slices.

    Parameters
    ----------
    mri_fname : str
        The name of the file containing anatomical data.
    surfaces : list of (str, str) tuples
        A list containing the BEM surfaces to plot as (filename, color) tuples.
        Colors should be matplotlib-compatible.
    src : None | SourceSpaces
        SourceSpaces object for plotting individual sources.
    orientation : str
        'coronal' or 'axial' or 'sagittal'
    slices : list of int
        Slice indices.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
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
            ax.tricontour(surf['rr'][:, x], surf['rr'][:, y],
                          surf['tris'], surf['rr'][:, z],
                          levels=[sl], colors=color, linewidths=2.0,
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
    fig : Instance of matplotlib.figure.Figure
        The figure.
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
        if isinstance(brain_surfaces, string_types):
            brain_surfaces = (brain_surfaces,)
        for surf_name in brain_surfaces:
            for hemi in ('lh', 'rh'):
                surf_fname = op.join(subjects_dir, subject, 'surf',
                                     hemi + '.' + surf_name)
                if op.exists(surf_fname):
                    surfaces.append((surf_fname, '#00DD00'))
                else:
                    raise IOError("Surface %s does not exist." % surf_fname)

    if isinstance(src, string_types):
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
        The index of the first sample. Typically the raw.first_samp
        attribute. It is needed for recordings on a Neuromag
        system as the events are defined relative to the system
        start and not to the beginning of the recording.
    color : dict | None
        Dictionary of event_id value and its associated color. If None,
        colors are automatically drawn from a default list (cycled through if
        number of events longer than list of default colors).
    event_id : dict | None
        Dictionary of event label (e.g. 'aud_l') and its associated
        event_id value. Label used to plot a legend. If None, no legend is
        drawn.
    axes : instance of matplotlib.axes.AxesSubplot
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
        xlabel = 'samples'
    else:
        xlabel = 'Time (s)'

    events = np.asarray(events)
    unique_events = np.unique(events[:, 2])

    if event_id is not None:
        # get labels and unique event ids from event_id dict,
        # sorted by value
        event_id_rev = dict((v, k) for k, v in event_id.items())
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
            event_label = '{0} ({1})'.format(event_id_rev[ev],
                                             np.sum(ev_mask))
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Events id')

    ax.grid('on')

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
    callbacks = fig.canvas.callbacks.callbacks['button_press_event']
    func = None
    for key, val in callbacks.items():
        if val.func.__class__.__name__ == 'partial':
            func = val.func
            break
    assert func is not None
    return func


def plot_dipole_amplitudes(dipoles, colors=None, show=True):
    """Plot the amplitude traces of a set of dipoles.

    Parameters
    ----------
    dipoles : list of instance of Dipoles
        The dipoles whose amplitudes should be shown.
    colors: list of colors | None
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
        colors = cycle(COLORS)
    fig, ax = plt.subplots(1, 1)
    xlim = [np.inf, -np.inf]
    for dip, color in zip(dipoles, colors):
        ax.plot(dip.times, dip.amplitude, color=color, linewidth=1.5)
        xlim[0] = min(xlim[0], dip.times[0])
        xlim[1] = max(xlim[1], dip.times[-1])
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Amplitude (nAm)')
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
    for exp in range(int(np.floor(np.log10(lims[0]))),
                     int(np.floor(np.log10(lims[1]))) + 1):
        ticks += (np.array([1, 2, 4]) * (10 ** exp)).tolist()
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
    if not isinstance(fscale, string_types) or fscale not in ('log', 'linear'):
        raise ValueError('fscale must be "log" or "linear", got %s'
                         % (fscale,))


def plot_filter(h, sfreq, freq=None, gain=None, title=None, color='#1f77b4',
                flim=None, fscale='log', alim=(-60, 10), show=True):
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
    from scipy.signal import freqz, group_delay
    import matplotlib.pyplot as plt
    sfreq = float(sfreq)
    _check_fscale(fscale)
    flim = _get_flim(flim, fscale, freq, sfreq)
    if fscale == 'log':
        omega = np.logspace(np.log10(flim[0]), np.log10(flim[1]), 1000)
    else:
        omega = np.linspace(flim[0], flim[1], 1000)
    omega /= sfreq / (2 * np.pi)
    if isinstance(h, dict):  # IIR h.ndim == 2:  # second-order sections
        if 'sos' in h:
            from scipy.signal import sosfilt
            h = h['sos']
            H = np.ones(len(omega), np.complex128)
            gd = np.zeros(len(omega))
            for section in h:
                this_H = freqz(section[:3], section[3:], omega)[1]
                H *= this_H
                with warnings.catch_warnings(record=True):  # singular GD
                    gd += group_delay((section[:3], section[3:]), omega)[1]
            n = estimate_ringing_samples(h)
            delta = np.zeros(n)
            delta[0] = 1
            h = sosfilt(h, delta)
        else:
            from scipy.signal import lfilter
            n = estimate_ringing_samples((h['b'], h['a']))
            delta = np.zeros(n)
            delta[0] = 1
            H = freqz(h['b'], h['a'], omega)[1]
            with warnings.catch_warnings(record=True):  # singular GD
                gd = group_delay((h['b'], h['a']), omega)[1]
            h = lfilter(h['b'], h['a'], delta)
        title = 'SOS (IIR) filter' if title is None else title
    else:
        H = freqz(h, worN=omega)[1]
        with warnings.catch_warnings(record=True):  # singular GD
            gd = group_delay((h, [1.]), omega)[1]
        title = 'FIR filter' if title is None else title
    gd /= sfreq
    fig, axes = plt.subplots(3)  # eventually axes could be a parameter
    t = np.arange(len(h)) / sfreq
    f = omega * sfreq / (2 * np.pi)
    axes[0].plot(t, h, color=color)
    axes[0].set(xlim=t[[0, -1]], xlabel='Time (sec)',
                ylabel='Amplitude h(n)', title=title)
    mag = 10 * np.log10(np.maximum((H * H.conj()).real, 1e-20))
    axes[1].plot(f, mag, color=color, linewidth=2, zorder=4)
    if freq is not None and gain is not None:
        plot_ideal_filter(freq, gain, axes[1], fscale=fscale,
                          title=None, show=False)
    axes[1].set(ylabel='Magnitude (dB)', xlabel='', xscale=fscale)
    sl = slice(0 if fscale == 'linear' else 1, None, None)
    axes[2].plot(f[sl], gd[sl], color=color, linewidth=2, zorder=4)
    axes[2].set(xlim=flim, ylabel='Group delay (sec)', xlabel='Frequency (Hz)',
                xscale=fscale)
    xticks, xticklabels = _filter_ticks(flim, fscale)
    dlim = [0, 1.05 * gd[1:].max()]
    for ax, ylim, ylabel in zip(axes[1:], (alim, dlim),
                                ('Amplitude (dB)', 'Delay (sec)')):
        if xticks is not None:
            ax.set(xticks=xticks)
            ax.set(xticklabels=xticklabels)
        ax.set(xlim=flim, ylim=ylim, xlabel='Frequency (Hz)', ylabel=ylabel)
    adjust_axes(axes)
    tight_layout()
    plt_show(show)
    return fig


def plot_ideal_filter(freq, gain, axes=None, title='', flim=None, fscale='log',
                      alim=(-60, 10), color='r', alpha=0.5, linestyle='--',
                      show=True):
    """Plot an ideal filter response.

    Parameters
    ----------
    freq : array-like
        The ideal response frequencies to plot (must be in ascending order).
    gain : array-like or None
        The ideal response gains to plot.
    axes : instance of matplotlib.axes.AxesSubplot | None
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
    fig : Instance of matplotlib.figure.Figure
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
        <matplotlib.figure.Figure object at ...>

    """
    import matplotlib.pyplot as plt
    xs, ys = list(), list()
    my_freq, my_gain = list(), list()
    if freq[0] != 0:
        raise ValueError('freq should start with DC (zero) and end with '
                         'Nyquist, but got %s for DC' % (freq[0],))
    freq = np.array(freq)
    # deal with semilogx problems @ x=0
    _check_fscale(fscale)
    if fscale == 'log':
        freq[0] = 0.1 * freq[1] if flim is None else min(flim[0], freq[1])
    flim = _get_flim(flim, fscale, freq)
    for ii in range(len(freq)):
        xs.append(freq[ii])
        ys.append(alim[0])
        if ii < len(freq) - 1 and gain[ii] != gain[ii + 1]:
            xs += [freq[ii], freq[ii + 1]]
            ys += [alim[1]] * 2
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
    xs = np.maximum(xs, flim[0])
    axes.fill_between(xs, alim[0], ys, color=color, alpha=0.1)
    axes.plot(my_freq, my_gain, color=color, linestyle=linestyle, alpha=0.5,
              linewidth=4, zorder=3)
    xticks, xticklabels = _filter_ticks(flim, fscale)
    axes.set(ylim=alim, xlabel='Frequency (Hz)', ylabel='Amplitude (dB)',
             xscale=fscale)
    if xticks is not None:
        axes.set(xticks=xticks)
        axes.set(xticklabels=xticklabels)
    axes.set(xlim=flim)
    adjust_axes(axes)
    tight_layout()
    plt_show(show)
    return axes.figure


def _handle_event_colors(unique_events, color, unique_events_id):
    """Function for handling event colors."""
    if color is None:
        if len(unique_events) > len(COLORS):
            warn('More events than colors available. You should pass a list '
                 'of unique colors.')
        colors = cycle(COLORS)
        color = dict()
        for this_event, this_color in zip(sorted(unique_events_id), colors):
            color[this_event] = this_color
    else:
        for this_event in color:
            if this_event not in unique_events_id:
                raise ValueError('%s from color is not present in events '
                                 'or event_id.' % this_event)

        for this_event in unique_events_id:
            if this_event not in color:
                warn('Color is not available for event %d. Default colors '
                     'will be used.' % this_event)
    return color
